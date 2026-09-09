import assert from "node:assert/strict";
import test from "node:test";
import { generateKeyPair, exportJWK, SignJWT } from "jose";
import worker, { consumeVenueBudget } from "../worker/index.js";
import { allowedPhoto, createSession, currentUser, digest, signPhotoUrls, verifyGoogleToken } from "../worker/auth.js";
import { createDb, testCookie, testEnvironment, testUser } from "./db.mjs";

const { publicKey, privateKey } = await generateKeyPair("RS256");
const publicJwk = { ...await exportJWK(publicKey), kid: "test-google", alg: "RS256" };
const user = { id: "ordinary-account", email: "ordinary@example.test", name: "", privileged: false };
const req = (path, cookie = testCookie, init = {}) => new Request(`https://example.test${path}`, {
  ...init, headers: { Cookie: cookie, Origin: "https://example.test", ...init.headers }
});
async function token(claims = {}, key = privateKey) {
  return new SignJWT({ nonce: "expected-nonce", email: "person@example.test", email_verified: true, ...claims })
    .setProtectedHeader({ alg: "RS256", kid: "test-google" }).setIssuer("https://accounts.google.com")
    .setAudience("test-client").setSubject("google-account-123").setIssuedAt().setExpirationTime("5m").sign(key);
}

test("password access is removed and anonymous requests cannot read protected assets or APIs", async () => {
  const env = testEnvironment();
  for (const path of ["/", "/model/atlas.json", "/assets/DeveloperMode-test.js"]) {
    const response = await worker.fetch(req(path, "equidistant_access=legacy-password-token"), env);
    assert.match(await response.text(), /Sign in with Google/);
    assert.match(response.headers.get("Cache-Control"), /no-store/);
  }
  for (const path of ["/api/session", "/api/usage", "/api/comparison-surface", "/api/venue-recommendations"]) {
    assert.equal((await worker.fetch(req(path, ""), env)).status, 401);
  }
  const unlocked = await worker.fetch(req("/unlock", "", { method: "POST", body: "password=anything" }), env);
  assert.equal(unlocked.headers.has("Set-Cookie"), false);
  assert.doesNotMatch(await unlocked.text(), /type="password"/);
});

test("missing Google configuration fails closed without exposing app assets", async () => {
  const env = { ...testEnvironment(), GOOGLE_CLIENT_SECRET: "" };
  assert.equal((await worker.fetch(req("/"), env)).status, 503);
});

test("permissions come from the current server allowlist, not request headers", async () => {
  const env = testEnvironment();
  const cookie = (await createSession(env.DB, user)).split(";")[0];
  let paidCalls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => { paidCalls++; throw new Error("unexpected provider call"); };
  try {
    for (const path of ["/api/usage", "/api/comparison-surface", "/debug", "/assets/DeveloperMode-test.js"]) {
      const response = await worker.fetch(req(path, cookie, { headers: { "X-Admin": "true", "oai-authenticated-user-email": testUser.email } }), env);
      assert.equal(response.status, 403);
    }
    const session = await (await worker.fetch(req("/api/session", cookie), env)).json();
    assert.equal(session.permissions.debug, false);
    assert.equal(paidCalls, 0);
    assert.equal((await currentUser(req("/", cookie), { ...env, ADMIN_EMAILS: user.email.toUpperCase() })).privileged, true);
    assert.equal((await currentUser(req("/"), { ...env, ADMIN_EMAILS: "" })).privileged, false);
  } finally { globalThis.fetch = originalFetch; }
});

test("forged, expired and revoked sessions are rejected", async () => {
  const env = testEnvironment();
  assert.equal(await currentUser(req("/", `__Host-equidistant_session=${"b".repeat(64)}`), env), null);
  env.DB.sqlite.exec("UPDATE auth_sessions SET expires_at = 1");
  assert.equal(await currentUser(req("/"), env), null);
  const cookie = (await createSession(env.DB, user)).split(";")[0];
  assert.equal((await currentUser(req("/", cookie), env)).id, user.id);
  assert.equal((await worker.fetch(req("/auth/logout", cookie, { method: "POST" }), env)).status, 303);
  assert.equal(await currentUser(req("/", cookie), env), null);
});

test("cross-site paid requests and logout cannot use a valid session", async () => {
  const env = testEnvironment();
  for (const path of ["/auth/logout", "/api/venue-recommendations", "/api/comparison-surface"]) {
    for (const origin of ["https://attacker.example", "null", ""]) {
      assert.equal((await worker.fetch(req(path, testCookie, { method: "POST", headers: { Origin: origin } }), env)).status, 403);
    }
  }
  assert.ok(await currentUser(req("/"), env));
});

test("a stale tab cannot send workspace requests under another account", async () => {
  const env = testEnvironment();
  const response = await worker.fetch(req("/api/venue-recommendations", testCookie, {
    method: "POST", headers: { "X-Equidistant-Account": user.id }
  }), env);
  assert.equal(response.status, 409);
});

test("the old sharing host redirects to the canonical host without copying OAuth parameters", async () => {
  const response = await worker.fetch(new Request("https://old.example/auth/google/callback?code=private&state=private"), testEnvironment());
  assert.equal(response.headers.get("Location"), "https://example.test/");
});

test("Google tokens require a valid signature, audience, issuer, nonce and verified email", async () => {
  const valid = await token();
  const identity = await verifyGoogleToken(valid, "test-client", "expected-nonce", publicKey);
  assert.equal(identity.id, await digest("google:google-account-123"));
  assert.equal(identity.name, "");
  await assert.rejects(verifyGoogleToken(valid, "other-client", "expected-nonce", publicKey));
  await assert.rejects(verifyGoogleToken(valid, "test-client", "wrong-nonce", publicKey));
  await assert.rejects(verifyGoogleToken(await token({ email_verified: false }), "test-client", "expected-nonce", publicKey));
  const other = await generateKeyPair("RS256");
  await assert.rejects(verifyGoogleToken(await token({}, other.privateKey), "test-client", "expected-nonce", publicKey));
  const expired = await new SignJWT({ nonce: "expected-nonce", email: user.email, email_verified: true })
    .setProtectedHeader({ alg: "RS256" }).setIssuer("https://accounts.google.com").setAudience("test-client")
    .setSubject("someone").setIssuedAt(1).setExpirationTime(2).sign(privateKey);
  await assert.rejects(verifyGoogleToken(expired, "test-client", "expected-nonce", publicKey));
  const wrongIssuer = await new SignJWT({ nonce: "expected-nonce", email: user.email, email_verified: true })
    .setProtectedHeader({ alg: "RS256" }).setIssuer("https://attacker.example").setAudience("test-client")
    .setSubject("someone").setIssuedAt().setExpirationTime("5m").sign(privateKey);
  await assert.rejects(verifyGoogleToken(wrongIssuer, "test-client", "expected-nonce", publicKey));
});

test("Google callback is browser-bound, PKCE-protected, one-use, and creates an opaque session", async (context) => {
  const env = testEnvironment();
  const start = await worker.fetch(req("/auth/google/start", ""), env);
  const authUrl = new URL(start.headers.get("Location"));
  const flowCookie = start.headers.get("Set-Cookie").split(";")[0];
  assert.equal(authUrl.origin, "https://accounts.google.com");
  assert.equal(authUrl.searchParams.get("scope"), "openid email profile");
  assert.equal(authUrl.searchParams.get("code_challenge_method"), "S256");
  assert.equal(authUrl.searchParams.get("redirect_uri"), "https://example.test/auth/google/callback");
  assert.match(start.headers.get("Set-Cookie"), /HttpOnly; Secure; SameSite=Lax/);
  let exchanges = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (String(url) === "https://www.googleapis.com/oauth2/v3/certs") return Response.json({ keys: [publicJwk] });
    assert.equal(String(url), "https://oauth2.googleapis.com/token");
    assert.equal(new URLSearchParams(init.body).get("code_verifier").length, 64);
    exchanges++;
    return Response.json({ id_token: await token({ nonce: authUrl.searchParams.get("nonce") }) });
  };
  context.after(() => { globalThis.fetch = originalFetch; });
  const callback = `/auth/google/callback?state=${authUrl.searchParams.get("state")}&code=one-use-code`;
  assert.equal((await worker.fetch(req(callback, ""), env)).headers.get("Location"), "/?signin=failed");
  assert.equal(exchanges, 0);
  const done = await worker.fetch(req(callback, flowCookie), env);
  assert.equal(done.headers.get("Location"), "/");
  const sessionCookie = done.headers.getSetCookie().find((value) => value.startsWith("__Host-equidistant_session="));
  assert.ok(sessionCookie);
  assert.match(sessionCookie, /HttpOnly; Secure; SameSite=Lax/);
  assert.equal((await currentUser(req("/", sessionCookie.split(";")[0]), env)).email, "person@example.test");
  assert.equal((await worker.fetch(req(callback, flowCookie), env)).headers.get("Location"), "/?signin=failed");
  assert.equal(exchanges, 1);
  assert.doesNotMatch(JSON.stringify(env.DB.sqlite.prepare("SELECT * FROM auth_sessions").all()), /one-use-code|test-client-secret/);
});

test("rolling quota survives a clock-hour boundary and expires at exactly 60 minutes", async () => {
  const env = testEnvironment();
  const first = Math.floor(new Date("2026-09-08T09:59:00Z").getTime() / 1000);
  for (let i = 0; i < 5; i++) await consumeVenueBudget(env, user, first + i);
  await assert.rejects(consumeVenueBudget(env, user, first + 120), (error) => error.status === 429 && error.retryAfter === 3480);
  await consumeVenueBudget(env, user, first + 3600);
  await assert.rejects(consumeVenueBudget(env, user, first + 3600), (error) => error.status === 429 && error.retryAfter === 1);
});

test("simultaneous requests cannot exceed five account reservations", async () => {
  const env = testEnvironment();
  const now = Math.floor(Date.now() / 1000);
  const results = await Promise.allSettled(Array.from({ length: 12 }, () => consumeVenueBudget(env, user, now)));
  assert.equal(results.filter((value) => value.status === "fulfilled").length, 5);
  assert.equal(env.DB.sqlite.prepare("SELECT COUNT(*) AS n FROM recommendation_events").get().n, 5);
  await consumeVenueBudget(env, { ...user, id: "another-account" }, now);
});

test("approved accounts bypass the personal limit but retain global cost caps", async () => {
  const env = testEnvironment();
  const now = Math.floor(Date.now() / 1000);
  for (let i = 0; i < 30; i++) await consumeVenueBudget(env, { ...user, privileged: true }, now);
  await assert.rejects(consumeVenueBudget(env, { ...user, privileged: true }, now), (error) => error.status === 429);
});

test("photos are limited to the account, photo and lifetime of issued recommendations", async () => {
  const env = testEnvironment();
  const result = await signPhotoUrls({ places: [{ photo_url: "/api/place-photo?name=places/a/photos/b" }] }, user, env);
  const request = req(result.places[0].photo_url);
  assert.equal(await allowedPhoto(request, user, env), true);
  assert.equal(await allowedPhoto(request, { ...user, id: "other" }, env), false);
  assert.equal(await allowedPhoto(req(result.places[0].photo_url.replace("photos%2Fb", "photos%2Fc")), user, env), false);
  assert.equal(await allowedPhoto(req("/api/place-photo?name=places/a/photos/b"), user, env), false);
});

test("sign-in starts are bounded by client without retaining raw IP addresses", async () => {
  const env = testEnvironment();
  for (let i = 0; i < 20; i++) {
    assert.equal((await worker.fetch(req("/auth/google/start", "", { headers: { "CF-Connecting-IP": "203.0.113.9" } }), env)).status, 303);
  }
  assert.equal((await worker.fetch(req("/auth/google/start", "", { headers: { "CF-Connecting-IP": "203.0.113.9" } }), env)).status, 429);
  assert.equal((await worker.fetch(req("/auth/google/start", "", { headers: { "CF-Connecting-IP": "203.0.113.10" } }), env)).status, 303);
  assert.doesNotMatch(JSON.stringify(env.DB.sqlite.prepare("SELECT * FROM auth_rate_limits").all()), /203\.0\.113/);
});


test("the privacy notice is public and the app links to it before sign-in", async () => {
  const response = await worker.fetch(req("/privacy", ""), testEnvironment());
  assert.equal(response.status, 200);
  assert.match(await response.text(), /Privacy at Equidistant/);
  const login = await worker.fetch(req("/", ""), testEnvironment());
  assert.match(await login.text(), /href="\/privacy"/);
});
