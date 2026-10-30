import assert from "node:assert/strict";
import test from "node:test";
import worker, { privateFiftyPubsPreset } from "../worker/index.js";
import { allowedReturnTo, createSession } from "../worker/auth.js";
import { testCookie, testEnvironment, testUser } from "./db.mjs";

const fixture = [
  { id: "couple-one", name: "Example & Partner", locationLabel: "Example station", lat: 51.55, lng: -0.19 },
  { id: "couple-two", name: "Another & Partner", locationLabel: "Another station", lat: 51.51, lng: -0.1 }
];
const request = (path, cookie = testCookie, init = {}) => new Request(`https://example.test${path}`, {
  ...init, headers: { Cookie: cookie, Origin: "https://example.test", ...init.headers }
});
const presetEnv = () => ({ ...testEnvironment(), FIFTY_PUBS_OWNER_EMAIL: testUser.email.toUpperCase(), FIFTY_PUBS_PRESET_JSON: JSON.stringify(fixture) });

test("50 pubs is an unlisted Google sign-in entry, not public planner access", async () => {
  const env = presetEnv();
  env.ASSETS = { fetch() { throw new Error("Anonymous requests must not reach private assets"); } };
  const response = await worker.fetch(request("/50pubs", ""), env);
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Sign in with Google/);
  assert.match(html, /href="\/auth\/google\/start\?returnTo=%2F50pubs"/);
  assert.doesNotMatch(html, /Example & Partner|Another & Partner|couple-one|51\.55/);
  assert.equal(response.headers.get("X-Robots-Tag"), "noindex");
  assert.match(response.headers.get("Cache-Control"), /private, no-store/);
  assert.equal((await worker.fetch(request("/api/50pubs/preset", ""), env)).status, 401);
  const root = await (await worker.fetch(request("/", ""), env)).text();
  assert.doesNotMatch(root, /50pubs/);
  assert.match(root, /href="\/auth\/google\/start"/);
  const robots = await (await worker.fetch(request("/robots.txt", ""), env)).text();
  assert.match(robots, /Disallow: \/50pubs/);
  assert.doesNotMatch(await (await worker.fetch(request("/sitemap.xml", ""), env)).text(), /50pubs/);
});

test("signed-in 50 pubs receives the protected app fallback and no indexing", async () => {
  const env = presetEnv();
  const paths = [];
  env.ASSETS = { fetch: async (req) => {
    const path = new URL(req.url).pathname;
    paths.push(path);
    return path.endsWith("index.html") ? new Response("private app shell", { headers: { "Content-Type": "text/html" } }) : new Response("", { status: 404 });
  } };
  const response = await worker.fetch(request("/50pubs", testCookie, { headers: { Accept: "text/html" } }), env);
  assert.equal(response.status, 200);
  assert.equal(await response.text(), "private app shell");
  assert.deepEqual(paths, ["/__EQUIDISTANT_ASSET_NAMESPACE__/50pubs", "/__EQUIDISTANT_ASSET_NAMESPACE__/index.html"]);
  assert.equal(response.headers.get("X-Robots-Tag"), "noindex");
  assert.match(response.headers.get("Cache-Control"), /private, no-store/);
});

test("only the configured owner receives the runtime preset and it is not cacheable", async () => {
  const env = presetEnv();
  const owner = await worker.fetch(request("/api/50pubs/preset"), env);
  assert.equal(owner.status, 200);
  assert.deepEqual(await owner.json(), { friends: fixture });
  assert.match(owner.headers.get("Cache-Control"), /private, no-store/);
  assert.equal(owner.headers.get("Vary"), "Cookie");
  assert.equal(owner.headers.get("X-Robots-Tag"), "noindex");
  const ordinary = { id: "different-account", email: "other@example.test", name: "Other" };
  const cookie = (await createSession(env.DB, ordinary)).split(";")[0];
  env.ADMIN_EMAILS = ordinary.email;
  const other = await worker.fetch(request("/api/50pubs/preset", cookie, {
    headers: { "X-Admin": "true", "oai-authenticated-user-email": testUser.email }
  }), env);
  assert.equal(other.status, 200);
  assert.deepEqual(await other.json(), { friends: [] });
  const changed = await worker.fetch(request("/api/50pubs/preset", cookie, { headers: { "X-Equidistant-Account": testUser.id } }), env);
  assert.equal(changed.status, 409);
  assert.equal((await worker.fetch(request("/api/50pubs/preset", testCookie, { method: "POST" }), env)).status, 404);
});

test("missing or malformed private preset configuration fails closed", () => {
  const env = presetEnv();
  assert.deepEqual(privateFiftyPubsPreset(testUser, { ...env, FIFTY_PUBS_OWNER_EMAIL: "" }), []);
  assert.deepEqual(privateFiftyPubsPreset(testUser, { ...env, FIFTY_PUBS_OWNER_EMAIL: `other@example.test,${testUser.email}` }), []);
  assert.deepEqual(privateFiftyPubsPreset(testUser, { ...env, FIFTY_PUBS_PRESET_JSON: undefined }), []);
  const invalid = [null, {}, [], Array(7).fill(fixture[0]), [fixture[0], fixture[0]], [{ ...fixture[0], lat: "51.55" }],
    [{ ...fixture[0], lng: 200 }], [{ ...fixture[0], name: "" }], [{ ...fixture[0], locationLabel: "" }], [{ ...fixture[0], id: "<script>" }]];
  for (const value of invalid) assert.deepEqual(privateFiftyPubsPreset(testUser, { ...env, FIFTY_PUBS_PRESET_JSON: JSON.stringify(value) }), []);
  assert.deepEqual(privateFiftyPubsPreset(testUser, { ...env, FIFTY_PUBS_PRESET_JSON: "not json" }), []);
  assert.deepEqual(privateFiftyPubsPreset(testUser, { ...env, FIFTY_PUBS_PRESET_JSON: JSON.stringify([{ ...fixture[0], secret: "discard me" }]) }), [fixture[0]]);
});

test("return-to supports only the exact local 50 pubs route", async () => {
  assert.equal(allowedReturnTo("/50pubs"), "/50pubs");
  const invalid = [null, "", "https://attacker.example/", "//attacker.example/50pubs", "/50pubs?next=evil", "/50pubs#evil", "/50pubs/", "javascript:alert(1)", "%2F50pubs", "/50pubs\r\nLocation: https://attacker.example/"];
  for (const value of invalid) assert.equal(allowedReturnTo(value), "/");
  for (const destination of ["/50pubs", ...invalid.filter((value) => typeof value === "string")]) {
    const env = presetEnv();
    const response = await worker.fetch(request(`/auth/google/start?returnTo=${encodeURIComponent(destination)}`, ""), env);
    const url = new URL(response.headers.get("Location"));
    assert.equal(url.origin, "https://accounts.google.com");
    const state = url.searchParams.get("state");
    assert.match(state, destination === "/50pubs" ? /^[a-f0-9]{64}:50pubs$/ : /^[a-f0-9]{64}$/);
    assert.equal(env.DB.sqlite.prepare("SELECT state FROM auth_flows").get().state, state);
  }
});

test("canonical redirects preserve only the special local route, never callback secrets", async () => {
  const response = await worker.fetch(new Request("https://old.example/50pubs?private=value"), presetEnv());
  assert.equal(response.headers.get("Location"), "https://example.test/50pubs");
});


test("explicit authenticated audience shares only the validated initial preset with every signed-in account", async () => {
  const env = { ...presetEnv(), FIFTY_PUBS_PRESET_AUDIENCE: "authenticated" };
  const otherUser = { id: "different-account", email: "other@example.test", name: "Other" };
  const cookie = (await createSession(env.DB, otherUser)).split(";")[0];
  const response = await worker.fetch(request("/api/50pubs/preset", cookie, {
    headers: { "X-Equidistant-Account": otherUser.id }
  }), env);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { friends: fixture });
  assert.match(response.headers.get("Cache-Control"), /private, no-store/);
  assert.equal(response.headers.get("Vary"), "Cookie");
  assert.equal(response.headers.get("X-Robots-Tag"), "noindex");
  assert.deepEqual(privateFiftyPubsPreset(otherUser, { ...env, FIFTY_PUBS_OWNER_EMAIL: "" }), fixture);
  // Shared initialization is not an API for saving or sharing later local edits.
  const write = await worker.fetch(request("/api/50pubs/preset", cookie, {
    method: "POST", body: JSON.stringify({ friends: [{ ...fixture[0], name: "Not shared" }] })
  }), env);
  assert.equal(write.status, 404);
  assert.deepEqual(await (await worker.fetch(request("/api/50pubs/preset", testCookie), env)).json(), { friends: fixture });
});

test("shared preset audience never bypasses authentication or account-change checks", async () => {
  const env = { ...presetEnv(), FIFTY_PUBS_PRESET_AUDIENCE: "authenticated" };
  for (const cookie of ["", "equidistant_access=legacy", `__Host-equidistant_session=${"b".repeat(64)}`]) {
    const response = await worker.fetch(request("/api/50pubs/preset", cookie, {
      headers: { "X-Admin": "true", "oai-authenticated-user-email": testUser.email, "X-Equidistant-Account": testUser.id }
    }), env);
    assert.equal(response.status, 401);
    assert.doesNotMatch(await response.text(), /Example & Partner|Another & Partner/);
  }
  const stale = await worker.fetch(request("/api/50pubs/preset", testCookie, {
    headers: { "X-Equidistant-Account": "different-account" }
  }), env);
  assert.equal(stale.status, 409);
  assert.doesNotMatch(await stale.text(), /Example & Partner|Another & Partner/);
  for (const invalid of [null, undefined, {}, { email: testUser.email }, { id: "", email: testUser.email }, { id: "one", email: "" }]) {
    assert.deepEqual(privateFiftyPubsPreset(invalid, env), []);
  }
  assert.deepEqual(privateFiftyPubsPreset(testUser, { ...env, FIFTY_PUBS_PRESET_JSON: "not json" }), []);
  assert.deepEqual(privateFiftyPubsPreset(testUser, { ...env, FIFTY_PUBS_PRESET_JSON: JSON.stringify([{ ...fixture[0], lat: 80 }]) }), []);
});

test("unknown or absent preset audience stays owner-only and cannot be selected through request headers", async () => {
  const otherUser = { id: "different-account", email: "other@example.test", name: "Other" };
  for (const audience of [undefined, "", "public", "all", "AUTHENTICATED", " authenticated", true]) {
    const env = { ...presetEnv(), FIFTY_PUBS_PRESET_AUDIENCE: audience };
    assert.deepEqual(privateFiftyPubsPreset(otherUser, env), []);
    assert.deepEqual(privateFiftyPubsPreset(testUser, env), fixture);
  }
  const env = presetEnv();
  const cookie = (await createSession(env.DB, otherUser)).split(";")[0];
  const response = await worker.fetch(request("/api/50pubs/preset?audience=authenticated", cookie, {
    headers: { "FIFTY_PUBS_PRESET_AUDIENCE": "authenticated", "X-Preset-Audience": "authenticated" }
  }), env);
  assert.deepEqual(await response.json(), { friends: [] });
});

test("privacy notice explains shared initial-group exposure and keeps later edits account-local", async () => {
  const response = await worker.fetch(request("/privacy", ""), presetEnv());
  const html = await response.text();
  assert.match(html, /any signed-in Equidistant user who opens that collection/);
  assert.match(html, /not only people sent its link/);
  assert.match(html, /later changes remain in your account's browser workspace and are not shared/);
  assert.doesNotMatch(html, /Example & Partner|Another & Partner/);
});
