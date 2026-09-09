import { createRemoteJWKSet, jwtVerify, SignJWT } from "jose";

export const SESSION_COOKIE = "__Host-equidistant_session";
const FLOW_COOKIE = "__Host-equidistant_oauth";
const SESSION_SECONDS = 7 * 24 * 60 * 60;
const googleKeys = createRemoteJWKSet(new URL("https://www.googleapis.com/oauth2/v3/certs"));
const encoder = new TextEncoder();
// Schema changes are applied by the hosting migration step, never during a request.
export async function requireAuthDatabase(db) {
  if (!db) throw new Error("Authentication database unavailable");
}

export function cookieValue(request, name) {
  return (request.headers.get("Cookie") || "").split(";")
    .map((part) => part.trim()).find((part) => part.startsWith(`${name}=`))?.slice(name.length + 1) || "";
}

function randomToken() {
  return Array.from(crypto.getRandomValues(new Uint8Array(32)), (v) => v.toString(16).padStart(2, "0")).join("");
}

export async function digest(value) {
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(value))), (v) => v.toString(16).padStart(2, "0")).join("");
}

function cookie(name, value, maxAge) {
  return `${name}=${value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}`;
}

function redirect(location, cookies = []) {
  const headers = new Headers({ Location: location, "Cache-Control": "no-store" });
  cookies.forEach((value) => headers.append("Set-Cookie", value));
  return new Response(null, { status: 303, headers });
}

export function configured(env) {
  try {
    const origin = new URL(env.AUTH_ORIGIN);
    return Boolean(env.DB && env.GOOGLE_CLIENT_ID && env.GOOGLE_CLIENT_SECRET &&
      env.AUTH_SECRET?.length >= 32 && origin.protocol === "https:" && origin.origin === env.AUTH_ORIGIN);
  } catch { return false; }
}

export function isSameOrigin(request) {
  return request.headers.get("Origin") === new URL(request.url).origin &&
    request.headers.get("Sec-Fetch-Site") !== "cross-site";
}

export async function verifyGoogleToken(token, clientId, nonce, keys = googleKeys) {
  const { payload } = await jwtVerify(token, keys, {
    algorithms: ["RS256"], audience: clientId,
    issuer: ["https://accounts.google.com", "accounts.google.com"],
    requiredClaims: ["sub", "iat", "exp", "nonce", "email", "email_verified"],
    maxTokenAge: "10 minutes", clockTolerance: 5
  });
  if (payload.nonce !== nonce || payload.email_verified !== true ||
      typeof payload.sub !== "string" || !payload.sub || payload.sub.length > 255 ||
      typeof payload.email !== "string" || payload.email.length > 320 ||
      !payload.email.includes("@") || (payload.azp && payload.azp !== clientId)) {
    throw new Error("Invalid Google identity");
  }
  // Only Google's verified ID-token claims become account data.
  return {
    id: await digest(`google:${payload.sub}`), email: payload.email.toLowerCase(),
    name: typeof payload.name === "string" ? payload.name.slice(0, 120) : ""
  };
}

export async function createSession(db, user, now = Math.floor(Date.now() / 1000)) {
  await requireAuthDatabase(db);
  const token = randomToken();
  await db.prepare("INSERT INTO auth_sessions (token_hash, user_id, email, name, expires_at) VALUES (?1, ?2, ?3, ?4, ?5)")
    .bind(await digest(token), user.id, user.email, user.name, now + SESSION_SECONDS).run();
  return cookie(SESSION_COOKIE, token, SESSION_SECONDS);
}

export async function currentUser(request, env) {
  const token = cookieValue(request, SESSION_COOKIE);
  if (!/^[a-f0-9]{64}$/.test(token)) return null;
  await requireAuthDatabase(env.DB);
  const row = await env.DB.prepare("SELECT user_id, email, name FROM auth_sessions WHERE token_hash = ?1 AND expires_at > ?2")
    .bind(await digest(token), Math.floor(Date.now() / 1000)).first();
  if (!row) return null;
  const allowlist = new Set((env.ADMIN_EMAILS || "").split(/[\s,;]+/).filter(Boolean).map((email) => email.toLowerCase()));
  return { id: row.user_id, email: row.email, name: row.name, privileged: allowlist.has(row.email.toLowerCase()) };
}

export async function authRoute(request, env) {
  const url = new URL(request.url);
  const now = Math.floor(Date.now() / 1000);
  if (url.pathname === "/auth/google/start" && request.method === "GET") {
    await requireAuthDatabase(env.DB);
    const clientKey = await digest(`${env.AUTH_SECRET}:login:${request.headers.get("CF-Connecting-IP") || "unknown"}`);
    const attempt = await env.DB.prepare(`
      INSERT INTO auth_rate_limits (client_key, window_started_at, attempts) VALUES (?1, ?2, 1)
      ON CONFLICT(client_key) DO UPDATE SET
        attempts = CASE WHEN window_started_at <= ?2 - 600 THEN 1 ELSE attempts + 1 END,
        window_started_at = CASE WHEN window_started_at <= ?2 - 600 THEN ?2 ELSE window_started_at END
      RETURNING attempts, window_started_at
    `).bind(clientKey, now).first();
    if (Number(attempt?.attempts) > 20) return new Response("Too many sign-in attempts. Please try again in a few minutes.", {
      status: 429, headers: { "Retry-After": String(Math.max(1, Number(attempt.window_started_at) + 600 - now)) }
    });
    const flow = randomToken(), state = randomToken(), nonce = randomToken(), verifier = randomToken();
    const challengeBytes = new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(verifier)));
    const challenge = btoa(String.fromCharCode(...challengeBytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    await env.DB.batch([
      env.DB.prepare("DELETE FROM auth_rate_limits WHERE window_started_at < ?1").bind(now - 86400),
      env.DB.prepare("DELETE FROM auth_flows WHERE expires_at <= ?1").bind(now),
      env.DB.prepare("DELETE FROM auth_sessions WHERE expires_at <= ?1").bind(now),
      env.DB.prepare("INSERT INTO auth_flows (flow_hash, state, nonce, verifier, expires_at) VALUES (?1, ?2, ?3, ?4, ?5)")
        .bind(await digest(flow), state, nonce, verifier, now + 600)
    ]);
    const authUrl = new URL("https://accounts.google.com/o/oauth2/v2/auth");
    authUrl.search = new URLSearchParams({ client_id: env.GOOGLE_CLIENT_ID,
      redirect_uri: `${env.AUTH_ORIGIN}/auth/google/callback`, response_type: "code",
      scope: "openid email profile", state, nonce, code_challenge: challenge,
      code_challenge_method: "S256", prompt: "select_account" }).toString();
    return redirect(authUrl.href, [cookie(FLOW_COOKIE, flow, 600)]);
  }
  if (url.pathname === "/auth/google/callback" && request.method === "GET") {
    try {
      const flow = cookieValue(request, FLOW_COOKIE);
      const state = url.searchParams.get("state"), code = url.searchParams.get("code");
      if (!/^[a-f0-9]{64}$/.test(flow) || !state || !code || code.length > 4096) throw new Error("Missing login flow");
      await requireAuthDatabase(env.DB);
      // Consume in one statement: parallel callbacks cannot reuse this login.
      const pending = await env.DB.prepare("DELETE FROM auth_flows WHERE flow_hash = ?1 AND state = ?2 AND expires_at > ?3 RETURNING nonce, verifier")
        .bind(await digest(flow), state, now).first();
      if (!pending) throw new Error("Invalid login flow");
      const response = await fetch("https://oauth2.googleapis.com/token", {
        method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ client_id: env.GOOGLE_CLIENT_ID, client_secret: env.GOOGLE_CLIENT_SECRET,
          code, code_verifier: pending.verifier, grant_type: "authorization_code",
          redirect_uri: `${env.AUTH_ORIGIN}/auth/google/callback` }), signal: AbortSignal.timeout(15000)
      });
      if (!response.ok) throw new Error("Google login unavailable");
      const result = await response.json();
      const user = await verifyGoogleToken(result.id_token, env.GOOGLE_CLIENT_ID, pending.nonce);
      const sessionCookie = await createSession(env.DB, user, now);
      const previous = cookieValue(request, SESSION_COOKIE);
      if (previous) await env.DB.prepare("DELETE FROM auth_sessions WHERE token_hash = ?1").bind(await digest(previous)).run();
      return redirect("/", [sessionCookie, cookie(FLOW_COOKIE, "", 0), cookie("equidistant_access", "", 0)]);
    } catch {
      // Never log OAuth codes, provider tokens, or callback URLs.
      return redirect("/?signin=failed", [cookie(FLOW_COOKIE, "", 0)]);
    }
  }
  if (url.pathname === "/auth/logout" && request.method === "POST") {
    if (!isSameOrigin(request)) return Response.json({ detail: "Request origin is not allowed." }, { status: 403 });
    await requireAuthDatabase(env.DB);
    await env.DB.prepare("DELETE FROM auth_sessions WHERE token_hash = ?1")
      .bind(await digest(cookieValue(request, SESSION_COOKIE))).run();
    return redirect("/", [cookie(SESSION_COOKIE, "", 0), cookie(FLOW_COOKIE, "", 0), cookie("equidistant_access", "", 0)]);
  }
  return null;
}

export async function signPhotoUrls(payload, user, env) {
  const places = await Promise.all(payload.places.map(async (place) => {
    if (!place.photo_url) return place;
    const url = new URL(place.photo_url, env.AUTH_ORIGIN);
    const photo = url.searchParams.get("name");
    const token = await new SignJWT({ photo }).setProtectedHeader({ alg: "HS256" })
      .setSubject(user.id).setAudience("equidistant-photo").setIssuedAt().setExpirationTime("24h")
      .sign(encoder.encode(env.AUTH_SECRET));
    url.searchParams.set("token", token);
    return { ...place, photo_url: `${url.pathname}${url.search}` };
  }));
  return { ...payload, places };
}

export async function allowedPhoto(request, user, env) {
  if (user.privileged) return true;
  try {
    const url = new URL(request.url);
    const { payload } = await jwtVerify(url.searchParams.get("token") || "", encoder.encode(env.AUTH_SECRET), {
      algorithms: ["HS256"], subject: user.id, audience: "equidistant-photo", requiredClaims: ["exp"]
    });
    return payload.photo === url.searchParams.get("name");
  } catch { return false; }
}
