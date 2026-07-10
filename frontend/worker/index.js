const ACCESS_COOKIE = "equidistant_access";
const VISITOR_COOKIE = "__Host-equidistant_visitor";
const ACCESS_MESSAGE = "equidistant-sites-access-v1";
const ASSET_NAMESPACE = "__EQUIDISTANT_ASSET_NAMESPACE__";
const RATE_LIMIT_MAX_FAILURES = 5;
const RATE_LIMIT_WINDOW_SECONDS = 10 * 60;
const RATE_LIMIT_RETENTION_SECONDS = 24 * 60 * 60;
const VISITOR_COOKIE_MAX_AGE = 365 * 24 * 60 * 60;
let databaseSchemaPromise;

function bytesToHex(bytes) {
  return [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function hmacToken(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  return bytesToHex(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message)));
}

async function accessToken(password) {
  return hmacToken(password, ACCESS_MESSAGE);
}

function secureHeaders(headers = new Headers()) {
  headers.set("Content-Security-Policy", "default-src 'self'; img-src 'self' data: https://*.tile.openstreetmap.org; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'");
  headers.set("Referrer-Policy", "no-referrer");
  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("X-Frame-Options", "DENY");
  headers.set("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
  return headers;
}

function htmlResponse(body, status = 200, extraHeaders = {}) {
  const headers = secureHeaders(new Headers({ "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store", ...extraHeaders }));
  return new Response(body, { status, headers });
}

function loginPage({ error = "", unavailable = false } = {}) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#087f73"><title>Equidistant · Private preview</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172019;background:#edf1ed}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:linear-gradient(160deg,#f8faf8 0 54%,#e4ece6 54% 100%)}main{width:min(100%,390px);border:1px solid #d5ddd7;border-radius:8px;padding:28px;background:#fff;box-shadow:0 24px 70px rgb(23 32 25 / 14%)}.brand{display:flex;align-items:center;gap:10px;margin-bottom:30px;font-weight:850}.mark{display:grid;place-items:center;width:34px;height:34px;border-radius:50%;color:#fff;background:#087f73;font-size:18px}p{color:#68736c;font-size:13px;line-height:1.5}h1{margin:0;font-family:Georgia,"Times New Roman",serif;font-size:30px;font-weight:600;letter-spacing:0}form{display:grid;gap:10px;margin-top:22px}label{font-size:11px;font-weight:850;text-transform:uppercase}input{width:100%;border:1px solid #bcc9c0;border-radius:7px;padding:12px;color:#172019;background:#fff;font:inherit}input:focus{outline:3px solid rgb(8 127 115 / 18%);border-color:#087f73}button{border:0;border-radius:7px;padding:12px;color:#fff;background:#087f73;font:inherit;font-weight:850;cursor:pointer}.error{margin:10px 0 0;color:#a43e2f}.note{margin:18px 0 0;font-size:11px}
</style></head><body><main><div class="brand"><span class="mark">◎</span>Equidistant</div><h1>${unavailable ? "Preview unavailable" : "Private preview"}</h1><p>${unavailable ? "Access has not been configured for this deployment." : "Enter the shared preview password to open the London travel-time app."}</p>${unavailable ? "" : `<form method="post" action="/unlock"><label for="password">Password</label><input id="password" name="password" type="password" autocomplete="current-password" autofocus required><button type="submit">Open Equidistant</button></form>`}${error ? `<p class="error">${error}</p>` : ""}<p class="note">This preview uses an offline model. Successful unlocks record an anonymous browser count and coarse city-level location; no raw IP is stored.</p></main></body></html>`;
}

function readCookie(request, name) {
  const cookie = request.headers.get("Cookie") || "";
  for (const part of cookie.split(";")) {
    const [key, ...value] = part.trim().split("=");
    if (key === name) {
      return value.join("=");
    }
  }
  return null;
}

function protectedAssetRequest(request, pathname) {
  const assetUrl = new URL(request.url);
  assetUrl.pathname = `${ASSET_NAMESPACE}${pathname}`;
  const headers = new Headers(request.headers);
  headers.delete("Cookie");
  return new Request(assetUrl, { method: request.method, headers });
}

async function ensureDatabaseSchema(db) {
  if (!databaseSchemaPromise) {
    databaseSchemaPromise = db.batch([
      db.prepare(`
        CREATE TABLE IF NOT EXISTS unlock_rate_limits (
          client_key TEXT PRIMARY KEY NOT NULL,
          window_started_at INTEGER NOT NULL,
          failures INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        )
      `),
      db.prepare(`
        CREATE TABLE IF NOT EXISTS analytics_visitors (
          visitor_key TEXT PRIMARY KEY NOT NULL,
          first_seen_at INTEGER NOT NULL,
          last_seen_at INTEGER NOT NULL,
          unlocks INTEGER NOT NULL,
          country TEXT,
          region TEXT,
          city TEXT
        )
      `)
    ]).catch((error) => {
      databaseSchemaPromise = undefined;
      throw error;
    });
  }
  await databaseSchemaPromise;
}

async function unlockRateLimit(env, request) {
  const clientIp = request.headers.get("CF-Connecting-IP")?.trim();
  if (!env.DB || !clientIp) {
    return { limited: false, clientKey: null, now: 0, retryAfter: 0 };
  }

  try {
    await ensureDatabaseSchema(env.DB);
    const now = Math.floor(Date.now() / 1000);
    const clientKey = await hmacToken(
      env.RATE_LIMIT_SECRET || env.SITE_PASSWORD,
      `unlock:${clientIp}`
    );
    const row = await env.DB.prepare(
      "SELECT window_started_at, failures FROM unlock_rate_limits WHERE client_key = ?1"
    ).bind(clientKey).first();
    if (row && now - Number(row.window_started_at) < RATE_LIMIT_WINDOW_SECONDS) {
      const retryAfter = Math.max(
        1,
        Number(row.window_started_at) + RATE_LIMIT_WINDOW_SECONDS - now
      );
      return {
        limited: Number(row.failures) >= RATE_LIMIT_MAX_FAILURES,
        clientKey,
        now,
        retryAfter
      };
    }
    return { limited: false, clientKey, now, retryAfter: 0 };
  } catch (error) {
    console.warn("Unlock rate limiter unavailable", error instanceof Error ? error.message : "unknown error");
    return { limited: false, clientKey: null, now: 0, retryAfter: 0 };
  }
}

function visitorIdentity(request) {
  const existing = readCookie(request, VISITOR_COOKIE);
  if (existing && /^[a-f0-9]{32}$/.test(existing)) {
    return existing;
  }
  return bytesToHex(crypto.getRandomValues(new Uint8Array(16)));
}

function geoText(value, maxLength) {
  return typeof value === "string" && value.trim()
    ? value.trim().slice(0, maxLength)
    : null;
}

async function recordVisitor(env, request, visitorId) {
  if (!env.DB) {
    return;
  }
  try {
    await ensureDatabaseSchema(env.DB);
    const visitorKey = await hmacToken(
      env.ANALYTICS_SECRET || env.RATE_LIMIT_SECRET || env.SITE_PASSWORD,
      `visitor:${visitorId}`
    );
    const now = Math.floor(Date.now() / 1000);
    const cf = request.cf || {};
    await env.DB.prepare(`
      INSERT INTO analytics_visitors (
        visitor_key, first_seen_at, last_seen_at, unlocks, country, region, city
      )
      VALUES (?1, ?2, ?2, 1, ?3, ?4, ?5)
      ON CONFLICT(visitor_key) DO UPDATE SET
        last_seen_at = ?2,
        unlocks = unlocks + 1,
        country = COALESCE(?3, country),
        region = COALESCE(?4, region),
        city = COALESCE(?5, city)
    `).bind(
      visitorKey,
      now,
      geoText(cf.country, 2),
      geoText(cf.region, 80),
      geoText(cf.city, 80)
    ).run();
  } catch (error) {
    console.warn("Unable to record visitor", error instanceof Error ? error.message : "unknown error");
  }
}

async function usageSummary(env) {
  if (!env.DB) {
    return Response.json({ detail: "Usage database unavailable" }, { status: 503, headers: secureHeaders() });
  }
  try {
    await ensureDatabaseSchema(env.DB);
    const now = Math.floor(Date.now() / 1000);
    const totals = await env.DB.prepare(`
      SELECT
        COUNT(*) AS all_time,
        COALESCE(SUM(CASE WHEN last_seen_at >= ?1 THEN 1 ELSE 0 END), 0) AS last_24h,
        COALESCE(SUM(CASE WHEN last_seen_at >= ?2 THEN 1 ELSE 0 END), 0) AS last_7d,
        COALESCE(SUM(CASE WHEN last_seen_at >= ?3 THEN 1 ELSE 0 END), 0) AS last_30d,
        COALESCE(SUM(unlocks), 0) AS total_unlocks,
        MAX(last_seen_at) AS last_seen_at
      FROM analytics_visitors
    `).bind(now - 86400, now - 7 * 86400, now - 30 * 86400).first();
    const locationResult = await env.DB.prepare(`
      SELECT
        COALESCE(country, 'Unknown') AS country,
        COALESCE(region, 'Unknown') AS region,
        COALESCE(city, 'Unknown') AS city,
        COUNT(*) AS unique_visitors,
        SUM(unlocks) AS unlocks
      FROM analytics_visitors
      GROUP BY country, region, city
      ORDER BY unique_visitors DESC, unlocks DESC
      LIMIT 20
    `).all();
    return Response.json({
      unique_visitors: {
        all_time: Number(totals?.all_time || 0),
        last_24h: Number(totals?.last_24h || 0),
        last_7d: Number(totals?.last_7d || 0),
        last_30d: Number(totals?.last_30d || 0)
      },
      total_unlocks: Number(totals?.total_unlocks || 0),
      last_seen_at: totals?.last_seen_at ? Number(totals.last_seen_at) : null,
      locations: (locationResult.results || []).map((row) => ({
        country: String(row.country),
        region: String(row.region),
        city: String(row.city),
        unique_visitors: Number(row.unique_visitors || 0),
        unlocks: Number(row.unlocks || 0)
      }))
    }, { headers: secureHeaders(new Headers({ "Cache-Control": "no-store" })) });
  } catch (error) {
    console.warn("Unable to load usage summary", error instanceof Error ? error.message : "unknown error");
    return Response.json({ detail: "Usage summary unavailable" }, { status: 503, headers: secureHeaders() });
  }
}

async function recordUnlockFailure(env, rateLimit) {
  if (!env.DB || !rateLimit.clientKey) {
    return;
  }
  try {
    await env.DB.prepare(`
      INSERT INTO unlock_rate_limits (client_key, window_started_at, failures, updated_at)
      VALUES (?1, ?2, 1, ?2)
      ON CONFLICT(client_key) DO UPDATE SET
        failures = CASE
          WHEN ?2 - window_started_at >= ?3 THEN 1
          ELSE failures + 1
        END,
        window_started_at = CASE
          WHEN ?2 - window_started_at >= ?3 THEN ?2
          ELSE window_started_at
        END,
        updated_at = ?2
    `).bind(rateLimit.clientKey, rateLimit.now, RATE_LIMIT_WINDOW_SECONDS).run();
    await env.DB.prepare(
      "DELETE FROM unlock_rate_limits WHERE updated_at < ?1"
    ).bind(rateLimit.now - RATE_LIMIT_RETENTION_SECONDS).run();
  } catch (error) {
    console.warn("Unable to record unlock failure", error instanceof Error ? error.message : "unknown error");
  }
}

async function clearUnlockFailures(env, clientKey) {
  if (!env.DB || !clientKey) {
    return;
  }
  try {
    await env.DB.prepare(
      "DELETE FROM unlock_rate_limits WHERE client_key = ?1"
    ).bind(clientKey).run();
  } catch (error) {
    console.warn("Unable to clear unlock failures", error instanceof Error ? error.message : "unknown error");
  }
}

async function geocode(request) {
  const query = new URL(request.url).searchParams.get("q")?.trim() || "";
  if (query.length < 2 || query.length > 120) {
    return Response.json({ detail: "Enter at least two characters." }, { status: 422, headers: secureHeaders() });
  }
  const url = new URL("https://nominatim.openstreetmap.org/search");
  url.search = new URLSearchParams({
    q: `${query}, London, UK`,
    format: "jsonv2",
    limit: "5",
    countrycodes: "gb",
    viewbox: "-0.60,51.75,0.35,51.20",
    bounded: "1",
    addressdetails: "1"
  }).toString();
  try {
    const upstream = await fetch(url, {
      headers: { "User-Agent": "equidistant-ml/0.2 (https://github.com/btjones-me/equidistant-ml)" }
    });
    if (!upstream.ok) {
      throw new Error(`Nominatim returned ${upstream.status}`);
    }
    const results = (await upstream.json()).map((item) => {
      const address = item.address || {};
      return {
        name: item.name || address.suburb || address.neighbourhood || address.road || String(item.display_name || "London").split(",", 1)[0],
        lat: Number(item.lat),
        lng: Number(item.lon),
        detail: String(item.display_name || "")
      };
    });
    return Response.json({ results }, { headers: secureHeaders(new Headers({ "Cache-Control": "private, max-age=3600" })) });
  } catch {
    return Response.json({ detail: "Location search unavailable" }, { status: 503, headers: secureHeaders() });
  }
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const password = env.SITE_PASSWORD;
    if (!password) {
      return htmlResponse(loginPage({ unavailable: true }), 503);
    }
    const expectedToken = await accessToken(password);

    if (url.pathname === "/unlock" && request.method === "POST") {
      const rateLimit = await unlockRateLimit(env, request);
      if (rateLimit.limited) {
        return htmlResponse(
          loginPage({ error: "Too many incorrect attempts. Try again in a few minutes." }),
          429,
          { "Retry-After": String(rateLimit.retryAfter) }
        );
      }
      const form = await request.formData();
      const suppliedPassword = String(form.get("password") || "");
      if (suppliedPassword !== password) {
        await recordUnlockFailure(env, rateLimit);
        return htmlResponse(loginPage({ error: "That password is not correct." }), 401);
      }
      await clearUnlockFailures(env, rateLimit.clientKey);
      const visitorId = visitorIdentity(request);
      const visitorWrite = recordVisitor(env, request, visitorId);
      if (ctx?.waitUntil) {
        ctx.waitUntil(visitorWrite);
      } else {
        await visitorWrite;
      }
      const headers = secureHeaders(new Headers({
        Location: "/",
        "Cache-Control": "no-store"
      }));
      headers.append(
        "Set-Cookie",
        `${ACCESS_COOKIE}=${expectedToken}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=604800`
      );
      headers.append(
        "Set-Cookie",
        `${VISITOR_COOKIE}=${visitorId}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=${VISITOR_COOKIE_MAX_AGE}`
      );
      return new Response(null, {
        status: 303,
        headers
      });
    }

    if (url.pathname === "/logout") {
      return new Response(null, {
        status: 303,
        headers: secureHeaders(new Headers({
          Location: "/",
          "Set-Cookie": `${ACCESS_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0`,
          "Cache-Control": "no-store"
        }))
      });
    }

    if (readCookie(request, ACCESS_COOKIE) !== expectedToken) {
      return htmlResponse(loginPage());
    }

    if (url.pathname === "/api/geocode") {
      return geocode(request);
    }
    if (url.pathname === "/api/usage" && request.method === "GET") {
      return usageSummary(env);
    }
    if (url.pathname.startsWith("/api/")) {
      return Response.json({ detail: "This hosted preview uses the browser atlas." }, { status: 404, headers: secureHeaders() });
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      return Response.json({ detail: "Method not allowed" }, { status: 405, headers: secureHeaders() });
    }

    const wantsHtml = request.headers.get("Accept")?.includes("text/html") ?? false;
    const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
    let assetResponse = await env.ASSETS.fetch(protectedAssetRequest(request, pathname));
    if (assetResponse.status === 404 && wantsHtml) {
      assetResponse = await env.ASSETS.fetch(protectedAssetRequest(request, "/index.html"));
    }
    const headers = secureHeaders(new Headers(assetResponse.headers));
    if (assetResponse.headers.get("Content-Type")?.includes("text/html")) {
      headers.set("Cache-Control", "no-store");
      headers.delete("Content-Length");
      const html = (await assetResponse.text()).replaceAll("__EQUIDISTANT_ORIGIN__", url.origin);
      return new Response(html, { status: assetResponse.status, statusText: assetResponse.statusText, headers });
    }
    return new Response(assetResponse.body, { status: assetResponse.status, statusText: assetResponse.statusText, headers });
  }
};
