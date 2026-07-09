const ACCESS_COOKIE = "equidistant_access";
const ACCESS_MESSAGE = "equidistant-sites-access-v1";

function bytesToHex(bytes) {
  return [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function accessToken(password) {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  return bytesToHex(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(ACCESS_MESSAGE)));
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
</style></head><body><main><div class="brand"><span class="mark">◎</span>Equidistant</div><h1>${unavailable ? "Preview unavailable" : "Private preview"}</h1><p>${unavailable ? "Access has not been configured for this deployment." : "Enter the shared preview password to open the London travel-time app."}</p>${unavailable ? "" : `<form method="post" action="/unlock"><label for="password">Password</label><input id="password" name="password" type="password" autocomplete="current-password" autofocus required><button type="submit">Open Equidistant</button></form>`}${error ? `<p class="error">${error}</p>` : ""}<p class="note">This preview uses an offline model and does not call TravelTime when you move people.</p></main></body></html>`;
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
  async fetch(request, env) {
    const url = new URL(request.url);
    const password = env.SITE_PASSWORD;
    if (!password) {
      return htmlResponse(loginPage({ unavailable: true }), 503);
    }
    const expectedToken = await accessToken(password);

    if (url.pathname === "/unlock" && request.method === "POST") {
      const form = await request.formData();
      const suppliedPassword = String(form.get("password") || "");
      if (suppliedPassword !== password) {
        return htmlResponse(loginPage({ error: "That password is not correct." }), 401);
      }
      return new Response(null, {
        status: 303,
        headers: secureHeaders(new Headers({
          Location: "/",
          "Set-Cookie": `${ACCESS_COOKIE}=${expectedToken}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=604800`,
          "Cache-Control": "no-store"
        }))
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
    if (url.pathname.startsWith("/api/")) {
      return Response.json({ detail: "This hosted preview uses the browser atlas." }, { status: 404, headers: secureHeaders() });
    }

    const assetResponse = await env.ASSETS.fetch(request);
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
