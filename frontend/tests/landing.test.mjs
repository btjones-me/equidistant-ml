import assert from "node:assert/strict";
import test from "node:test";
import worker from "../worker/index.js";
import { testEnvironment, testCookie } from "./db.mjs";

const request = (path, init) => new Request(`https://example.test${path}`, init);

test("anonymous landing and decoration need no database or upstream calls", async () => {
  const env = testEnvironment();
  env.DB = { prepare() { throw new Error("Public landing must not query D1"); } };
  env.ASSETS = { fetch() { throw new Error("Public landing must not load private assets"); } };
  const saved = globalThis.fetch;
  globalThis.fetch = () => { throw new Error("Public landing must not call providers"); };
  try {
    const response = await worker.fetch(request("/"), env);
    assert.equal(response.status, 200);
    const html = await response.text();
    assert.match(html, /Find a fair meeting place in London/);
    assert.match(html, /href="\/auth\/google\/start"/);
    assert.match(html, /rel="canonical"/);
    assert.doesNotMatch(html, /\/assets\/|\/model\//);
    const script = await worker.fetch(request("/welcome-motion.js"), env);
    assert.equal(script.status, 200);
    assert.match(script.headers.get("Content-Type"), /javascript/);
    assert.match(script.headers.get("Cache-Control"), /public/);
    assert.doesNotMatch(await script.text(), /fetch\(|XMLHttpRequest|sendBeacon|WebSocket|localStorage|API_KEY|model\.u8/);
    assert.match(response.headers.get("Content-Security-Policy"), /script-src 'self' https:\/\/www.googletagmanager.com;/);
  } finally { globalThis.fetch = saved; }
});

test("public resources are an exact allowlist, not a bypass into app assets", async () => {
  const env = testEnvironment();
  for (const path of ["/api/session", "/api/venue-recommendations", "/api/place-photo", "/api/geocode"]) {
    assert.equal((await worker.fetch(request(path), env)).status, 401);
  }
  for (const path of ["/assets/app.js", "/model/atlas.json", "/model/model.u8", "/welcome-motion.js/extra", "/missing-page"]) {
    const response = await worker.fetch(request(path), env);
    assert.equal(response.status, 404);
    assert.match(await response.text(), /Sign in with Google/);
    assert.equal(response.headers.get("X-Robots-Tag"), "noindex");
  }
  for (const path of ["/welcome-motion.js", "/robots.txt", "/sitemap.xml"]) {
    assert.equal((await worker.fetch(request(path, { method: "POST" }), env)).status, 405);
    const response = await worker.fetch(request(path, { method: "HEAD" }), env);
    assert.equal(response.status, 200);
    assert.equal(await response.text(), "");
  }
});

test("robots and sitemap expose public pages while signed-in root remains private", async () => {
  const env = testEnvironment();
  const robots = await (await worker.fetch(request("/robots.txt"), env)).text();
  assert.match(robots, /Disallow: \/api\//);
  const sitemap = await (await worker.fetch(request("/sitemap.xml"), env)).text();
  assert.match(sitemap, /<loc>https:\/\/equidistant.me\/<\/loc>/);
  assert.doesNotMatch(sitemap, /\/api\/|\/auth\//);
  const signed = await worker.fetch(request("/", { headers: { Cookie: testCookie } }), env);
  assert.equal(signed.status, 200);
  assert.match(signed.headers.get("Cache-Control"), /private, no-store/);
  assert.doesNotMatch(await signed.text(), /Give London a little ripple/);
});
