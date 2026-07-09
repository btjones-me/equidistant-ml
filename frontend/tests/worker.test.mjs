import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import worker from "../worker/index.js";

const env = {
  SITE_PASSWORD: "equidistantftw",
  ASSETS: {
    fetch: async () => new Response("protected app", { headers: { "Content-Type": "text/plain" } })
  }
};

test("hosted assets always pass through the password worker", async () => {
  const config = JSON.parse(
    await readFile(new URL("../sites.wrangler.json", import.meta.url), "utf8")
  );
  assert.equal(config.assets.binding, "ASSETS");
  assert.equal(config.assets.run_worker_first, true);
});

test("password gate hides every asset until unlocked", async () => {
  const locked = await worker.fetch(new Request("https://example.test/"), env);
  assert.equal(locked.status, 200);
  assert.match(await locked.text(), /Private preview/);
  assert.equal(locked.headers.get("x-frame-options"), "DENY");
});

test("incorrect passwords are rejected", async () => {
  const body = new URLSearchParams({ password: "wrong" });
  const response = await worker.fetch(
    new Request("https://example.test/unlock", { method: "POST", body }),
    env
  );
  assert.equal(response.status, 401);
  assert.match(await response.text(), /not correct/);
});

test("a successful unlock grants access with a secure cookie", async () => {
  const body = new URLSearchParams({ password: env.SITE_PASSWORD });
  const unlocked = await worker.fetch(
    new Request("https://example.test/unlock", { method: "POST", body }),
    env
  );
  assert.equal(unlocked.status, 303);
  const setCookie = unlocked.headers.get("set-cookie");
  assert.match(setCookie, /HttpOnly/);
  assert.match(setCookie, /Secure/);
  assert.match(setCookie, /SameSite=Strict/);

  const cookie = setCookie.split(";", 1)[0];
  const appResponse = await worker.fetch(
    new Request("https://example.test/", { headers: { Cookie: cookie } }),
    env
  );
  assert.equal(appResponse.status, 200);
  assert.equal(await appResponse.text(), "protected app");
});

test("hosted HTML receives an absolute social preview URL", async () => {
  const body = new URLSearchParams({ password: env.SITE_PASSWORD });
  const unlocked = await worker.fetch(
    new Request("https://equidistant.example/unlock", { method: "POST", body }),
    env
  );
  const cookie = unlocked.headers.get("set-cookie").split(";", 1)[0];
  const htmlEnv = {
    ...env,
    ASSETS: {
      fetch: async () => new Response(
        '<meta property="og:image" content="__EQUIDISTANT_ORIGIN__/og.png">',
        { headers: { "Content-Type": "text/html" } }
      )
    }
  };

  const response = await worker.fetch(
    new Request("https://equidistant.example/", { headers: { Cookie: cookie } }),
    htmlEnv
  );
  const html = await response.text();

  assert.match(html, /https:\/\/equidistant\.example\/og\.png/);
  assert.doesNotMatch(html, /__EQUIDISTANT_ORIGIN__/);
});
