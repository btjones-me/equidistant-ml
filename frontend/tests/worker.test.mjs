import assert from "node:assert/strict";
import test from "node:test";
import worker from "../worker/index.js";

const env = {
  SITE_PASSWORD: "test-password-do-not-use",
  ASSETS: {
    fetch: async () => new Response("protected app", { headers: { "Content-Type": "text/plain" } })
  }
};

function createRateLimitDb() {
  const rows = new Map();
  const db = {
    prepare(sql) {
      let parameters = [];
      const statement = {
        bind(...values) {
          parameters = values;
          return statement;
        },
        async first() {
          if (sql.includes("SELECT window_started_at")) {
            return rows.get(parameters[0]) ?? null;
          }
          throw new Error(`Unexpected query: ${sql}`);
        },
        async run() {
          if (sql.includes("CREATE TABLE")) {
            return { success: true };
          }
          if (sql.includes("INSERT INTO unlock_rate_limits")) {
            const [clientKey, now, windowSeconds] = parameters;
            const current = rows.get(clientKey);
            if (!current || now - current.window_started_at >= windowSeconds) {
              rows.set(clientKey, {
                window_started_at: now,
                failures: 1,
                updated_at: now
              });
            } else {
              current.failures += 1;
              current.updated_at = now;
            }
            return { success: true };
          }
          if (sql.includes("updated_at <")) {
            for (const [key, row] of rows) {
              if (row.updated_at < parameters[0]) {
                rows.delete(key);
              }
            }
            return { success: true };
          }
          if (sql.includes("client_key =")) {
            rows.delete(parameters[0]);
            return { success: true };
          }
          throw new Error(`Unexpected query: ${sql}`);
        }
      };
      return statement;
    }
  };
  return { db, rows };
}

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

test("password failures are limited per client IP", async () => {
  const { db } = createRateLimitDb();
  const limitedEnv = { ...env, DB: db };
  const requestFor = (ip) => new Request("https://example.test/unlock", {
    method: "POST",
    headers: { "CF-Connecting-IP": ip },
    body: new URLSearchParams({ password: "wrong" })
  });

  for (let attempt = 0; attempt < 5; attempt += 1) {
    const response = await worker.fetch(requestFor("203.0.113.10"), limitedEnv);
    assert.equal(response.status, 401);
  }

  const blocked = await worker.fetch(requestFor("203.0.113.10"), limitedEnv);
  assert.equal(blocked.status, 429);
  assert.ok(Number(blocked.headers.get("retry-after")) > 0);
  assert.match(await blocked.text(), /Too many incorrect attempts/);

  const otherClient = await worker.fetch(requestFor("203.0.113.11"), limitedEnv);
  assert.equal(otherClient.status, 401);
});

test("successful unlock clears prior failures", async () => {
  const { db, rows } = createRateLimitDb();
  const limitedEnv = { ...env, DB: db };
  const headers = { "CF-Connecting-IP": "203.0.113.12" };

  await worker.fetch(
    new Request("https://example.test/unlock", {
      method: "POST",
      headers,
      body: new URLSearchParams({ password: "wrong" })
    }),
    limitedEnv
  );
  assert.equal(rows.size, 1);

  const unlocked = await worker.fetch(
    new Request("https://example.test/unlock", {
      method: "POST",
      headers,
      body: new URLSearchParams({ password: env.SITE_PASSWORD })
    }),
    limitedEnv
  );
  assert.equal(unlocked.status, 303);
  assert.equal(rows.size, 0);
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

test("authenticated public paths map to the private asset namespace", async () => {
  let requestedPath = "";
  const mappedEnv = {
    ...env,
    ASSETS: {
      fetch: async (request) => {
        requestedPath = new URL(request.url).pathname;
        return new Response("model bytes");
      }
    }
  };
  const body = new URLSearchParams({ password: env.SITE_PASSWORD });
  const unlocked = await worker.fetch(
    new Request("https://example.test/unlock", { method: "POST", body }),
    mappedEnv
  );
  const cookie = unlocked.headers.get("set-cookie").split(";", 1)[0];

  await worker.fetch(
    new Request("https://example.test/model/model.u8", { headers: { Cookie: cookie } }),
    mappedEnv
  );

  assert.equal(
    requestedPath,
    "/__EQUIDISTANT_ASSET_NAMESPACE__/model/model.u8"
  );
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
