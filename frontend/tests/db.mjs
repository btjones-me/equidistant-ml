import { DatabaseSync } from "node:sqlite";
import { readFileSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";

export const testCookie = `__Host-equidistant_session=${"a".repeat(64)}`;
export const testUser = { id: "test-user", email: "admin@example.test", name: "Test Admin" };

export function createDb() {
  const sqlite = new DatabaseSync(":memory:");
  for (const name of readdirSync(new URL("../drizzle/", import.meta.url)).filter((name) => name.endsWith(".sql")).sort()) {
    sqlite.exec(readFileSync(new URL(`../drizzle/${name}`, import.meta.url), "utf8"));
  }
  sqlite.prepare("INSERT INTO auth_sessions VALUES (?, ?, ?, ?, ?)").run(
    createHash("sha256").update("a".repeat(64)).digest("hex"), testUser.id, testUser.email, testUser.name,
    Math.floor(Date.now() / 1000) + 3600
  );
  const db = {
    sqlite,
    prepare(sql) {
      let parameters = [];
      const invoke = (method) => {
        const statement = sqlite.prepare(sql);
        const args = /\?\d/.test(sql) ? [Object.fromEntries(parameters.map((v, i) => [i + 1, v]))] : parameters;
        const result = statement[method](...args);
        return result;
      };
      const query = {
        bind(...values) { parameters = values; return query; },
        async first() { return invoke("get") ?? null; },
        async all() { return { results: invoke("all") }; },
        async run() { return { success: true, meta: { changes: invoke("run").changes } }; }
      };
      return query;
    },
    async batch(statements) {
      sqlite.exec("BEGIN");
      try { const results = []; for (const statement of statements) results.push(await statement.run()); sqlite.exec("COMMIT"); return results; }
      catch (error) { sqlite.exec("ROLLBACK"); throw error; }
    }
  };
  return db;
}

export function testEnvironment(db = createDb()) {
  return { DB: db, AUTH_ORIGIN: "https://example.test", GOOGLE_CLIENT_ID: "test-client",
    GOOGLE_CLIENT_SECRET: "test-client-secret", AUTH_SECRET: "test-auth-secret-must-be-at-least-32-characters",
    ADMIN_EMAILS: testUser.email,
    ASSETS: { fetch: async () => new Response("protected app", { headers: { "Content-Type": "text/plain" } }) }
  };
}
