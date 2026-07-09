import { copyFile, mkdir, readdir, rename } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const output = resolve(root, "dist");
const client = resolve(output, "client");

await mkdir(client, { recursive: true });
for (const entry of await readdir(output, { withFileTypes: true })) {
  if (entry.name === "client" || entry.name === "server" || entry.name === ".openai") {
    continue;
  }
  await rename(resolve(output, entry.name), resolve(client, entry.name));
}

await mkdir(resolve(root, "dist/server"), { recursive: true });
await copyFile(resolve(root, "worker/index.js"), resolve(root, "dist/server/index.js"));
await copyFile(resolve(root, "sites.wrangler.json"), resolve(root, "dist/wrangler.json"));
