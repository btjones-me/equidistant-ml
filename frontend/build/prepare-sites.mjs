import { mkdir, readFile, readdir, rename, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const LOCAL_ASSET_NAMESPACE = "_eq_local_protected_assets";
const ASSET_NAMESPACE = process.env.SITE_ASSET_NAMESPACE || LOCAL_ASSET_NAMESPACE;
const ASSET_NAMESPACE_PLACEHOLDER = "__EQUIDISTANT_ASSET_NAMESPACE__";
if (
  ASSET_NAMESPACE !== LOCAL_ASSET_NAMESPACE &&
  !/^_eq_[a-f0-9]{64}$/.test(ASSET_NAMESPACE)
) {
  throw new Error("SITE_ASSET_NAMESPACE must be _eq_ followed by 64 lowercase hex characters");
}
const root = resolve(import.meta.dirname, "..");
const output = resolve(root, "dist");
const client = resolve(output, "client", ASSET_NAMESPACE);

await mkdir(client, { recursive: true });
for (const entry of await readdir(output, { withFileTypes: true })) {
  if (entry.name === "client" || entry.name === "server" || entry.name === ".openai") {
    continue;
  }
  await rename(resolve(output, entry.name), resolve(client, entry.name));
}

await mkdir(resolve(root, "dist/server"), { recursive: true });
const workerSource = await readFile(resolve(root, "worker/index.js"), "utf8");
if (!workerSource.includes(ASSET_NAMESPACE_PLACEHOLDER)) {
  throw new Error("Worker asset namespace placeholder is missing");
}
await writeFile(
  resolve(root, "dist/server/index.js"),
  workerSource.replaceAll(ASSET_NAMESPACE_PLACEHOLDER, `/${ASSET_NAMESPACE}`)
);
