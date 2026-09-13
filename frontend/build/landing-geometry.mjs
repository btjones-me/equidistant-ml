// Regenerate decorative geometry after intentionally updating the coverage atlas.
// This is not part of runtime: no model values or participant data are published.
import { readFile, writeFile } from "node:fs/promises";

const atlas = JSON.parse(await readFile(new URL("../public/model/atlas.json", import.meta.url), "utf8"));
const point = (lat, lng) => [Number(((lng + 0.335) * 2000).toFixed(1)), Number(((51.59 - lat) * 3200).toFixed(1))];
const cells = atlas.cells.map(({ boundary, lat, lng }) => ({ p: boundary.map(([lat, lng]) => point(lat, lng)), c: point(lat, lng) }));
await writeFile(new URL("../worker/landing-geometry.js", import.meta.url),
  "// Display geometry only, derived from the existing London atlas. No travel-time values.\nexport const cells = " + JSON.stringify(cells) + ";\n");
