import { brotliCompressSync, gzipSync } from "node:zlib";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";

const atlasDir = path.resolve(process.argv[2]);
const startCore = performance.now();
const metadataBytes = await readFile(path.join(atlasDir, "atlas.json"));
const metadata = JSON.parse(metadataBytes.toString("utf8"));
const modelBytes = await readFile(path.join(atlasDir, metadata.model_file));
const coldCoreMs = performance.now() - startCore;

const participants = [
  { lat: 51.5518, lng: -0.1956 },
  { lat: 51.5007, lng: -0.1246 },
  { lat: 51.4626, lng: 0.0286 }
];

function radians(value) {
  return (value * Math.PI) / 180;
}

function distanceMetres(a, b) {
  const dLat = radians(b.lat - a.lat);
  const dLng = radians(b.lng - a.lng);
  const value =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(radians(a.lat)) * Math.cos(radians(b.lat)) * Math.sin(dLng / 2) ** 2;
  return 6_371_000 * 2 * Math.atan2(Math.sqrt(value), Math.sqrt(1 - value));
}

function selectCompatibleAnchors(ranked) {
  const threshold = metadata.discontinuity_threshold_minutes;
  if (threshold === undefined || ranked.length < 3 || ranked.some((item) => item.signature === undefined)) {
    return ranked;
  }
  const bySignature = [...ranked].sort((left, right) => left.signature - right.signature);
  let largestGap = 0;
  let splitAfter = -1;
  for (let index = 0; index < bySignature.length - 1; index += 1) {
    const gap = bySignature[index + 1].signature - bySignature[index].signature;
    if (gap > largestGap) {
      largestGap = gap;
      splitAfter = index;
    }
  }
  if (largestGap <= threshold || splitAfter < 0) {
    return ranked;
  }
  const groups = [bySignature.slice(0, splitAfter + 1), bySignature.slice(splitAfter + 1)];
  if (groups[0].length !== groups[1].length) {
    return groups[0].length > groups[1].length ? groups[0] : groups[1];
  }
  return groups.find((group) => group.some((item) => item.index === ranked[0].index)) ?? ranked;
}

function render() {
  const surfaces = participants.map((participant) => {
    const nearest = selectCompatibleAnchors(metadata.origins
      .map((origin, index) => ({
        index,
        distance: distanceMetres(participant, origin),
        signature: origin.surface_signature_minutes
      }))
      .sort((left, right) => left.distance - right.distance)
      .slice(0, metadata.interpolation_neighbours));
    const inverse = nearest.map((item) => 1 / Math.max(item.distance, 40) ** 2);
    const total = inverse.reduce((sum, value) => sum + value, 0);
    const output = new Float32Array(metadata.cell_count);
    nearest.forEach((anchor, anchorIndex) => {
      const weight = inverse[anchorIndex] / total;
      const offset = anchor.index * metadata.cell_count;
      for (let cell = 0; cell < metadata.cell_count; cell += 1) {
        output[cell] += modelBytes[offset + cell] * metadata.quantisation_step_minutes * weight;
      }
    });
    return output;
  });
  const scores = new Float32Array(metadata.cell_count);
  for (let cell = 0; cell < metadata.cell_count; cell += 1) {
    const values = surfaces.map((surface) => surface[cell]);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
    scores[cell] = mean + Math.sqrt(variance) * 0.5;
  }
  return scores;
}

const firstRenderStart = performance.now();
render();
const firstRenderMs = performance.now() - firstRenderStart;
const warmRenderStart = performance.now();
render();
const warmRenderMs = performance.now() - warmRenderStart;
const graphStart = performance.now();
const graphBytes = await readFile(path.join(atlasDir, metadata.graph_file));
const lazyGraphLoadMs = performance.now() - graphStart;

const assetSizes = Object.fromEntries(
  [
    ["metadata", metadataBytes],
    ["model", modelBytes],
    ["graph", graphBytes]
  ].map(([name, bytes]) => [
    name,
    {
      raw_bytes: bytes.byteLength,
      gzip_bytes: gzipSync(bytes).byteLength,
      brotli_bytes: brotliCompressSync(bytes).byteLength
    }
  ])
);

console.log(
  JSON.stringify({
    origin_count: metadata.origin_count,
    cell_count: metadata.cell_count,
    cold_core_load_ms: coldCoreMs,
    first_surface_render_ms: firstRenderMs,
    warm_surface_render_ms: warmRenderMs,
    lazy_graph_load_ms: lazyGraphLoadMs,
    asset_sizes: assetSizes
  })
);
