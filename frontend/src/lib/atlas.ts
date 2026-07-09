import type { CombineMode, Friend, SurfaceCell, SurfaceResponse } from "../types";

type AtlasOrigin = {
  origin_id: string;
  lat: number;
  lng: number;
  lat_index: number;
  lng_index: number;
};

type AtlasCell = Pick<
  SurfaceCell,
  | "destination_id"
  | "lat"
  | "lng"
  | "boundary"
  | "h3_cell"
  | "h3_resolution"
  | "grid_band"
  | "grid_priority"
  | "cell_area_km2"
> & {
  nearest_station_name?: string;
  nearest_station_lines?: string;
};

type AtlasMetadata = {
  version: number;
  quantisation_step_minutes: number;
  origin_count: number;
  cell_count: number;
  interpolation_neighbours: number;
  model_type: string;
  source_model_sha256?: string;
  model_file: string;
  graph_file: string;
  interpolation_validation?: {
    mae_minutes_vs_direct_model?: number;
    p90_abs_error_minutes_vs_direct_model?: number;
  };
  origins: AtlasOrigin[];
  cells: AtlasCell[];
};

export type AtlasSurfaceRequest = {
  friends: Friend[];
  includedFriendIndexes: number[];
  combine: CombineMode;
  focus: "central" | "inner" | "wide";
  includeGraph?: boolean;
};

type CoreAtlas = {
  metadata: AtlasMetadata;
  model: Uint8Array;
};

let coreAtlasPromise: Promise<CoreAtlas> | null = null;
let graphAtlasPromise: Promise<Uint8Array> | null = null;
const responseCache = new Map<string, SurfaceResponse>();
const MAX_CACHE_ITEMS = 20;

function assetUrl(path: string): string {
  const base = import.meta.env.BASE_URL.endsWith("/") ? import.meta.env.BASE_URL : `${import.meta.env.BASE_URL}/`;
  return new URL(`${base}model/${path}`, window.location.origin).toString();
}

async function loadCoreAtlas(): Promise<CoreAtlas> {
  const metadataResponse = await fetch(assetUrl("atlas.json"), { cache: "no-cache" });
  if (!metadataResponse.ok) {
    throw new Error("The offline London model is not available in this build.");
  }
  if (!metadataResponse.headers.get("content-type")?.includes("application/json")) {
    throw new Error(`The offline London model metadata was not served correctly (${metadataResponse.url}).`);
  }
  const metadata = (await metadataResponse.json()) as AtlasMetadata;
  const modelResponse = await fetch(assetUrl(metadata.model_file), { cache: "force-cache" });
  if (!modelResponse.ok) {
    throw new Error("The offline London model could not be loaded.");
  }
  const modelBuffer = await modelResponse.arrayBuffer();
  const expectedLength = metadata.origin_count * metadata.cell_count;
  if (modelBuffer.byteLength !== expectedLength) {
    throw new Error("The offline London model files do not match their metadata.");
  }
  return {
    metadata,
    model: new Uint8Array(modelBuffer)
  };
}

export function preloadAtlas(): Promise<void> {
  coreAtlasPromise ??= loadCoreAtlas();
  return coreAtlasPromise.then(() => undefined);
}

async function loadGraphAtlas(metadata: AtlasMetadata): Promise<Uint8Array> {
  const graphResponse = await fetch(assetUrl(metadata.graph_file), { cache: "force-cache" });
  if (!graphResponse.ok) {
    throw new Error("The offline graph baseline could not be loaded.");
  }
  const graphBuffer = await graphResponse.arrayBuffer();
  if (graphBuffer.byteLength !== metadata.origin_count * metadata.cell_count) {
    throw new Error("The offline graph baseline does not match its metadata.");
  }
  return new Uint8Array(graphBuffer);
}

function radians(value: number): number {
  return (value * Math.PI) / 180;
}

function distanceMetres(latA: number, lngA: number, latB: number, lngB: number): number {
  const dLat = radians(latB - latA);
  const dLng = radians(lngB - lngA);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(radians(latA)) * Math.cos(radians(latB)) * Math.sin(dLng / 2) ** 2;
  return 6_371_000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function nearestAnchorWeights(origin: Friend, metadata: AtlasMetadata): Array<{ index: number; weight: number }> {
  const ranked = metadata.origins
    .map((anchor, index) => ({
      index,
      distance: distanceMetres(origin.lat, origin.lng, anchor.lat, anchor.lng)
    }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, Math.max(1, metadata.interpolation_neighbours));
  if (ranked[0].distance < 20) {
    return [{ index: ranked[0].index, weight: 1 }];
  }
  const raw = ranked.map(({ index, distance }) => ({ index, weight: 1 / Math.max(distance, 40) ** 2 }));
  const total = raw.reduce((sum, item) => sum + item.weight, 0);
  return raw.map((item) => ({ ...item, weight: item.weight / total }));
}

function interpolateSurface(
  origin: Friend,
  values: Uint8Array,
  metadata: AtlasMetadata
): Float32Array {
  const output = new Float32Array(metadata.cell_count);
  const anchors = nearestAnchorWeights(origin, metadata);
  for (const anchor of anchors) {
    const offset = anchor.index * metadata.cell_count;
    for (let cellIndex = 0; cellIndex < metadata.cell_count; cellIndex += 1) {
      output[cellIndex] += values[offset + cellIndex] * metadata.quantisation_step_minutes * anchor.weight;
    }
  }
  return output;
}

function mean(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

export function standardDeviation(values: number[]): number {
  if (values.length <= 1) {
    return 0;
  }
  const average = mean(values);
  const sumSquares = values.reduce((total, value) => total + (value - average) ** 2, 0);
  return Math.sqrt(sumSquares / (values.length - 1));
}

export function combineValues(values: number[], mode: CombineMode): number {
  if (values.length === 1) {
    return values[0];
  }
  if (mode === "max") {
    return Math.max(...values);
  }
  if (mode === "mean") {
    return mean(values);
  }
  const fairness = standardDeviation(values);
  return mode === "fairness" ? fairness : mean(values) + fairness * 0.5;
}

function percentile(values: number[], quantile: number): number | null {
  if (!values.length) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * quantile))];
}

function cacheKey(request: AtlasSurfaceRequest): string {
  return JSON.stringify({
    friends: request.friends.map((friend) => [Number(friend.lat.toFixed(6)), Number(friend.lng.toFixed(6))]),
    included: request.includedFriendIndexes,
    combine: request.combine,
    focus: request.focus,
    graph: Boolean(request.includeGraph)
  });
}

function putCache(key: string, response: SurfaceResponse) {
  if (responseCache.has(key)) {
    responseCache.delete(key);
  }
  responseCache.set(key, response);
  while (responseCache.size > MAX_CACHE_ITEMS) {
    const oldest = responseCache.keys().next().value;
    if (typeof oldest !== "string") {
      break;
    }
    responseCache.delete(oldest);
  }
}

export async function getAtlasSurface(request: AtlasSurfaceRequest): Promise<SurfaceResponse> {
  const key = cacheKey(request);
  const cached = responseCache.get(key);
  if (cached) {
    responseCache.delete(key);
    responseCache.set(key, cached);
    return cached;
  }
  coreAtlasPromise ??= loadCoreAtlas();
  const atlas = await coreAtlasPromise;
  if (request.includeGraph) {
    graphAtlasPromise ??= loadGraphAtlas(atlas.metadata);
  }
  const graphAtlas = request.includeGraph ? await graphAtlasPromise : null;
  const modelSurfaces = request.friends.map((friend) => interpolateSurface(friend, atlas.model, atlas.metadata));
  const graphSurfaces = graphAtlas
    ? request.friends.map((friend) => interpolateSurface(friend, graphAtlas, atlas.metadata))
    : null;
  const included = request.includedFriendIndexes.length ? request.includedFriendIndexes : [0];
  const sourceCellIndexes = atlas.metadata.cells
    .map((cell, index) => ({ cell, index }))
    .filter(({ cell }) => request.focus !== "central" || cell.grid_band === "Zone 1 core");
  const cells = sourceCellIndexes.map(({ cell, index }) => {
    const modelFriendValues = modelSurfaces.map((surface) => surface[index]);
    const graphFriendValues = graphSurfaces?.map((surface) => surface[index]) ?? [];
    const includedModelValues = included.map((friendIndex) => modelFriendValues[friendIndex]);
    const includedGraphValues = graphSurfaces ? included.map((friendIndex) => graphFriendValues[friendIndex]) : [];
    const modelScore = combineValues(includedModelValues, request.combine);
    const graphScore = graphSurfaces ? combineValues(includedGraphValues, request.combine) : null;
    const next: SurfaceCell = {
      ...cell,
      x_index: index,
      y_index: 0,
      score_minutes: modelScore,
      model_score_minutes: modelScore,
      max_minutes: Math.max(...includedModelValues),
      mean_minutes: mean(includedModelValues),
      fairness_minutes: standardDeviation(includedModelValues),
      included_friend_indexes: included.join(",")
    };
    if (graphScore !== null) {
      next.graph_score_minutes = graphScore;
      next.model_residual_minutes = modelScore - graphScore;
    }
    modelFriendValues.forEach((value, friendIndex) => {
      next[`friend_${friendIndex}_minutes`] = value;
      next[`friend_${friendIndex}_model_minutes`] = value;
      if (graphSurfaces) {
        next[`friend_${friendIndex}_graph_minutes`] = graphFriendValues[friendIndex];
        next[`friend_${friendIndex}_model_residual_minutes`] = value - graphFriendValues[friendIndex];
      }
      next[`friend_${friendIndex}_name`] = request.friends[friendIndex].name;
    });
    return next;
  });
  const scores = cells.map((cell) => cell.model_score_minutes ?? 0);
  const response: SurfaceResponse = {
    lats: [],
    lngs: [],
    Z: [],
    cells,
    metadata: {
      value_column: "model_score_minutes",
      cell_count: cells.length,
      grid_type: "h3",
      friend_columns: request.friends.map((_, index) => `friend_${index}_model_minutes`),
      value_columns: {
        model: "model_score_minutes",
        graph: "graph_score_minutes",
        residual: "model_residual_minutes",
        reference: "reference_score_minutes",
        error: "signed_error_minutes"
      },
      min: scores.length ? Math.min(...scores) : null,
      max: scores.length ? Math.max(...scores) : null,
      p10: percentile(scores, 0.1),
      p50: percentile(scores, 0.5),
      p90: percentile(scores, 0.9),
      source: "browser_atlas",
      model_type: atlas.metadata.model_type,
      interpolation_mae_minutes: atlas.metadata.interpolation_validation?.mae_minutes_vs_direct_model,
      coverage_notice: request.focus === "wide" ? "The hosted model is optimised for Zones 1-3." : undefined
    }
  };
  putCache(key, response);
  return response;
}
