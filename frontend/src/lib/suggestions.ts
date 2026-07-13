import type { SurfaceCell } from "../types";

const EARTH_RADIUS_KM = 6371;

function radians(value: number): number {
  return (value * Math.PI) / 180;
}

export function distanceKm(left: Pick<SurfaceCell, "lat" | "lng">, right: Pick<SurfaceCell, "lat" | "lng">): number {
  const deltaLat = radians(right.lat - left.lat);
  const deltaLng = radians(right.lng - left.lng);
  const a =
    Math.sin(deltaLat / 2) ** 2 +
    Math.cos(radians(left.lat)) * Math.cos(radians(right.lat)) * Math.sin(deltaLng / 2) ** 2;
  return EARTH_RADIUS_KM * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function score(cell: SurfaceCell): number {
  const value = cell.model_score_minutes ?? cell.score_minutes ?? cell.travel_time_minutes;
  return typeof value === "number" && Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
}

export function selectSeparatedSuggestions(
  cells: SurfaceCell[],
  minimumDistanceKm: number,
  limit = 3
): SurfaceCell[] {
  const selected: SurfaceCell[] = [];
  const sorted = cells.slice().sort((left, right) => score(left) - score(right));
  for (const cell of sorted) {
    if (!Number.isFinite(score(cell))) {
      continue;
    }
    if (selected.every((candidate) => distanceKm(candidate, cell) >= minimumDistanceKm)) {
      selected.push(cell);
      if (selected.length >= limit) {
        break;
      }
    }
  }
  return selected;
}
