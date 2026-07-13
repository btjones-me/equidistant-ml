import { describe, expect, it } from "vitest";
import type { SurfaceCell } from "../types";
import { distanceKm, selectSeparatedSuggestions } from "./suggestions";

function cell(id: string, lat: number, lng: number, score: number): SurfaceCell {
  return {
    destination_id: id,
    lat,
    lng,
    x_index: 0,
    y_index: 0,
    model_score_minutes: score
  };
}

describe("meeting-area suggestions", () => {
  it("keeps the best three options geographically distinct", () => {
    const cells = [
      cell("best", 51.5074, -0.1278, 20),
      cell("too-close", 51.508, -0.126, 21),
      cell("west", 51.51, -0.18, 22),
      cell("east", 51.51, -0.075, 23),
      cell("farther", 51.56, -0.13, 24)
    ];

    expect(selectSeparatedSuggestions(cells, 3).map((candidate) => candidate.destination_id)).toEqual([
      "best",
      "west",
      "east"
    ]);
  });

  it("uses a geographic rather than coordinate-space distance", () => {
    expect(distanceKm(cell("a", 51.5, -0.1, 1), cell("b", 51.5, -0.055, 2))).toBeCloseTo(3.11, 1);
  });
});
