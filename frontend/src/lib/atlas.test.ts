import { describe, expect, it, vi } from "vitest";
import {
  cellOwnedByFocus,
  clampToAtlasBounds,
  combineValues,
  isWithinAtlasBounds,
  selectCompatibleAnchors,
  standardDeviation
} from "./atlas";

describe("offline atlas group scoring", () => {
  it("matches the backend sample standard deviation", () => {
    expect(standardDeviation([20, 30])).toBeCloseTo(Math.sqrt(50));
    expect(standardDeviation([20])).toBe(0);
  });

  it("supports every group combination strategy", () => {
    const values = [20, 30, 40];
    expect(combineValues(values, "mean")).toBe(30);
    expect(combineValues(values, "max")).toBe(40);
    expect(combineValues(values, "fairness")).toBe(10);
    expect(combineValues(values, "balanced")).toBe(35);
  });

  it("uses the participant surface unchanged for one person", () => {
    expect(combineValues([27.5], "balanced")).toBe(27.5);
    expect(combineValues([27.5], "fairness")).toBe(27.5);
  });

  it("uses explicit v2 ownership for central, inner, and wide cells", () => {
    const metadata = {
      version: 2,
      band_ownership: {
        central: { grid_bands: ["Zone 1 core"] },
        inner: { coverage_regions: ["original"] },
        wide: { coverage_regions: ["original", "outer"] }
      }
    };
    const central = { grid_band: "Zone 1 core", coverage_region: "original" as const };
    const inner = { grid_band: "Zones 2-3", coverage_region: "original" as const };
    const outer = { grid_band: "Expanded ring", coverage_region: "outer" as const };

    expect(cellOwnedByFocus(central, "central", metadata)).toBe(true);
    expect(cellOwnedByFocus(inner, "central", metadata)).toBe(false);
    expect(cellOwnedByFocus(inner, "inner", metadata)).toBe(true);
    expect(cellOwnedByFocus(outer, "inner", metadata)).toBe(false);
    expect(cellOwnedByFocus(outer, "wide", metadata)).toBe(true);
  });

  it("rejects origins outside atlas coverage bounds", () => {
    const metadata = {
      origin_bounds: { south: 51.434, north: 51.579, west: -0.335, east: 0.104 },
      coverage_bounds: { south: 51.434, north: 51.579, west: -0.335, east: 0.104 }
    };
    expect(isWithinAtlasBounds({ lat: 51.5, lng: -0.1 }, metadata)).toBe(true);
    expect(isWithinAtlasBounds({ lat: 51.6, lng: -0.1 }, metadata)).toBe(false);
  });

  it("clamps an out-of-coverage marker to the nearest boundary", () => {
    const bounds = { south: 51.434, north: 51.579, west: -0.335, east: 0.104 };

    expect(clampToAtlasBounds({ lat: 51.6, lng: -0.4 }, bounds)).toEqual({
      lat: 51.579,
      lng: -0.335,
      wasClamped: true
    });
    expect(clampToAtlasBounds({ lat: 51.5, lng: -0.1 }, bounds)).toEqual({
      lat: 51.5,
      lng: -0.1,
      wasClamped: false
    });
  });

  it("does not blend a lone discontinuous journey-time regime", () => {
    const ranked = [
      { index: 0, distance: 350, signature: 181 },
      { index: 1, distance: 950, signature: 67 },
      { index: 2, distance: 1145, signature: 58 },
      { index: 3, distance: 1415, signature: 66 }
    ];

    expect(selectCompatibleAnchors(ranked, 20).map(({ index }) => index)).toEqual([2, 3, 1]);
  });

  it("uses the nearest regime when a discontinuity splits neighbours evenly", () => {
    const ranked = [
      { index: 4, distance: 300, signature: 180 },
      { index: 5, distance: 850, signature: 67 },
      { index: 6, distance: 1300, signature: 190 },
      { index: 7, distance: 1450, signature: 66 }
    ];

    expect(selectCompatibleAnchors(ranked, 20).map(({ index }) => index)).toEqual([4, 6]);
  });

  it("preserves legacy interpolation without signatures", () => {
    const ranked = [
      { index: 0, distance: 100 },
      { index: 1, distance: 200 },
      { index: 2, distance: 300 }
    ];

    expect(selectCompatibleAnchors(ranked, 20)).toEqual(ranked);
  });

  it("updates cached map labels when a participant is renamed", async () => {
    vi.resetModules();
    const { getAtlasSurface } = await import("./atlas");
    const metadata = {
      version: 1, quantisation_step_minutes: 1,
      origin_count: 1, cell_count: 1, interpolation_neighbours: 1,
      model_file: "model.u8", graph_file: "graph.u8", model_type: "test",
      origins: [{ origin_id: "o", lat: 51.51, lng: -0.1, lat_index: 0, lng_index: 0 }],
      cells: [{ destination_id: "d", lat: 51.52, lng: -0.12, grid_band: "Zone 1 core", boundary: [] }]
    };
    const fetchMock = vi.fn(async (url: string) => url.includes("atlas.json")
      ? Response.json(metadata)
      : new Response(new Uint8Array([20])));
    vi.stubGlobal("fetch", fetchMock);
    try {
      const request = {
        friends: [{ id: "person", name: "Original name", lat: 51.51, lng: -0.1 }],
        includedFriendIndexes: [0], combine: "balanced" as const, focus: "central" as const
      };
      const first = await getAtlasSurface(request);
      const renamed = await getAtlasSurface({ ...request, friends: [{ ...request.friends[0], name: "Updated name" }] });
      expect(first.cells[0].friend_0_name).toBe("Original name");
      expect(renamed.cells[0].friend_0_name).toBe("Updated name");
      expect(renamed.cells[0].model_score_minutes).toBe(first.cells[0].model_score_minutes);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
