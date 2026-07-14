import { describe, expect, it } from "vitest";
import type { Friend, SurfaceCell } from "../types";
import { locationLabelForFriend } from "./locations";

const friend: Friend = { id: "friend-1", name: "Alex", lat: 51.5, lng: -0.1 };

function cell(overrides: Partial<SurfaceCell>): SurfaceCell {
  return {
    destination_id: "cell-1",
    lat: 51.5,
    lng: -0.1,
    x_index: 0,
    y_index: 0,
    ...overrides
  };
}

describe("locationLabelForFriend", () => {
  it("uses the label for the H3 cell containing the participant", () => {
    const cells = [
      cell({
        nearest_station_name: "Borough",
        boundary: [
          [51.49, -0.11],
          [51.49, -0.09],
          [51.51, -0.09],
          [51.51, -0.11]
        ]
      })
    ];

    expect(locationLabelForFriend(friend, cells)).toBe("Borough");
  });

  it("falls back to the searched place name while cell labels are unavailable", () => {
    expect(locationLabelForFriend({ ...friend, locationLabel: "De Beauvoir Town" }, [])).toBe("De Beauvoir Town");
  });
});
