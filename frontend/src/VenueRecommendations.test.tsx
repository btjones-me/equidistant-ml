import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import VenueRecommendations from "./VenueRecommendations";
import type { VenueRecommendation, VenueRecommendationsResponse } from "./types";

const places: VenueRecommendation[] = [1, 2, 3].map((index) => ({
  place_id: `place-${index}`,
  name: `Place ${index}`,
  address: `${index} Test Street, London`,
  lat: 51.51 + index / 1000,
  lng: -0.12 - index / 1000,
  primary_type: "pub",
  rating: 4.4,
  user_rating_count: 240,
  price_level: "PRICE_LEVEL_MODERATE",
  open_now: true,
  opening_summary: null,
  website_url: `https://place-${index}.example`,
  google_maps_url: `https://maps.google.com/?q=place-${index}`,
  photo_url: null,
  photo_attribution: null,
  why: `Place ${index} matches the request and is close to the meeting point.`,
  verified_details: ["Open this evening", "Suitable for groups"],
  source_urls: [`https://place-${index}.example/details`]
}));

const response: VenueRecommendationsResponse = {
  area: { name: "Soho", lat: 51.513, lng: -0.132 },
  query: "A relaxed pub",
  places,
  generated_at: "2026-07-15T12:00:00.000Z",
  cached: false
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("venue recommendations", () => {
  it("stays out of the way until a meeting area opens it", () => {
    render(
      <VenueRecommendations
        open={false}
        area={{ id: "soho-hidden", name: "Soho", lat: 51.513, lng: -0.132 }}
        activeVenueId={null}
        onClose={() => undefined}
        onResults={() => undefined}
        onSelectVenue={() => undefined}
      />
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("returns three researched places and selects the first map marker", async () => {
    const onResults = vi.fn();
    const onSelectVenue = vi.fn();
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(response), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    })));

    render(
      <VenueRecommendations
        open
        area={{ id: "soho-live", name: "Soho", lat: 51.513, lng: -0.132 }}
        activeVenueId="place-1"
        onClose={() => undefined}
        onResults={onResults}
        onSelectVenue={onSelectVenue}
      />
    );

    fireEvent.change(screen.getByLabelText("What would suit the group?"), { target: { value: "A relaxed pub" } });
    fireEvent.click(screen.getByRole("button", { name: /Find 3 places/i }));

    await waitFor(() => expect(screen.getByText("Place 1")).toBeInTheDocument());
    expect(screen.getAllByText(/^Place [123]$/)).toHaveLength(3);
    expect(onResults).toHaveBeenCalledWith(places);
    expect(onSelectVenue).toHaveBeenCalledWith("place-1");
    expect(screen.getAllByRole("link", { name: /Google Maps/i })).toHaveLength(3);
  });
});
