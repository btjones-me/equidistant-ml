import {
  Clock3,
  ExternalLink,
  LoaderCircle,
  MapPinned,
  Search,
  Sparkles,
  Star,
  X
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { VenueRecommendation, VenueRecommendationsResponse } from "./types";

type VenueArea = {
  id: string;
  name: string;
  lat: number;
  lng: number;
};

type VenueRecommendationsProps = {
  open: boolean;
  area: VenueArea;
  activeVenueId: string | null;
  onClose: () => void;
  onResults: (places: VenueRecommendation[]) => void;
  onSelectVenue: (placeId: string) => void;
};

const quickPrompts = [
  { label: "Pub", query: "A relaxed pub with good drinks and enough space to talk" },
  { label: "Dinner", query: "A well-reviewed restaurant that works for a group dinner" },
  { label: "Drinks", query: "A lively bar for drinks that is still good for conversation" },
  { label: "Museum", query: "An interesting museum or gallery that is open today" },
  { label: "Something fun", query: "A memorable activity or destination for a group of friends" }
];

const sessionCache = new Map<string, VenueRecommendationsResponse>();

function cacheKey(area: VenueArea, query: string): string {
  return `${area.id}:${query.trim().toLowerCase().replace(/\s+/g, " ")}`;
}

function displayPrice(value: string | null): string | null {
  if (!value || value === "PRICE_LEVEL_UNSPECIFIED") {
    return null;
  }
  const prices: Record<string, string> = {
    PRICE_LEVEL_FREE: "Free",
    PRICE_LEVEL_INEXPENSIVE: "£",
    PRICE_LEVEL_MODERATE: "££",
    PRICE_LEVEL_EXPENSIVE: "£££",
    PRICE_LEVEL_VERY_EXPENSIVE: "££££"
  };
  return prices[value] ?? null;
}

function sourceLabel(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "Source";
  }
}

function openStatus(place: VenueRecommendation): string {
  if (place.open_now === true) {
    return "Open now";
  }
  if (place.open_now === false) {
    return "Closed now";
  }
  return place.opening_summary || "Check opening hours";
}

export default function VenueRecommendations({
  open,
  area,
  activeVenueId,
  onClose,
  onResults,
  onSelectVenue
}: VenueRecommendationsProps) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<VenueRecommendationsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const responseAreaId = useMemo(
    () => response ? `${response.area.lat.toFixed(5)},${response.area.lng.toFixed(5)}` : null,
    [response]
  );
  const currentAreaId = `${area.lat.toFixed(5)},${area.lng.toFixed(5)}`;

  useEffect(() => {
    if (responseAreaId && responseAreaId !== currentAreaId) {
      setResponse(null);
      setError(null);
      onResults([]);
    }
  }, [currentAreaId, onResults, responseAreaId]);

  useEffect(() => {
    if (open) {
      window.setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  async function findPlaces(event: FormEvent) {
    event.preventDefault();
    const requestQuery = query.trim();
    if (requestQuery.length < 2 || loading) {
      return;
    }

    const key = cacheKey(area, requestQuery);
    const cached = sessionCache.get(key);
    if (cached) {
      setResponse({ ...cached, cached: true });
      setError(null);
      onResults(cached.places);
      onSelectVenue(cached.places[0]?.place_id ?? "");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const upstream = await fetch("/api/venue-recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: requestQuery,
          area_name: area.name,
          lat: area.lat,
          lng: area.lng
        })
      });
      const body = (await upstream.json()) as VenueRecommendationsResponse & { detail?: string };
      if (!upstream.ok) {
        throw new Error(body.detail || "Recommendations are temporarily unavailable.");
      }
      sessionCache.set(key, body);
      setResponse(body);
      onResults(body.places);
      onSelectVenue(body.places[0]?.place_id ?? "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Recommendations are temporarily unavailable.");
    } finally {
      setLoading(false);
    }
  }

  if (!open) {
    return null;
  }

  return (
    <aside className="venue-drawer" role="dialog" aria-label={`Places near ${area.name}`} aria-modal="false">
      <header className="venue-drawer-header">
        <div>
          <span><MapPinned size={15} aria-hidden="true" /> Near your meeting area</span>
          <h2>{area.name}</h2>
        </div>
        <button type="button" onClick={onClose} aria-label="Close place recommendations">
          <X size={20} aria-hidden="true" />
        </button>
      </header>

      <form className="venue-search" onSubmit={findPlaces}>
        <label htmlFor="venue-query">What would suit the group?</label>
        <div className="venue-query-box">
          <Search size={17} aria-hidden="true" />
          <textarea
            id="venue-query"
            ref={inputRef}
            rows={2}
            maxLength={300}
            value={query}
            placeholder="For example, a cosy pub with food and room for six"
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="submit" disabled={loading || query.trim().length < 2}>
            {loading ? <LoaderCircle className="venue-spin" size={17} aria-hidden="true" /> : <Sparkles size={17} aria-hidden="true" />}
            {response ? "Search again" : "Find 3 places"}
          </button>
        </div>
        <div className="venue-quick-prompts" aria-label="Quick searches">
          {quickPrompts.map((prompt) => (
            <button
              type="button"
              key={prompt.label}
              aria-pressed={query === prompt.query}
              onClick={() => setQuery(prompt.query)}
            >
              {prompt.label}
            </button>
          ))}
        </div>
      </form>

      {loading ? (
        <div className="venue-loading" aria-live="polite">
          <LoaderCircle className="venue-spin" size={23} aria-hidden="true" />
          <div><strong>Checking the best matches</strong><span>Searching live place details and opening information</span></div>
        </div>
      ) : null}
      {error ? <div className="venue-error" role="alert">{error}</div> : null}

      {response && !loading ? (
        <div className="venue-results" aria-live="polite">
          <div className="venue-results-heading">
            <strong>Best matches</strong>
            <span>{response.cached ? "Saved result" : "Researched now"}</span>
          </div>
          {response.places.map((place, index) => {
            const price = displayPrice(place.price_level);
            const active = activeVenueId === place.place_id;
            return (
              <article className={`venue-card${active ? " active" : ""}`} key={place.place_id}>
                <button
                  className="venue-card-select"
                  type="button"
                  onClick={() => onSelectVenue(place.place_id)}
                  onMouseEnter={() => onSelectVenue(place.place_id)}
                  aria-label={`Show ${place.name} on the map`}
                >
                  <span className="venue-photo-wrap">
                    {place.photo_url ? <img src={place.photo_url} alt="" loading="lazy" /> : <span className="venue-photo-fallback"><MapPinned size={24} /></span>}
                    <b>{index + 1}</b>
                  </span>
                  <span className="venue-card-copy">
                    <span className="venue-card-title"><strong>{place.name}</strong>{price ? <small>{price}</small> : null}</span>
                    <span className="venue-card-meta">
                      {place.rating ? <span><Star size={12} fill="currentColor" /> {place.rating.toFixed(1)}{place.user_rating_count ? ` (${place.user_rating_count.toLocaleString()})` : ""}</span> : null}
                      <span className={place.open_now === true ? "open" : place.open_now === false ? "closed" : ""}><Clock3 size={12} /> {openStatus(place)}</span>
                    </span>
                    <span className="venue-address">{place.address}</span>
                  </span>
                </button>
                <p>{place.why}</p>
                {place.verified_details.length ? (
                  <ul>
                    {place.verified_details.slice(0, 3).map((detail) => <li key={detail}>{detail}</li>)}
                  </ul>
                ) : null}
                <div className="venue-card-links">
                  <a href={place.google_maps_url} target="_blank" rel="noreferrer">Google Maps <ExternalLink size={12} /></a>
                  {place.website_url ? <a href={place.website_url} target="_blank" rel="noreferrer">Website <ExternalLink size={12} /></a> : null}
                  {place.source_urls.slice(0, 2).map((url) => <a key={url} href={url} target="_blank" rel="noreferrer">{sourceLabel(url)} <ExternalLink size={12} /></a>)}
                </div>
                {place.photo_attribution ? (
                  <small className="venue-photo-credit">
                    Photo: {place.photo_attribution.uri ? <a href={place.photo_attribution.uri} target="_blank" rel="noreferrer">{place.photo_attribution.name}</a> : place.photo_attribution.name}
                  </small>
                ) : null}
              </article>
            );
          })}
          <p className="venue-provider-note">Place details and photos from Google. Recommendations use live web research; verify details before travelling.</p>
        </div>
      ) : null}
    </aside>
  );
}
