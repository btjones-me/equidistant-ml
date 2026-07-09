import {
  Check,
  ChevronDown,
  FlaskConical,
  LocateFixed,
  LockKeyhole,
  MapPin,
  Menu,
  Plus,
  RotateCcw,
  Search,
  Settings2,
  Sparkles,
  Trash2,
  Users,
  X
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import MapView from "./MapView";
import { getAtlasSurface, preloadAtlas } from "./lib/atlas";
import { useAppState } from "./state/AppStateContext";
import type { CombineMode, Friend, SurfaceCell, SurfaceResponse } from "./types";

const markerColours = ["#0ea5e9", "#f97316", "#22c55e", "#a855f7", "#e11d48", "#64748b"];

const strategyOptions: Array<{ value: CombineMode; label: string; description: string }> = [
  { value: "balanced", label: "Balanced", description: "Short journeys, without leaving anyone behind" },
  { value: "max", label: "Limit the longest", description: "Optimise for the friend with the furthest trip" },
  { value: "mean", label: "Shortest overall", description: "Minimise the group's average journey" }
];

type LocalPlace = {
  name: string;
  lat: number;
  lng: number;
  detail?: string;
};

function emptyResponse(): SurfaceResponse {
  return { lats: [], lngs: [], Z: [], cells: [] };
}

function validCoordinates(friend: Friend): boolean {
  return (
    Number.isFinite(friend.lat) &&
    Number.isFinite(friend.lng) &&
    friend.lat >= 51.2 &&
    friend.lat <= 51.75 &&
    friend.lng >= -0.6 &&
    friend.lng <= 0.35
  );
}

function numericCellValue(cell: SurfaceCell | null, key: string): number | null {
  const value = cell?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function minutes(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "-" : `${Math.round(value)} min`;
}

function locationSummary(friend: Friend): string {
  if (friend.locationLabel?.trim()) {
    return friend.locationLabel;
  }
  return `${friend.lat.toFixed(4)}, ${friend.lng.toFixed(4)}`;
}

function bestCell(cells: SurfaceCell[]): SurfaceCell | null {
  return cells.reduce<SurfaceCell | null>((best, cell) => {
    const score = numericCellValue(cell, "model_score_minutes");
    const bestScore = numericCellValue(best, "model_score_minutes");
    if (score === null) {
      return best;
    }
    return bestScore === null || score < bestScore ? cell : best;
  }, null);
}

function uniqueLocalPlaces(cells: SurfaceCell[], query: string): LocalPlace[] {
  const normalised = query.trim().toLowerCase();
  if (normalised.length < 2) {
    return [];
  }
  const matches = new Map<string, LocalPlace>();
  for (const cell of cells) {
    const name = typeof cell.nearest_station_name === "string" ? cell.nearest_station_name : "";
    if (!name || !name.toLowerCase().includes(normalised) || matches.has(name)) {
      continue;
    }
    matches.set(name, {
      name,
      lat: cell.lat,
      lng: cell.lng,
      detail: typeof cell.nearest_station_lines === "string" ? cell.nearest_station_lines : undefined
    });
    if (matches.size >= 5) {
      break;
    }
  }
  return [...matches.values()];
}

export default function ProductMode({ onDeveloperMode }: { onDeveloperMode: () => void }) {
  const {
    friends,
    included,
    combine,
    focus,
    palette,
    customColorStops,
    colorScale,
    setFriends,
    setIncluded,
    setCombine,
    setFocus,
    setPalette,
    changeFriendCount,
    updateFriend,
    toggleFriend,
    resetWorkspace
  } = useAppState();
  const [surface, setSurface] = useState<SurfaceResponse>(emptyResponse);
  const [selectedCell, setSelectedCell] = useState<SurfaceCell | null>(null);
  const [activeFriendIndex, setActiveFriendIndex] = useState(0);
  const [placingFriendIndex, setPlacingFriendIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [remotePlaces, setRemotePlaces] = useState<LocalPlace[]>([]);
  const requestId = useRef(0);

  const includedFriendIndexes = useMemo(
    () => included.map((value, index) => (value ? index : -1)).filter((index) => index >= 0),
    [included]
  );
  const serialisedStops = useMemo(
    () => customColorStops.map(({ position, color }) => ({ position, color })).sort((a, b) => a.position - b.position),
    [customColorStops]
  );
  const localPlaces = useMemo(() => uniqueLocalPlaces(surface.cells, searchQuery), [searchQuery, surface.cells]);
  const suggestedCell = useMemo(() => bestCell(surface.cells), [surface.cells]);
  const inspectedCell = selectedCell ?? suggestedCell;
  const isSuggested = inspectedCell?.destination_id === suggestedCell?.destination_id;
  const activeStrategy = strategyOptions.find((option) => option.value === combine);

  useEffect(() => {
    void preloadAtlas().catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!friends.every(validCoordinates) || !includedFriendIndexes.length) {
      setError("Place at least one friend within the London coverage area.");
      setLoading(false);
      return;
    }
    const nextRequestId = requestId.current + 1;
    requestId.current = nextRequestId;
    setLoading(true);
    setError(null);
    const timer = window.setTimeout(() => {
      void getAtlasSurface({ friends, includedFriendIndexes, combine, focus })
        .then((response) => {
          if (requestId.current !== nextRequestId) {
            return;
          }
          setSurface(response);
          setSelectedCell((current) => {
            if (!current) {
              return null;
            }
            return response.cells.find((cell) => cell.destination_id === current.destination_id) ?? null;
          });
        })
        .catch((reason) => {
          if (requestId.current === nextRequestId) {
            setError(reason instanceof Error ? reason.message : "The model could not be loaded.");
          }
        })
        .finally(() => {
          if (requestId.current === nextRequestId) {
            setLoading(false);
          }
        });
    }, 80);
    return () => window.clearTimeout(timer);
  }, [combine, focus, friends, includedFriendIndexes]);

  const moveFriend = useCallback(
    (index: number, lat: number, lng: number) => {
      updateFriend(index, { lat, lng, locationLabel: undefined });
      setActiveFriendIndex(index);
    },
    [updateFriend]
  );

  const placeFriend = useCallback(
    (index: number, lat: number, lng: number) => {
      moveFriend(index, lat, lng);
      setPlacingFriendIndex(null);
    },
    [moveFriend]
  );

  function addFriend() {
    if (friends.length >= 6) {
      return;
    }
    const nextIndex = friends.length;
    changeFriendCount(nextIndex + 1);
    setActiveFriendIndex(nextIndex);
    setPlacingFriendIndex(nextIndex);
    setSearchQuery("");
  }

  function removeFriend(index: number) {
    if (friends.length <= 2) {
      return;
    }
    const nextFriends = friends.filter((_, friendIndex) => friendIndex !== index);
    const nextIncluded = included.filter((_, friendIndex) => friendIndex !== index);
    if (!nextIncluded.some(Boolean)) {
      nextIncluded[0] = true;
    }
    setFriends(nextFriends);
    setIncluded(nextIncluded);
    setActiveFriendIndex(Math.min(activeFriendIndex, nextFriends.length - 1));
    setPlacingFriendIndex(null);
  }

  function choosePlace(place: LocalPlace) {
    updateFriend(activeFriendIndex, {
      lat: place.lat,
      lng: place.lng,
      locationLabel: place.name
    });
    setSearchQuery("");
    setRemotePlaces([]);
    setPlacingFriendIndex(null);
  }

  async function searchLocation(event: FormEvent) {
    event.preventDefault();
    const query = searchQuery.trim();
    if (!query) {
      return;
    }
    if (localPlaces.length) {
      choosePlace(localPlaces[0]);
      return;
    }
    setSearching(true);
    setRemotePlaces([]);
    try {
      const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
      if (!response.ok) {
        throw new Error("Location search is temporarily unavailable.");
      }
      const body = (await response.json()) as { results?: LocalPlace[] };
      setRemotePlaces(body.results ?? []);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Location search failed.");
    } finally {
      setSearching(false);
    }
  }

  const areaName =
    (typeof inspectedCell?.nearest_station_name === "string" && inspectedCell.nearest_station_name) || "Central London";
  const score = numericCellValue(inspectedCell, "model_score_minutes");

  return (
    <main className="product-shell">
      <header className="product-topbar">
        <div className="product-brand" aria-label="Equidistant">
          <span className="brand-glyph" aria-hidden="true"><LocateFixed size={20} /></span>
          <span>
            <strong>Equidistant</strong>
            <small>London beta</small>
          </span>
        </div>
        <div className="topbar-actions">
          <span className="offline-badge"><span aria-hidden="true" /> Offline model</span>
          <button
            className="topbar-icon-button"
            type="button"
            aria-label="Open menu"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((current) => !current)}
          >
            <Menu size={20} aria-hidden="true" />
          </button>
          {menuOpen ? (
            <div className="app-menu" role="menu">
              <button type="button" role="menuitem" onClick={onDeveloperMode}>
                <FlaskConical size={17} aria-hidden="true" />
                <span><strong>Developer mode</strong><small>Inspect layers and model diagnostics</small></span>
              </button>
              <button type="button" role="menuitem" onClick={() => { resetWorkspace(); setMenuOpen(false); }}>
                <RotateCcw size={17} aria-hidden="true" />
                <span><strong>Reset workspace</strong><small>Restore the sample group and map style</small></span>
              </button>
              <button type="button" role="menuitem" onClick={() => window.location.assign("/logout")}>
                <LockKeyhole size={17} aria-hidden="true" />
                <span><strong>Lock app</strong><small>Require the preview password again</small></span>
              </button>
            </div>
          ) : null}
        </div>
      </header>

      <section className="product-sidebar" aria-label="Meeting setup">
        <div className="product-intro">
          <p className="product-kicker"><Sparkles size={14} aria-hidden="true" /> Find a fair place to meet</p>
          <h1>Meet in the middle.<br />By time, not miles.</h1>
          <p>Add each starting point. The map shows where the group can arrive in a similar amount of time.</p>
        </div>

        <section className="people-section" aria-labelledby="people-title">
          <div className="product-section-heading">
            <div>
              <h2 id="people-title"><Users size={17} aria-hidden="true" /> Your group</h2>
              <span>{includedFriendIndexes.length} of {friends.length} included</span>
            </div>
            <button className="small-icon-button" type="button" onClick={addFriend} disabled={friends.length >= 6} title="Add friend">
              <Plus size={18} aria-hidden="true" />
            </button>
          </div>

          <div className="participant-list">
            {friends.map((friend, index) => (
              <article
                className={`participant-card${activeFriendIndex === index ? " active" : ""}${included[index] ? "" : " excluded"}`}
                key={friend.id}
              >
                <button
                  className="participant-select"
                  type="button"
                  onClick={() => { setActiveFriendIndex(index); setSearchQuery(""); setRemotePlaces([]); }}
                  aria-label={`Edit ${friend.name || `friend ${index + 1}`}`}
                >
                  <span className="participant-number" style={{ "--participant-colour": markerColours[index % markerColours.length] } as React.CSSProperties}>{index + 1}</span>
                </button>
                <div className="participant-copy">
                  <input
                    aria-label={`Friend ${index + 1} name`}
                    className="participant-name"
                    value={friend.name}
                    onFocus={() => setActiveFriendIndex(index)}
                    onChange={(event) => updateFriend(index, { name: event.target.value })}
                  />
                  <button className="participant-location" type="button" onClick={() => { setActiveFriendIndex(index); setPlacingFriendIndex(index); }}>
                    <MapPin size={13} aria-hidden="true" /> {locationSummary(friend)}
                  </button>
                </div>
                <label className="include-control" title={included[index] ? "Included in the group" : "Excluded from the group"}>
                  <input type="checkbox" checked={included[index]} onChange={() => toggleFriend(index)} />
                  <span><Check size={13} aria-hidden="true" /></span>
                </label>
                <button className="remove-person" type="button" onClick={() => removeFriend(index)} disabled={friends.length <= 2} title="Remove friend">
                  <Trash2 size={15} aria-hidden="true" />
                </button>
              </article>
            ))}
          </div>

          <form className="location-search" onSubmit={searchLocation}>
            <Search size={16} aria-hidden="true" />
            <input
              aria-label={`Search location for ${friends[activeFriendIndex]?.name || "friend"}`}
              placeholder={`Search a place for ${friends[activeFriendIndex]?.name || "friend"}`}
              value={searchQuery}
              onChange={(event) => { setSearchQuery(event.target.value); setRemotePlaces([]); }}
            />
            {searchQuery ? <button type="button" onClick={() => { setSearchQuery(""); setRemotePlaces([]); }} aria-label="Clear location search"><X size={15} /></button> : null}
            <button className="search-submit" type="submit" disabled={searching}>{searching ? "Searching" : "Find"}</button>
            {localPlaces.length || remotePlaces.length ? (
              <div className="place-results" role="listbox" aria-label="Location results">
                {[...localPlaces, ...remotePlaces].slice(0, 5).map((place) => (
                  <button key={`${place.name}-${place.lat}-${place.lng}`} type="button" role="option" onClick={() => choosePlace(place)}>
                    <MapPin size={15} aria-hidden="true" />
                    <span><strong>{place.name}</strong>{place.detail ? <small>{place.detail}</small> : null}</span>
                  </button>
                ))}
              </div>
            ) : null}
          </form>
          <button
            className={placingFriendIndex === activeFriendIndex ? "place-on-map active" : "place-on-map"}
            type="button"
            onClick={() => setPlacingFriendIndex((current) => current === activeFriendIndex ? null : activeFriendIndex)}
          >
            <LocateFixed size={15} aria-hidden="true" />
            {placingFriendIndex === activeFriendIndex ? "Click the map to place them" : "Place on map"}
          </button>
        </section>

        <section className="result-card" aria-live="polite">
          <div className="result-card-heading">
            <span>{isSuggested ? "Suggested meeting area" : "Selected area"}</span>
            <button type="button" onClick={() => setSelectedCell(null)} disabled={isSuggested} title="Return to suggestion"><Sparkles size={15} /></button>
          </div>
          {error ? <p className="product-error">{error}</p> : inspectedCell ? (
            <>
              <div className="result-place">
                <div><MapPin size={20} aria-hidden="true" /></div>
                <span><strong>{areaName}</strong><small>{inspectedCell.nearest_station_lines || "London public transport"}</small></span>
                <b>{minutes(score)}</b>
              </div>
              <div className="journey-times">
                {friends.map((friend, index) => (
                  <div className={included[index] ? "" : "muted"} key={`${friend.id}-journey`}>
                    <span style={{ "--participant-colour": markerColours[index % markerColours.length] } as React.CSSProperties}>{index + 1}</span>
                    <small>{friend.name || `Friend ${index + 1}`}</small>
                    <strong>{minutes(numericCellValue(inspectedCell, `friend_${index}_model_minutes`))}</strong>
                  </div>
                ))}
              </div>
            </>
          ) : <div className="result-skeleton"><span /><span /><span /></div>}
        </section>

        <button className="settings-toggle" type="button" onClick={() => setSettingsOpen((current) => !current)} aria-expanded={settingsOpen}>
          <span><Settings2 size={16} aria-hidden="true" /> {activeStrategy?.label || "Advanced scoring"}</span>
          <ChevronDown size={16} aria-hidden="true" />
        </button>
        {settingsOpen ? (
          <section className="product-settings" aria-label="Map settings">
            <div>
              <p>How should we choose?</p>
              {strategyOptions.map((option) => (
                <button className={combine === option.value ? "strategy-option active" : "strategy-option"} type="button" key={option.value} onClick={() => setCombine(option.value)}>
                  <span>{combine === option.value ? <Check size={14} /> : null}</span>
                  <div><strong>{option.label}</strong><small>{option.description}</small></div>
                </button>
              ))}
            </div>
            <div className="compact-setting-row">
              <span>Coverage</span>
              <div className="mini-segmented">
                {(["central", "inner"] as const).map((value) => <button className={focus === value ? "active" : ""} type="button" key={value} onClick={() => setFocus(value)}>{value === "central" ? "Zone 1" : "Zones 1-3"}</button>)}
              </div>
            </div>
            <div className="compact-setting-row">
              <span>Colours</span>
              <div className="palette-swatches">
                {(["central", "green-red", "viridis", "inferno", "custom"] as const).map((value) => <button className={palette === value ? "active" : ""} aria-label={`${value} colour map`} title={value} type="button" key={value} onClick={() => setPalette(value)} data-palette={value} />)}
              </div>
            </div>
          </section>
        ) : null}
      </section>

      <section className={`product-map-panel${placingFriendIndex !== null ? " placing" : ""}`} aria-label="Travel-time map">
        {placingFriendIndex !== null ? <div className="placement-banner"><MapPin size={15} /> Click anywhere to move {friends[placingFriendIndex]?.name || "this friend"}</div> : null}
        <MapView
          friends={friends}
          includedFriendIndexes={includedFriendIndexes}
          cells={surface.cells}
          selectedCell={inspectedCell}
          valueKey="model_score_minutes"
          valueLabel="Group travel time"
          palette={palette}
          customColorMap={serialisedStops}
          colorScale={colorScale}
          isLoading={loading}
          loadingLabel="Finding your middle"
          onSelectCell={setSelectedCell}
          variant="product"
          activeFriendIndex={activeFriendIndex}
          placingFriendIndex={placingFriendIndex}
          onMoveFriend={moveFriend}
          onPlaceFriend={placeFriend}
        />
        <div className="map-model-note">
          <span aria-hidden="true" />
          Offline estimate · weekday morning · public transport
        </div>
      </section>
    </main>
  );
}
