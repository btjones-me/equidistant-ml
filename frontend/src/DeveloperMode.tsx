import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Copy, Database, Plus, RotateCcw, RefreshCw, Trash2 } from "lucide-react";
import MapView from "./MapView";
import { getAtlasSurface } from "./lib/atlas";
import { DEFAULT_SURFACE_OPACITY, recommendedColorScale, useAppState } from "./state/AppStateContext";
import type {
  ColorMapStop,
  ColorScale,
  CombineMode,
  ComparisonMetrics,
  DetailMode,
  EditableColorMapStop,
  FocusMode,
  Friend,
  PaletteMode,
  SurfaceCell,
  SurfaceResponse,
  ViewMode
} from "./types";

const combineModes: CombineMode[] = ["balanced", "max", "mean", "fairness"];
const detailModes: DetailMode[] = ["fine", "fast"];
const paletteModes: Exclude<PaletteMode, "error">[] = ["central", "green-red", "viridis", "inferno", "custom"];
const viewModes: ViewMode[] = ["model", "graph", "residual", "reference", "error"];
const maxSurfaceCacheItems = 12;
const maxComparisonCacheItems = 6;
let customStopCounter = 0;

type SurfaceRequestPayload = {
  friends: Friend[];
  combine: CombineMode;
  included_friend_indexes: number[];
  grid_mode: "h3";
  focus: FocusMode;
  detail: DetailMode;
};

type UsageLocation = {
  country: string;
  region: string;
  city: string;
  unique_visitors: number;
  unlocks: number;
};

type UsageSummary = {
  unique_visitors: {
    all_time: number;
    last_24h: number;
    last_7d: number;
    last_30d: number;
  };
  total_unlocks: number;
  last_seen_at: number | null;
  locations: UsageLocation[];
};

function emptyResponse(): SurfaceResponse {
  return { lats: [], lngs: [], Z: [], cells: [] };
}

function formatMinutes(value?: number | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "-";
  }
  return `${value.toFixed(1)} min`;
}

function formatPct(value?: number | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "-";
  }
  return `${value.toFixed(0)}%`;
}

function formatSigned(value?: number | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)} min`;
}

function formatSecondsAsMinutes(value?: number | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "-";
  }
  return `${(value / 60).toFixed(1)} min`;
}

function cellValue(cell: SurfaceCell | null, key: string): number | undefined {
  if (!cell) {
    return undefined;
  }
  const value = cell[key];
  return typeof value === "number" ? value : undefined;
}

function cellText(cell: SurfaceCell | null, key: string): string {
  if (!cell) {
    return "-";
  }
  const value = cell[key];
  return typeof value === "string" && value.trim() ? value : "-";
}

function responseCellById(response: SurfaceResponse | null, destinationId?: string): SurfaceCell | null {
  if (!response || !destinationId) {
    return null;
  }
  return response.cells.find((cell) => cell.destination_id === destinationId) ?? null;
}

function hasValidLondonCoordinates(friends: Friend[]): boolean {
  return friends.every(
    (friend) =>
      Number.isFinite(friend.lat) &&
      Number.isFinite(friend.lng) &&
      friend.lat >= 51.2 &&
      friend.lat <= 51.75 &&
      friend.lng >= -0.6 &&
      friend.lng <= 0.35
  );
}

function viewValueKey(view: ViewMode, hasComparison: boolean): string {
  if (view === "graph") {
    return "graph_score_minutes";
  }
  if (view === "residual") {
    return "model_residual_minutes";
  }
  if (view === "reference" && hasComparison) {
    return "reference_score_minutes";
  }
  if (view === "error" && hasComparison) {
    return "signed_error_minutes";
  }
  return "model_score_minutes";
}

function viewLabel(view: ViewMode, hasComparison: boolean): string {
  if (view === "graph") {
    return "Graph baseline";
  }
  if (view === "residual") {
    return "Model residual";
  }
  if (view === "reference" && hasComparison) {
    return "TravelTime";
  }
  if (view === "error" && hasComparison) {
    return "Signed error";
  }
  return "Model";
}

function metricRows(metrics?: ComparisonMetrics) {
  if (!metrics) {
    return [];
  }
  return [
    { label: "MAE", value: formatMinutes(metrics.mae_minutes) },
    { label: "Median error", value: formatMinutes(metrics.median_abs_error_minutes) },
    { label: "P90 error", value: formatMinutes(metrics.p90_abs_error_minutes) },
    { label: "Within 5 min", value: formatPct(metrics.within_5_min_pct) },
    { label: "Within 10 min", value: formatPct(metrics.within_10_min_pct) },
    { label: "Bias", value: formatSigned(metrics.mean_signed_error_minutes) }
  ];
}

function clampColorScale(next: ColorScale): ColorScale {
  const lower = Math.max(0, Math.min(45, Math.round(next.lowerPercentile)));
  const upper = Math.max(55, Math.min(100, Math.round(next.upperPercentile)));
  return {
    lowerPercentile: Math.min(lower, upper - 5),
    upperPercentile: Math.max(upper, lower + 5),
    contrast: Math.max(0.6, Math.min(2.4, Number(next.contrast.toFixed(2))))
  };
}

function cacheGet<T>(cache: Map<string, T>, key: string): T | undefined {
  const value = cache.get(key);
  if (value !== undefined) {
    cache.delete(key);
    cache.set(key, value);
  }
  return value;
}

function cacheSet<T>(cache: Map<string, T>, key: string, value: T, maxItems: number) {
  if (cache.has(key)) {
    cache.delete(key);
  }
  cache.set(key, value);
  while (cache.size > maxItems) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey === undefined) {
      return;
    }
    cache.delete(oldestKey);
  }
}

function clampStopPosition(value: number): number {
  return Math.max(0, Math.min(100, Number(value.toFixed(1))));
}

function normaliseHexColor(value: string): string {
  const trimmed = value.trim();
  const six = /^#?([0-9a-fA-F]{6})$/.exec(trimmed);
  if (six) {
    return `#${six[1].toLowerCase()}`;
  }
  const three = /^#?([0-9a-fA-F]{3})$/.exec(trimmed);
  if (three) {
    return `#${three[1]
      .split("")
      .map((part) => `${part}${part}`)
      .join("")
      .toLowerCase()}`;
  }
  return value;
}

function isValidHexColor(value: string): boolean {
  return /^#[0-9a-f]{6}$/.test(normaliseHexColor(value));
}

function makeColorStop(position: number, color: string): EditableColorMapStop {
  customStopCounter += 1;
  return {
    id: `custom-stop-${customStopCounter}`,
    position: clampStopPosition(position),
    color: normaliseHexColor(color)
  };
}

function serialiseColorStops(stops: EditableColorMapStop[]): ColorMapStop[] {
  return stops
    .map((stop) => ({
      position: clampStopPosition(stop.position),
      color: normaliseHexColor(stop.color)
    }))
    .filter((stop) => isValidHexColor(stop.color))
    .sort((left, right) => left.position - right.position);
}

function gradientFromStops(stops: ColorMapStop[]): string {
  if (!stops.length) {
    return "linear-gradient(90deg, #15803d 0%, #7f1d1d 100%)";
  }
  const sorted = [...stops].sort((left, right) => left.position - right.position);
  return `linear-gradient(90deg, ${sorted.map((stop) => `${stop.color} ${stop.position}%`).join(", ")})`;
}

function hexToRgb(value: string): [number, number, number] {
  const color = normaliseHexColor(value);
  return [
    Number.parseInt(color.slice(1, 3), 16),
    Number.parseInt(color.slice(3, 5), 16),
    Number.parseInt(color.slice(5, 7), 16)
  ];
}

function rgbToHex(rgb: [number, number, number]): string {
  return `#${rgb.map((channel) => Math.round(channel).toString(16).padStart(2, "0")).join("")}`;
}

function interpolateColor(stops: ColorMapStop[], position: number): string {
  const sorted = [...stops].sort((left, right) => left.position - right.position);
  if (!sorted.length) {
    return "#facc15";
  }
  const rightIndex = sorted.findIndex((stop) => stop.position >= position);
  if (rightIndex <= 0) {
    return sorted[0].color;
  }
  const left = sorted[rightIndex - 1];
  const right = sorted[rightIndex] ?? sorted[sorted.length - 1];
  const span = Math.max(right.position - left.position, 0.1);
  const local = (position - left.position) / span;
  const leftRgb = hexToRgb(left.color);
  const rightRgb = hexToRgb(right.color);
  return rgbToHex(leftRgb.map((channel, index) => channel + (rightRgb[index] - channel) * local) as [number, number, number]);
}

function nextStopPosition(stops: ColorMapStop[]): number {
  const sorted = [...stops].sort((left, right) => left.position - right.position);
  if (sorted.length < 2) {
    return 50;
  }
  let bestStart = sorted[0].position;
  let bestEnd = sorted[1].position;
  for (let index = 1; index < sorted.length - 1; index += 1) {
    const left = sorted[index];
    const right = sorted[index + 1];
    if (right.position - left.position > bestEnd - bestStart) {
      bestStart = left.position;
      bestEnd = right.position;
    }
  }
  return clampStopPosition((bestStart + bestEnd) / 2);
}

export default function DeveloperMode({ onExit }: { onExit: () => void }) {
  const {
    friends,
    included,
    combine,
    focus,
    detail,
    mapStyle,
    palette,
    customColorStops,
    colorScale,
    surfaceOpacity,
    suggestionMinDistanceKm,
    setCombine,
    setDetail,
    setMapStyle,
    setPalette,
    setCustomColorStops,
    setColorScale,
    setSurfaceOpacity,
    setSuggestionMinDistanceKm,
    changeFriendCount: setSharedFriendCount,
    updateFriend: patchFriend,
    toggleFriend: toggleSharedFriend
  } = useAppState();
  const friendCount = friends.length;
  const [selectedColorStopId, setSelectedColorStopId] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState("");
  const [view, setView] = useState<ViewMode>("model");
  const [surface, setSurface] = useState<SurfaceResponse>(emptyResponse);
  const [comparison, setComparison] = useState<SurfaceResponse | null>(null);
  const [selectedCell, setSelectedCell] = useState<SurfaceCell | null>(null);
  const [loading, setLoading] = useState(false);
  const [referenceLoading, setReferenceLoading] = useState(false);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [usageLoading, setUsageLoading] = useState(false);
  const [usageError, setUsageError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const surfaceRequestId = useRef(0);
  const comparisonRequestId = useRef(0);
  const surfaceCache = useRef<Map<string, SurfaceResponse>>(new Map());
  const comparisonCache = useRef<Map<string, SurfaceResponse>>(new Map());
  const spectrumRef = useRef<HTMLDivElement | null>(null);
  const draggingColorStopId = useRef<string | null>(null);
  const suppressSpectrumClick = useRef(false);

  const includedFriendIndexes = useMemo(
    () => included.map((isIncluded, index) => (isIncluded ? index : -1)).filter((index) => index >= 0),
    [included]
  );
  const surfaceRequestPayload = useMemo<SurfaceRequestPayload>(
    () => ({
      friends,
      combine,
      included_friend_indexes: includedFriendIndexes,
      grid_mode: "h3",
      focus,
      detail
    }),
    [combine, detail, focus, friends, includedFriendIndexes]
  );
  const requestKey = useMemo(() => JSON.stringify(surfaceRequestPayload), [surfaceRequestPayload]);
  const hasComparison = Boolean(comparison?.cells.length);
  const activeResponse = hasComparison ? comparison : surface;
  const isBrowserAtlas = surface.metadata?.source === "browser_atlas";
  const hasGraph = Boolean(
    activeResponse?.cells.some((cell) => typeof cell.graph_score_minutes === "number")
  );
  const selectedCellForActiveResponse =
    responseCellById(activeResponse, selectedCell?.destination_id) ?? selectedCell;
  const valueKey = viewValueKey(view, hasComparison);
  const valueLabel = viewLabel(view, hasComparison);
  const mapPalette: PaletteMode = view === "error" && hasComparison ? "error" : palette;
  const comparisonMetrics = comparison?.metadata?.comparison;
  const activeRecommendedColorScale = useMemo(
    () => recommendedColorScale(includedFriendIndexes.length),
    [includedFriendIndexes.length]
  );
  const customColorMap = useMemo(() => serialiseColorStops(customColorStops), [customColorStops]);
  const customColorMapJson = useMemo(
    () =>
      JSON.stringify(
        {
          name: "custom",
          stops: customColorMap
        },
        null,
        2
      ),
    [customColorMap]
  );
  const customGradient = useMemo(() => gradientFromStops(customColorMap), [customColorMap]);
  const customMarkerStops = useMemo(() => {
    if (!selectedColorStopId) {
      return customColorStops;
    }
    const selected = customColorStops.find((stop) => stop.id === selectedColorStopId);
    if (!selected) {
      return customColorStops;
    }
    return [...customColorStops.filter((stop) => stop.id !== selectedColorStopId), selected];
  }, [customColorStops, selectedColorStopId]);
  const selectedCustomStop =
    customColorStops.find((stop) => stop.id === selectedColorStopId) ?? customColorStops[0] ?? null;
  const selectedScore =
    cellValue(selectedCellForActiveResponse, valueKey) ??
    cellValue(selectedCellForActiveResponse, "model_score_minutes") ??
    cellValue(selectedCellForActiveResponse, "score_minutes");

  const requestSurface = useCallback(async (forceRefresh = false) => {
    if (!includedFriendIndexes.length) {
      setError("Select at least one friend.");
      return;
    }
    if (!hasValidLondonCoordinates(friends)) {
      setError("Enter valid London latitude/longitude values before refreshing.");
      return;
    }
    const cached = forceRefresh ? undefined : cacheGet(surfaceCache.current, requestKey);
    if (cached) {
      surfaceRequestId.current += 1;
      setSurface(cached);
      setSelectedCell((current) => responseCellById(cached, current?.destination_id));
      setLoading(false);
      setError(null);
      return;
    }
    const requestId = surfaceRequestId.current + 1;
    surfaceRequestId.current = requestId;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/group-surface", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestKey
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const body = (await response.json()) as SurfaceResponse;
      if (surfaceRequestId.current === requestId) {
        cacheSet(surfaceCache.current, requestKey, body, maxSurfaceCacheItems);
        setSurface(body);
        setSelectedCell((current) => responseCellById(body, current?.destination_id));
      }
    } catch {
      try {
        const body = await getAtlasSurface({
          friends,
          includedFriendIndexes,
          combine,
          focus,
          includeGraph: true
        });
        if (surfaceRequestId.current === requestId) {
          cacheSet(surfaceCache.current, requestKey, body, maxSurfaceCacheItems);
          setSurface(body);
          setSelectedCell((current) => responseCellById(body, current?.destination_id));
          setError(null);
        }
      } catch (fallbackError) {
        if (surfaceRequestId.current === requestId) {
          setError(fallbackError instanceof Error ? fallbackError.message : "Surface request failed");
        }
      }
    } finally {
      if (surfaceRequestId.current === requestId) {
        setLoading(false);
      }
    }
  }, [combine, focus, friends, includedFriendIndexes, requestKey]);

  const refreshSurface = useCallback(() => {
    void requestSurface(true);
  }, [requestSurface]);

  const requestComparison = useCallback(async () => {
    if (!includedFriendIndexes.length) {
      setError("Select at least one friend.");
      return;
    }
    if (!hasValidLondonCoordinates(friends)) {
      setError("Enter valid London latitude/longitude values before fetching a reference.");
      return;
    }
    const cached = cacheGet(comparisonCache.current, requestKey);
    if (cached) {
      comparisonRequestId.current += 1;
      setComparison(cached);
      setView("reference");
      setSelectedCell((current) => responseCellById(cached, current?.destination_id));
      setReferenceLoading(false);
      setError(null);
      return;
    }
    const requestId = comparisonRequestId.current + 1;
    comparisonRequestId.current = requestId;
    setReferenceLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/comparison-surface", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestKey
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const body = (await response.json()) as SurfaceResponse;
      if (comparisonRequestId.current === requestId) {
        cacheSet(comparisonCache.current, requestKey, body, maxComparisonCacheItems);
        setComparison(body);
        setView("reference");
        setSelectedCell((current) => responseCellById(body, current?.destination_id));
      }
    } catch (err) {
      if (comparisonRequestId.current === requestId) {
        setError(err instanceof Error ? err.message : "TravelTime comparison failed");
      }
    } finally {
      if (comparisonRequestId.current === requestId) {
        setReferenceLoading(false);
      }
    }
  }, [friends, includedFriendIndexes.length, requestKey]);

  const loadUsage = useCallback(async () => {
    setUsageLoading(true);
    setUsageError(null);
    try {
      const response = await fetch("/api/usage", { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Usage data is unavailable in this environment.");
      }
      setUsage((await response.json()) as UsageSummary);
    } catch (reason) {
      setUsageError(reason instanceof Error ? reason.message : "Usage data is unavailable.");
    } finally {
      setUsageLoading(false);
    }
  }, []);

  useEffect(() => {
    requestSurface();
  }, [requestSurface]);

  useEffect(() => {
    setComparison(cacheGet(comparisonCache.current, requestKey) ?? null);
    setView("model");
  }, [requestKey]);

  useEffect(() => {
    setColorScale(activeRecommendedColorScale);
  }, [activeRecommendedColorScale]);

  function updateFriend(index: number, field: "name" | "lat" | "lng", value: string) {
    const parsed = field === "name" ? value : value.trim() === "" ? Number.NaN : Number(value);
    patchFriend(index, { [field]: parsed });
  }

  function changeFriendCount(value: number) {
    setSharedFriendCount(value);
  }

  function toggleFriend(index: number) {
    toggleSharedFriend(index);
  }

  function updateColorScale(field: keyof ColorScale, value: number) {
    setColorScale((current) => clampColorScale({ ...current, [field]: value }));
  }

  function spectrumPositionFromClientX(clientX: number): number | null {
    const rect = spectrumRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) {
      return null;
    }
    return clampStopPosition(((clientX - rect.left) / rect.width) * 100);
  }

  function moveCustomStopToPointer(id: string, clientX: number) {
    const position = spectrumPositionFromClientX(clientX);
    if (position === null) {
      return;
    }
    updateCustomStop(id, { position });
  }

  function addCustomStop(position = 50) {
    const nextPosition = clampStopPosition(position);
    const stop = makeColorStop(nextPosition, interpolateColor(customColorMap, nextPosition));
    setCustomColorStops((current) => [...current, stop]);
    setSelectedColorStopId(stop.id);
    setPalette("custom");
    setCopyStatus("");
  }

  function updateCustomStop(id: string, patch: Partial<ColorMapStop>) {
    setCustomColorStops((current) =>
      current.map((stop) => {
        if (stop.id !== id) {
          return stop;
        }
        return {
          ...stop,
          position: patch.position === undefined ? stop.position : clampStopPosition(patch.position),
          color: patch.color === undefined ? stop.color : normaliseHexColor(patch.color)
        };
      })
    );
    setCopyStatus("");
  }

  function removeCustomStop(id: string) {
    setCustomColorStops((current) => {
      if (current.length <= 2) {
        return current;
      }
      const next = current.filter((stop) => stop.id !== id);
      if (selectedColorStopId === id) {
        setSelectedColorStopId(next.at(-1)?.id ?? null);
      }
      return next;
    });
    setCopyStatus("");
  }

  async function copyCustomColorMap() {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(customColorMapJson);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = customColorMapJson;
        textarea.setAttribute("readonly", "true");
        textarea.style.position = "fixed";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      setCopyStatus("Copied");
    } catch {
      setCopyStatus("Copy failed");
    }
  }

  return (
    <main className="app-shell">
      <section className="control-panel" aria-label="Surface controls">
        <div className="brand-row">
          <div>
            <button className="developer-exit" type="button" onClick={onExit}>
              <ArrowLeft size={14} aria-hidden="true" />
              Equidistant
            </button>
            <h1>Developer mode</h1>
          </div>
          <button className="icon-button" type="button" onClick={refreshSurface} disabled={loading} title="Refresh model surface">
            <RefreshCw size={18} aria-hidden="true" />
          </button>
        </div>

        <div className="field-row">
          <label>
            Friends
            <input
              type="number"
              min={1}
              max={6}
              value={friendCount}
              onChange={(event) => changeFriendCount(Number(event.target.value))}
            />
          </label>
          <label>
            Suggestion spacing (km)
            <input
              type="number"
              min={0.5}
              max={20}
              step={0.5}
              value={suggestionMinDistanceKm}
              onChange={(event) => setSuggestionMinDistanceKm(Math.min(20, Math.max(0.5, Number(event.target.value))))}
            />
          </label>
          <label>
            Cells
            <output>{activeResponse?.metadata?.cell_count?.toLocaleString() ?? activeResponse?.cells.length.toLocaleString() ?? "0"}</output>
          </label>
        </div>

        <section className="usage-panel" aria-label="Usage analytics">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Usage</p>
              {usage?.last_seen_at ? (
                <span className="usage-last-seen">
                  Last unlock {new Date(usage.last_seen_at * 1000).toLocaleString()}
                </span>
              ) : null}
            </div>
            <button
              className="reset-button"
              type="button"
              onClick={loadUsage}
              disabled={usageLoading}
              title="Refresh usage"
            >
              <RefreshCw size={14} aria-hidden="true" />
            </button>
          </div>
          {usage ? (
            <>
              <div className="metric-grid usage-metrics">
                <div className="metric-card">
                  <span>24 hours</span>
                  <strong>{usage.unique_visitors.last_24h.toLocaleString()}</strong>
                </div>
                <div className="metric-card">
                  <span>7 days</span>
                  <strong>{usage.unique_visitors.last_7d.toLocaleString()}</strong>
                </div>
                <div className="metric-card">
                  <span>30 days</span>
                  <strong>{usage.unique_visitors.last_30d.toLocaleString()}</strong>
                </div>
                <div className="metric-card">
                  <span>All time</span>
                  <strong>{usage.unique_visitors.all_time.toLocaleString()}</strong>
                </div>
                <div className="metric-card">
                  <span>Unlocks</span>
                  <strong>{usage.total_unlocks.toLocaleString()}</strong>
                </div>
              </div>
              {usage.locations.length ? (
                <div className="usage-location-list">
                  {usage.locations.map((location) => (
                    <div
                      className="usage-location-row"
                      key={`${location.country}-${location.region}-${location.city}`}
                    >
                      <span>{[location.city, location.region, location.country].join(", ")}</span>
                      <strong>
                        {location.unique_visitors.toLocaleString()} / {location.unlocks.toLocaleString()}
                      </strong>
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <p className="muted-copy">
              {usageError ?? (usageLoading ? "Loading usage..." : "No usage summary loaded.")}
            </p>
          )}
        </section>

        <div className="control-grid">
          <div>
            <p className="eyebrow">Detail</p>
            <div className="segmented two" aria-label="Detail">
              {detailModes.map((mode) => (
                <button key={mode} className={detail === mode ? "active" : ""} type="button" onClick={() => setDetail(mode)}>
                  {mode}
                </button>
              ))}
            </div>
          </div>
          <div>
            <p className="eyebrow">Colours</p>
            <div className="segmented five" aria-label="Colour map">
              {paletteModes.map((mode) => (
                <button key={mode} className={palette === mode ? "active" : ""} type="button" onClick={() => setPalette(mode)}>
                  {mode}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="control-group">
          <p className="eyebrow">Basemap</p>
          <div className="segmented three" aria-label="Basemap style">
            <button className={mapStyle === "positron" ? "active" : ""} type="button" onClick={() => setMapStyle("positron")}>
              light
            </button>
            <button className={mapStyle === "voyager" ? "active" : ""} type="button" onClick={() => setMapStyle("voyager")}>
              street
            </button>
            <button className={mapStyle === "dark-matter" ? "active" : ""} type="button" onClick={() => setMapStyle("dark-matter")}>
              dark
            </button>
          </div>
        </div>

        {palette === "custom" ? (
          <section className="custom-map-panel" aria-label="Custom colour map">
            <div className="section-heading">
              <p className="eyebrow">Custom spectrum</p>
              <div className="tool-row">
                <button
                  className="reset-button"
                  type="button"
                  onClick={() => addCustomStop(nextStopPosition(customColorMap))}
                  title="Add marker"
                >
                  <Plus size={14} aria-hidden="true" />
                </button>
                <button className="reset-button" type="button" onClick={copyCustomColorMap} title="Copy JSON">
                  <Copy size={14} aria-hidden="true" />
                </button>
              </div>
            </div>
            <div
              className="custom-spectrum"
              ref={spectrumRef}
              onClick={(event) => {
                if (suppressSpectrumClick.current) {
                  suppressSpectrumClick.current = false;
                  return;
                }
                const position = spectrumPositionFromClientX(event.clientX);
                if (position !== null) {
                  addCustomStop(position);
                }
              }}
              role="presentation"
            >
              <div className="custom-spectrum-ramp" style={{ background: customGradient }} />
              {customMarkerStops.map((stop) => (
                <button
                  className={selectedCustomStop?.id === stop.id ? "custom-stop-marker active" : "custom-stop-marker"}
                  key={stop.id}
                  style={{ left: `${clampStopPosition(stop.position)}%`, background: isValidHexColor(stop.color) ? normaliseHexColor(stop.color) : "#ffffff" }}
                  type="button"
                  aria-label={`Drag marker at ${clampStopPosition(stop.position)} percent`}
                  onPointerDown={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    event.currentTarget.setPointerCapture(event.pointerId);
                    draggingColorStopId.current = stop.id;
                    suppressSpectrumClick.current = false;
                    setSelectedColorStopId(stop.id);
                    setPalette("custom");
                    moveCustomStopToPointer(stop.id, event.clientX);
                  }}
                  onPointerMove={(event) => {
                    if (draggingColorStopId.current !== stop.id) {
                      return;
                    }
                    event.preventDefault();
                    event.stopPropagation();
                    suppressSpectrumClick.current = true;
                    moveCustomStopToPointer(stop.id, event.clientX);
                  }}
                  onPointerUp={(event) => {
                    if (draggingColorStopId.current !== stop.id) {
                      return;
                    }
                    event.preventDefault();
                    event.stopPropagation();
                    draggingColorStopId.current = null;
                    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                      event.currentTarget.releasePointerCapture(event.pointerId);
                    }
                    window.setTimeout(() => {
                      suppressSpectrumClick.current = false;
                    }, 0);
                  }}
                  onPointerCancel={(event) => {
                    if (draggingColorStopId.current !== stop.id) {
                      return;
                    }
                    event.preventDefault();
                    event.stopPropagation();
                    draggingColorStopId.current = null;
                    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                      event.currentTarget.releasePointerCapture(event.pointerId);
                    }
                    window.setTimeout(() => {
                      suppressSpectrumClick.current = false;
                    }, 0);
                  }}
                  onLostPointerCapture={() => {
                    if (draggingColorStopId.current === stop.id) {
                      draggingColorStopId.current = null;
                    }
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    setSelectedColorStopId(stop.id);
                  }}
                  title={`${clampStopPosition(stop.position)}% ${normaliseHexColor(stop.color)}`}
                />
              ))}
            </div>
            <div className="custom-stop-list">
              {customColorStops.map((stop) => {
                const color = normaliseHexColor(stop.color);
                const validColor = isValidHexColor(color) ? color : "#000000";
                return (
                  <div className={selectedCustomStop?.id === stop.id ? "custom-stop-row active" : "custom-stop-row"} key={stop.id}>
                    <button
                      className="swatch-button"
                      style={{ background: validColor }}
                      type="button"
                      onClick={() => setSelectedColorStopId(stop.id)}
                      title="Select marker"
                    />
                    <input
                      aria-label="Marker position"
                      type="number"
                      min={0}
                      max={100}
                      step={0.1}
                      value={stop.position}
                      onChange={(event) => updateCustomStop(stop.id, { position: Number(event.target.value) })}
                      onFocus={() => setSelectedColorStopId(stop.id)}
                    />
                    <input
                      aria-label="Marker colour"
                      type="color"
                      value={validColor}
                      onChange={(event) => updateCustomStop(stop.id, { color: event.target.value })}
                      onFocus={() => setSelectedColorStopId(stop.id)}
                    />
                    <input
                      aria-label="Marker hex code"
                      value={stop.color}
                      onChange={(event) => updateCustomStop(stop.id, { color: event.target.value })}
                      onBlur={(event) => updateCustomStop(stop.id, { color: normaliseHexColor(event.target.value) })}
                      onFocus={() => setSelectedColorStopId(stop.id)}
                    />
                    <button
                      className="reset-button"
                      type="button"
                      onClick={() => removeCustomStop(stop.id)}
                      disabled={customColorStops.length <= 2}
                      title="Remove marker"
                    >
                      <Trash2 size={14} aria-hidden="true" />
                    </button>
                  </div>
                );
              })}
            </div>
            {copyStatus ? <p className="custom-copy-status">{copyStatus}</p> : null}
          </section>
        ) : null}

        <section className="scale-panel" aria-label="Surface appearance">
          <div className="section-heading">
            <p className="eyebrow">Surface appearance</p>
            <button
              className="reset-button"
              type="button"
              onClick={() => {
                setColorScale(activeRecommendedColorScale);
                setSurfaceOpacity(DEFAULT_SURFACE_OPACITY);
              }}
              title="Reset surface appearance"
            >
              <RotateCcw size={14} aria-hidden="true" />
            </button>
          </div>
          <div className="slider-grid">
            <label>
              <span>
                Low
                <input
                  aria-label="Low percentile"
                  className="scale-value"
                  type="number"
                  min={0}
                  max={45}
                  step={1}
                  value={colorScale.lowerPercentile}
                  onChange={(event) => updateColorScale("lowerPercentile", Number(event.target.value))}
                />
              </span>
              <input
                type="range"
                min={0}
                max={45}
                step={1}
                value={colorScale.lowerPercentile}
                onChange={(event) => updateColorScale("lowerPercentile", Number(event.target.value))}
              />
            </label>
            <label>
              <span>
                High
                <input
                  aria-label="High percentile"
                  className="scale-value"
                  type="number"
                  min={55}
                  max={100}
                  step={1}
                  value={colorScale.upperPercentile}
                  onChange={(event) => updateColorScale("upperPercentile", Number(event.target.value))}
                />
              </span>
              <input
                type="range"
                min={55}
                max={100}
                step={1}
                value={colorScale.upperPercentile}
                onChange={(event) => updateColorScale("upperPercentile", Number(event.target.value))}
              />
            </label>
            <label>
              <span>
                Contrast
                <input
                  aria-label="Colour contrast"
                  className="scale-value"
                  type="number"
                  min={0.6}
                  max={2.4}
                  step={0.05}
                  value={colorScale.contrast}
                  onChange={(event) => updateColorScale("contrast", Number(event.target.value))}
                />
              </span>
              <input
                type="range"
                min={0.6}
                max={2.4}
                step={0.05}
                value={colorScale.contrast}
                onChange={(event) => updateColorScale("contrast", Number(event.target.value))}
              />
            </label>
            <label>
              <span>
                Heatmap opacity
                <input
                  aria-label="Heatmap opacity"
                  className="scale-value"
                  type="number"
                  min={15}
                  max={90}
                  step={1}
                  value={Math.round(surfaceOpacity * 100)}
                  onChange={(event) => setSurfaceOpacity(Math.min(0.9, Math.max(0.15, Number(event.target.value) / 100)))}
                />
              </span>
              <input
                aria-label="Heatmap opacity slider"
                type="range"
                min={15}
                max={90}
                step={1}
                value={Math.round(surfaceOpacity * 100)}
                onChange={(event) => setSurfaceOpacity(Number(event.target.value) / 100)}
              />
            </label>
          </div>
        </section>

        <div className="control-group">
          <p className="eyebrow">Group score</p>
          <div className="segmented" aria-label="Combine mode">
            {combineModes.map((mode) => (
              <button key={mode} className={combine === mode ? "active" : ""} type="button" onClick={() => setCombine(mode)}>
                {mode}
              </button>
            ))}
          </div>
        </div>

        <section className="participant-panel" aria-label="Participants">
          <div className="section-heading">
            <p className="eyebrow">Participants</p>
            <strong>{includedFriendIndexes.length} active</strong>
          </div>
          <div className="friends-list">
            {friends.map((friend, index) => (
              <div className={included[index] ? "friend-card included" : "friend-card"} key={`${friend.name}-${index}`}>
                <label className="friend-toggle">
                  <input
                    aria-label={`Include ${friend.name || `Friend ${index + 1}`}`}
                    type="checkbox"
                    checked={included[index]}
                    onChange={() => toggleFriend(index)}
                  />
                  <span>{index + 1}</span>
                </label>
                <input
                  aria-label={`Friend ${index + 1} name`}
                  value={friend.name}
                  onChange={(event) => updateFriend(index, "name", event.target.value)}
                />
                <input
                  aria-label={`Friend ${index + 1} latitude`}
                  type="number"
                  step="0.000001"
                  value={Number.isFinite(friend.lat) ? friend.lat : ""}
                  onChange={(event) => updateFriend(index, "lat", event.target.value)}
                />
                <input
                  aria-label={`Friend ${index + 1} longitude`}
                  type="number"
                  step="0.000001"
                  value={Number.isFinite(friend.lng) ? friend.lng : ""}
                  onChange={(event) => updateFriend(index, "lng", event.target.value)}
                />
              </div>
            ))}
          </div>
        </section>

        <section className="comparison-panel" aria-label="Model comparison">
          <div className="section-heading">
            <p className="eyebrow">Inspect</p>
            <button className="action-button" type="button" onClick={requestComparison} disabled={referenceLoading || isBrowserAtlas} title={isBrowserAtlas ? "Live references are available when the local API is running" : "Fetch a TravelTime reference"}>
              <Database size={16} aria-hidden="true" />
              {referenceLoading ? "Fetching..." : "Fetch TravelTime"}
            </button>
          </div>
          <div className="segmented five" aria-label="Map value">
            {viewModes.map((mode) => (
              <button
                key={mode}
                className={view === mode ? "active" : ""}
                type="button"
                disabled={
                  (mode === "reference" || mode === "error") ? !hasComparison : (mode === "graph" || mode === "residual") && !hasGraph
                }
                onClick={() => setView(mode)}
              >
                {mode}
              </button>
            ))}
          </div>
          {comparisonMetrics ? (
            <div className="metric-grid">
              {metricRows(comparisonMetrics).map((row) => (
                <div className="metric-card" key={row.label}>
                  <span>{row.label}</span>
                  <strong>{row.value}</strong>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted-copy">
              {isBrowserAtlas
                ? "This hosted view uses the browser atlas. Run the local API to fetch live TravelTime references."
                : "Fetch a TravelTime reference to compare the offline model against ground truth for the checked participants."}
            </p>
          )}
        </section>

        {error ? <div className="error-banner">{error}</div> : null}

        <section className="summary-panel" aria-label="Selected destination">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Selected cell</p>
              <h2>{formatMinutes(selectedScore)}</h2>
            </div>
            <span className="value-pill">{valueLabel}</span>
          </div>
          {selectedCellForActiveResponse ? (
            <>
              <dl>
                <div>
                  <dt>Graph baseline</dt>
                  <dd>{formatMinutes(cellValue(selectedCellForActiveResponse, "graph_score_minutes"))}</dd>
                </div>
                <div>
                  <dt>Residual</dt>
                  <dd>{formatSigned(cellValue(selectedCellForActiveResponse, "model_residual_minutes"))}</dd>
                </div>
                <div>
                  <dt>Modes</dt>
                  <dd>{cellText(selectedCellForActiveResponse, "graph_modes")}</dd>
                </div>
                <div>
                  <dt>Corridors</dt>
                  <dd>{cellText(selectedCellForActiveResponse, "nearest_corridors")}</dd>
                </div>
                <div>
                  <dt>Interchanges</dt>
                  <dd>{cellValue(selectedCellForActiveResponse, "graph_interchanges")?.toFixed(0) ?? "-"}</dd>
                </div>
                <div>
                  <dt>Access / egress</dt>
                  <dd>
                    {formatSecondsAsMinutes(cellValue(selectedCellForActiveResponse, "graph_access_seconds"))} /{" "}
                    {formatSecondsAsMinutes(cellValue(selectedCellForActiveResponse, "graph_egress_seconds"))}
                  </dd>
                </div>
                <div>
                  <dt>Lat</dt>
                  <dd>{selectedCellForActiveResponse.lat.toFixed(5)}</dd>
                </div>
                <div>
                  <dt>Lng</dt>
                  <dd>{selectedCellForActiveResponse.lng.toFixed(5)}</dd>
                </div>
                <div>
                  <dt>Band</dt>
                  <dd>{selectedCellForActiveResponse.grid_band ?? "-"}</dd>
                </div>
                <div>
                  <dt>Model error</dt>
                  <dd>{formatSigned(cellValue(selectedCellForActiveResponse, "signed_error_minutes"))}</dd>
                </div>
              </dl>
              <div className="friend-times">
                {friends.map((friend, index) => (
                  <div className={included[index] ? "included" : ""} key={`${friend.name}-${index}-time`}>
                    <span>{friend.name || `Friend ${index + 1}`}</span>
                    <strong>
                      {formatMinutes(cellValue(selectedCellForActiveResponse, `friend_${index}_model_minutes`))}
                      {cellValue(selectedCellForActiveResponse, `friend_${index}_graph_minutes`) !== undefined
                        ? ` / graph ${formatMinutes(cellValue(selectedCellForActiveResponse, `friend_${index}_graph_minutes`))}`
                        : ""}
                      {cellValue(selectedCellForActiveResponse, `friend_${index}_reference_minutes`) !== undefined
                        ? ` / ${formatMinutes(cellValue(selectedCellForActiveResponse, `friend_${index}_reference_minutes`))}`
                        : ""}
                    </strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="selected-empty">Click a cell to inspect model, reference, and per-person values.</p>
          )}
        </section>
      </section>

      <section className="visual-panel" aria-label="Travel-time map">
        <MapView
          friends={friends}
          includedFriendIndexes={includedFriendIndexes}
          cells={activeResponse?.cells ?? []}
          selectedCell={selectedCellForActiveResponse}
          valueKey={valueKey}
          valueLabel={valueLabel}
          palette={mapPalette}
          customColorMap={customColorMap}
          colorScale={colorScale}
          surfaceOpacity={surfaceOpacity}
          mapStyle={mapStyle}
          isLoading={loading || referenceLoading}
          loadingLabel={referenceLoading ? "Fetching TravelTime" : "Updating map"}
          onSelectCell={setSelectedCell}
        />
      </section>
    </main>
  );
}
