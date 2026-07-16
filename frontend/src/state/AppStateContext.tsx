import {
  createContext,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";
import type {
  ColorScale,
  CombineMode,
  DetailMode,
  EditableColorMapStop,
  FocusMode,
  Friend,
  MapStyle,
  PaletteMode
} from "../types";

const STORAGE_KEY = "equidistant:workspace:v2";
export const DEFAULT_SURFACE_OPACITY = 0.52;

export const sampleFriends: Friend[] = [
  { id: "alex", name: "Alex", lat: 51.551808, lng: -0.195603, locationLabel: "West Hampstead" },
  { id: "sam", name: "Sam", lat: 51.5364, lng: -0.075, locationLabel: "De Beauvoir Town" },
  { id: "priya", name: "Priya", lat: 51.4701, lng: -0.2106, locationLabel: "Putney" },
  { id: "maya", name: "Maya", lat: 51.548, lng: -0.191, locationLabel: "West Hampstead" },
  { id: "tom", name: "Tom", lat: 51.5033, lng: -0.1195, locationLabel: "Waterloo" },
  { id: "nina", name: "Nina", lat: 51.4628, lng: -0.114, locationLabel: "Brixton" }
];

export const singleParticipantColorScale: ColorScale = {
  lowerPercentile: 1,
  upperPercentile: 100,
  contrast: 0.75
};

export const groupColorScale: ColorScale = {
  lowerPercentile: 1,
  upperPercentile: 100,
  contrast: 1.1
};

export function recommendedColorScale(activeFriendCount: number): ColorScale {
  return activeFriendCount <= 1 ? singleParticipantColorScale : groupColorScale;
}

type PersistedAppState = {
  friends: Friend[];
  included: boolean[];
  combine: CombineMode;
  focus: FocusMode;
  detail: DetailMode;
  mapStyle: MapStyle;
  palette: Exclude<PaletteMode, "error">;
  customColorStops: EditableColorMapStop[];
  colorScale: ColorScale;
  surfaceOpacity: number;
  suggestionMinDistanceKm: number;
};

type AppStateContextValue = PersistedAppState & {
  setFriends: Dispatch<SetStateAction<Friend[]>>;
  setIncluded: Dispatch<SetStateAction<boolean[]>>;
  setCombine: Dispatch<SetStateAction<CombineMode>>;
  setFocus: Dispatch<SetStateAction<FocusMode>>;
  setDetail: Dispatch<SetStateAction<DetailMode>>;
  setMapStyle: Dispatch<SetStateAction<MapStyle>>;
  setPalette: Dispatch<SetStateAction<Exclude<PaletteMode, "error">>>;
  setCustomColorStops: Dispatch<SetStateAction<EditableColorMapStop[]>>;
  setColorScale: Dispatch<SetStateAction<ColorScale>>;
  setSurfaceOpacity: Dispatch<SetStateAction<number>>;
  setSuggestionMinDistanceKm: Dispatch<SetStateAction<number>>;
  changeFriendCount: (count: number) => void;
  updateFriend: (index: number, patch: Partial<Friend>) => void;
  toggleFriend: (index: number) => void;
  resetWorkspace: () => void;
};

function defaultState(): PersistedAppState {
  return {
    friends: sampleFriends.slice(0, 3),
    included: [true, true, true],
    combine: "balanced",
    focus: "inner",
    detail: "fine",
    mapStyle: "positron",
    palette: "central",
    customColorStops: [
      { id: "stop-green", position: 0, color: "#15803d" },
      { id: "stop-yellow", position: 46, color: "#facc15" },
      { id: "stop-orange", position: 70, color: "#f97316" },
      { id: "stop-red", position: 100, color: "#7f1d1d" }
    ],
    colorScale: recommendedColorScale(3),
    surfaceOpacity: DEFAULT_SURFACE_OPACITY,
    suggestionMinDistanceKm: 3
  };
}

function isFiniteFriend(friend: Friend): boolean {
  return Boolean(friend?.id && typeof friend.name === "string" && Number.isFinite(friend.lat) && Number.isFinite(friend.lng));
}

function loadState(): PersistedAppState {
  const fallback = defaultState();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return fallback;
    }
    const parsed = JSON.parse(raw) as Partial<PersistedAppState>;
    const friends = Array.isArray(parsed.friends) ? parsed.friends.filter(isFiniteFriend).slice(0, 6) : [];
    if (friends.length < 1) {
      return fallback;
    }
    const included = friends.map((_, index) => parsed.included?.[index] !== false);
    if (!included.some(Boolean)) {
      included[0] = true;
    }
    const customColorStops = Array.isArray(parsed.customColorStops)
      ? parsed.customColorStops.filter(
          (stop) =>
            typeof stop?.id === "string" &&
            Number.isFinite(stop.position) &&
            typeof stop.color === "string"
        )
      : [];
    return {
      friends,
      included,
      combine: ["balanced", "max", "mean", "fairness"].includes(parsed.combine ?? "")
        ? (parsed.combine as CombineMode)
        : fallback.combine,
      focus: "inner",
      detail: ["fine", "fast"].includes(parsed.detail ?? "")
        ? (parsed.detail as DetailMode)
        : fallback.detail,
      mapStyle: ["positron", "voyager", "dark-matter"].includes(parsed.mapStyle ?? "")
        ? (parsed.mapStyle as MapStyle)
        : fallback.mapStyle,
      palette: ["central", "green-red", "viridis", "inferno", "custom"].includes(parsed.palette ?? "")
        ? (parsed.palette as Exclude<PaletteMode, "error">)
        : fallback.palette,
      customColorStops: customColorStops.length >= 2 ? customColorStops : fallback.customColorStops,
      colorScale:
        parsed.colorScale &&
        Number.isFinite(parsed.colorScale.lowerPercentile) &&
        Number.isFinite(parsed.colorScale.upperPercentile) &&
        Number.isFinite(parsed.colorScale.contrast)
          ? parsed.colorScale
          : fallback.colorScale,
      surfaceOpacity:
        Number.isFinite(parsed.surfaceOpacity) &&
        Number(parsed.surfaceOpacity) >= 0.15 &&
        Number(parsed.surfaceOpacity) <= 0.9
          ? Number(parsed.surfaceOpacity)
          : fallback.surfaceOpacity,
      suggestionMinDistanceKm:
        Number.isFinite(parsed.suggestionMinDistanceKm) &&
        Number(parsed.suggestionMinDistanceKm) >= 0.5 &&
        Number(parsed.suggestionMinDistanceKm) <= 20
          ? Number(parsed.suggestionMinDistanceKm)
          : fallback.suggestionMinDistanceKm
    };
  } catch {
    return fallback;
  }
}

const AppStateContext = createContext<AppStateContextValue | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(loadState, []);
  const [friends, setFriends] = useState(initial.friends);
  const [included, setIncluded] = useState(initial.included);
  const [combine, setCombine] = useState(initial.combine);
  const [focus, setFocus] = useState(initial.focus);
  const [detail, setDetail] = useState(initial.detail);
  const [mapStyle, setMapStyle] = useState(initial.mapStyle);
  const [palette, setPalette] = useState(initial.palette);
  const [customColorStops, setCustomColorStops] = useState(initial.customColorStops);
  const [colorScale, setColorScale] = useState(initial.colorScale);
  const [surfaceOpacity, setSurfaceOpacity] = useState(initial.surfaceOpacity);
  const [suggestionMinDistanceKm, setSuggestionMinDistanceKm] = useState(initial.suggestionMinDistanceKm);

  useEffect(() => {
    const state: PersistedAppState = {
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
      suggestionMinDistanceKm
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [colorScale, combine, customColorStops, detail, focus, friends, included, mapStyle, palette, suggestionMinDistanceKm, surfaceOpacity]);

  function changeFriendCount(count: number) {
    const nextCount = Math.max(1, Math.min(6, Math.round(count)));
    setFriends((current) => {
      if (nextCount <= current.length) {
        return current.slice(0, nextCount);
      }
      return [...current, ...sampleFriends.slice(current.length, nextCount)];
    });
    setIncluded((current) => {
      if (nextCount <= current.length) {
        const next = current.slice(0, nextCount);
        return next.some(Boolean) ? next : next.map((_, index) => index === 0);
      }
      return [...current, ...Array.from({ length: nextCount - current.length }, () => true)];
    });
  }

  function updateFriend(index: number, patch: Partial<Friend>) {
    setFriends((current) => current.map((friend, friendIndex) => (friendIndex === index ? { ...friend, ...patch } : friend)));
  }

  function toggleFriend(index: number) {
    setIncluded((current) => {
      const next = current.slice();
      if (next[index] && next.filter(Boolean).length === 1) {
        return next;
      }
      next[index] = !next[index];
      return next;
    });
  }

  function resetWorkspace() {
    const next = defaultState();
    setFriends(next.friends);
    setIncluded(next.included);
    setCombine(next.combine);
    setFocus(next.focus);
    setDetail(next.detail);
    setMapStyle(next.mapStyle);
    setPalette(next.palette);
    setCustomColorStops(next.customColorStops);
    setColorScale(next.colorScale);
    setSurfaceOpacity(next.surfaceOpacity);
    setSuggestionMinDistanceKm(next.suggestionMinDistanceKm);
  }

  const value: AppStateContextValue = {
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
    setFriends,
    setIncluded,
    setCombine,
    setFocus,
    setDetail,
    setMapStyle,
    setPalette,
    setCustomColorStops,
    setColorScale,
    setSurfaceOpacity,
    setSuggestionMinDistanceKm,
    changeFriendCount,
    updateFriend,
    toggleFriend,
    resetWorkspace
  };

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState(): AppStateContextValue {
  const value = useContext(AppStateContext);
  if (!value) {
    throw new Error("useAppState must be used inside AppStateProvider");
  }
  return value;
}
