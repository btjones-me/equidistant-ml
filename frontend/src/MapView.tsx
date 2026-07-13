import { useEffect, useMemo, useRef, useState } from "react";
import L, { type LatLngBoundsExpression } from "leaflet";
import type { ColorMapStop, ColorScale, Friend, PaletteMode, SurfaceCell } from "./types";

type MapViewProps = {
  friends: Friend[];
  includedFriendIndexes: number[];
  cells: SurfaceCell[];
  selectedCell: SurfaceCell | null;
  valueKey: string;
  valueLabel: string;
  palette: PaletteMode;
  customColorMap: ColorMapStop[];
  colorScale: ColorScale;
  isLoading: boolean;
  loadingLabel: string;
  onSelectCell: (cell: SurfaceCell | null) => void;
  variant?: "developer" | "product";
  activeFriendIndex?: number | null;
  placingFriendIndex?: number | null;
  onMoveFriend?: (index: number, lat: number, lng: number) => void;
  onPlaceFriend?: (index: number, lat: number, lng: number) => void;
};

type RGB = [number, number, number];
type RenderPaletteMode = Exclude<PaletteMode, "custom"> | "residual";

type ColorStats = {
  min: number;
  max: number;
  p50: number;
  p90: number;
  centered: boolean;
  neutralLimit: number;
};

type PaletteConfig = {
  stops: { at: number; rgb: RGB }[];
};

type TubeLine = {
  id: string;
  name: string;
  color: string;
  segments: [[number, number], [number, number]][];
};

const palettes: Record<RenderPaletteMode, PaletteConfig> = {
  central: {
    stops: [
      { at: 0, rgb: [30, 64, 175] },
      { at: 0.07, rgb: [37, 99, 235] },
      { at: 0.14, rgb: [14, 165, 233] },
      { at: 0.24, rgb: [45, 212, 191] },
      { at: 0.36, rgb: [132, 204, 22] },
      { at: 0.52, rgb: [234, 179, 8] },
      { at: 0.7, rgb: [249, 115, 22] },
      { at: 0.86, rgb: [220, 38, 38] },
      { at: 1, rgb: [136, 19, 55] }
    ]
  },
  "green-red": {
    stops: [
      { at: 0, rgb: [21, 128, 61] },
      { at: 0.12, rgb: [34, 197, 94] },
      { at: 0.28, rgb: [132, 204, 22] },
      { at: 0.46, rgb: [250, 204, 21] },
      { at: 0.64, rgb: [249, 115, 22] },
      { at: 0.82, rgb: [220, 38, 38] },
      { at: 1, rgb: [127, 29, 29] }
    ]
  },
  viridis: {
    stops: [
      { at: 0, rgb: [68, 1, 84] },
      { at: 0.12, rgb: [70, 50, 127] },
      { at: 0.25, rgb: [54, 92, 141] },
      { at: 0.42, rgb: [39, 127, 142] },
      { at: 0.62, rgb: [31, 161, 135] },
      { at: 0.82, rgb: [122, 209, 81] },
      { at: 1, rgb: [253, 231, 37] }
    ]
  },
  inferno: {
    stops: [
      { at: 0, rgb: [0, 0, 4] },
      { at: 0.1, rgb: [40, 11, 84] },
      { at: 0.2, rgb: [101, 21, 110] },
      { at: 0.34, rgb: [159, 42, 99] },
      { at: 0.52, rgb: [212, 72, 66] },
      { at: 0.74, rgb: [245, 135, 48] },
      { at: 1, rgb: [252, 255, 164] }
    ]
  },
  error: {
    stops: [
      { at: 0, rgb: [22, 163, 74] },
      { at: 0.34, rgb: [250, 204, 21] },
      { at: 0.68, rgb: [249, 115, 22] },
      { at: 1, rgb: [185, 28, 28] }
    ]
  },
  residual: {
    stops: [
      { at: 0, rgb: [37, 99, 235] },
      { at: 0.22, rgb: [56, 189, 248] },
      { at: 0.5, rgb: [248, 250, 252] },
      { at: 0.78, rgb: [251, 113, 133] },
      { at: 1, rgb: [185, 28, 28] }
    ]
  }
};

const markerColours = ["#0ea5e9", "#f97316", "#22c55e", "#a855f7", "#e11d48", "#64748b"];
const zeroCenteredValueKeys = new Set(["model_residual_minutes", "signed_error_minutes"]);
const surfacePaneName = "surface-cells";
const tubePaneName = "tube-lines";
const placeLabelsPaneName = "place-labels";
const selectionPaneName = "surface-selection";
const friendMarkerPaneName = "friend-markers";

function valueForCell(cell: SurfaceCell, valueKey: string): number {
  const value = cell[valueKey];
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return cell.model_score_minutes ?? cell.score_minutes ?? cell.travel_time_minutes ?? 0;
}

function numericCellField(cell: SurfaceCell, key: string, fallback = 0): number {
  const value = cell[key];
  return typeof value === "number" ? value : fallback;
}

function quantile(values: number[], q: number): number {
  if (!values.length) {
    return 0;
  }
  const index = Math.min(values.length - 1, Math.max(0, Math.floor((values.length - 1) * q)));
  return values[index];
}

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function rgbString(rgb: RGB): string {
  return `rgb(${rgb.join(", ")})`;
}

function normaliseHexColor(value: string): string | null {
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
  return null;
}

function hexToRgb(value: string): RGB | null {
  const normalised = normaliseHexColor(value);
  if (!normalised) {
    return null;
  }
  return [
    Number.parseInt(normalised.slice(1, 3), 16),
    Number.parseInt(normalised.slice(3, 5), 16),
    Number.parseInt(normalised.slice(5, 7), 16)
  ];
}

function customPaletteConfig(customStops: ColorMapStop[]): PaletteConfig {
  const stops = customStops
    .map((stop) => ({
      at: clamp(Number(stop.position) / 100),
      rgb: hexToRgb(stop.color)
    }))
    .filter((stop): stop is { at: number; rgb: RGB } => stop.rgb !== null)
    .sort((left, right) => left.at - right.at);
  if (!stops.length) {
    return palettes.central;
  }
  const first = stops[0];
  const last = stops[stops.length - 1];
  return {
    stops: [
    first.at > 0 ? { at: 0, rgb: first.rgb } : null,
    ...stops,
    last.at < 1 ? { at: 1, rgb: last.rgb } : null
  ].filter((stop): stop is { at: number; rgb: RGB } => stop !== null)
  };
}

function interpolateStops(value: number, stops: PaletteConfig["stops"]): string {
  const t = clamp(value);
  const rightIndex = stops.findIndex((stop) => stop.at >= t);
  if (rightIndex <= 0) {
    return rgbString(stops[0].rgb);
  }
  const left = stops[rightIndex - 1];
  const right = stops[rightIndex];
  const span = Math.max(right.at - left.at, 0.001);
  const local = (t - left.at) / span;
  const rgb = left.rgb.map((channel, index) => Math.round(channel + (right.rgb[index] - channel) * local)) as RGB;
  return rgbString(rgb);
}

function scaleColor(value: number, domain: ColorStats, palette: PaletteConfig, contrast: number): string {
  if (domain.centered) {
    const limit = Math.max(Math.abs(domain.min), Math.abs(domain.max), 0.1);
    const neutralLimit = Math.min(Math.abs(domain.neutralLimit), limit * 0.95);
    const magnitude = Math.abs(value);
    const scaledMagnitude =
      magnitude <= neutralLimit ? 0 : (magnitude - neutralLimit) / Math.max(limit - neutralLimit, 0.1);
    const raw = Math.sign(value) * Math.max(0, Math.min(1, scaledMagnitude));
    const exponent = 1 / Math.max(0.2, contrast);
    const adjusted = Math.sign(raw) * Math.pow(Math.abs(raw), exponent);
    return interpolateStops(0.5 + adjusted * 0.5, palette.stops);
  }
  const max = Math.max(domain.max, domain.min + 0.1);
  const raw = clamp((value - domain.min) / (max - domain.min));
  const exponent = 1 / Math.max(0.2, contrast);
  return interpolateStops(Math.pow(raw, exponent), palette.stops);
}

function paletteGradient(palette: PaletteConfig): string {
  return `linear-gradient(90deg, ${palette.stops
    .map((stop) => `${rgbString(stop.rgb)} ${Math.round(stop.at * 100)}%`)
    .join(", ")})`;
}

function cellBounds(cell: SurfaceCell, sortedLats: number[], sortedLngs: number[]): LatLngBoundsExpression {
  if (typeof cell.south === "number" && typeof cell.north === "number" && typeof cell.west === "number" && typeof cell.east === "number") {
    return [
      [cell.south, cell.west],
      [cell.north, cell.east]
    ];
  }
  const latIndex = sortedLats.indexOf(cell.lat);
  const lngIndex = sortedLngs.indexOf(cell.lng);
  const latStep =
    sortedLats.length > 1
      ? Math.abs(sortedLats[Math.min(latIndex + 1, sortedLats.length - 1)] - sortedLats[Math.max(latIndex - 1, 0)]) /
        (latIndex === 0 || latIndex === sortedLats.length - 1 ? 1 : 2)
      : 0.01;
  const lngStep =
    sortedLngs.length > 1
      ? Math.abs(sortedLngs[Math.min(lngIndex + 1, sortedLngs.length - 1)] - sortedLngs[Math.max(lngIndex - 1, 0)]) /
        (lngIndex === 0 || lngIndex === sortedLngs.length - 1 ? 1 : 2)
      : 0.01;

  return [
    [cell.lat - latStep / 2, cell.lng - lngStep / 2],
    [cell.lat + latStep / 2, cell.lng + lngStep / 2]
  ];
}

function cellLatLngs(cell: SurfaceCell): [number, number][] | null {
  if (Array.isArray(cell.boundary) && cell.boundary.length >= 3) {
    return cell.boundary;
  }
  return null;
}

function boundsFromCells(cells: SurfaceCell[], sortedLats: number[], sortedLngs: number[]): L.LatLngBounds {
  const bounds = L.latLngBounds([]);
  cells.forEach((cell) => {
    const polygon = cellLatLngs(cell);
    if (polygon) {
      polygon.forEach((point) => bounds.extend(point));
      return;
    }
    const rect = cellBounds(cell, sortedLats, sortedLngs) as [number, number][];
    rect.forEach((point) => bounds.extend(point));
  });
  return bounds;
}

function edgeFadeOpacity(cell: SurfaceCell, bounds: L.LatLngBounds): number {
  const latitudeKm = Math.min(cell.lat - bounds.getSouth(), bounds.getNorth() - cell.lat) * 111.2;
  const longitudeScale = 111.2 * Math.cos((cell.lat * Math.PI) / 180);
  const longitudeKm = Math.min(cell.lng - bounds.getWest(), bounds.getEast() - cell.lng) * longitudeScale;
  const distanceKm = Math.max(0, Math.min(latitudeKm, longitudeKm));
  const progress = clamp(distanceKm / 4.5);
  const smooth = progress * progress * (3 - 2 * progress);
  return 0.08 + smooth * 0.92;
}

function tooltipForCell(
  cell: SurfaceCell,
  friends: Friend[],
  includedFriendIndexes: number[]
): HTMLElement {
  const container = document.createElement("div");
  container.className = "travel-tooltip";
  const place = document.createElement("strong");
  place.textContent =
    typeof cell.nearest_station_name === "string" && cell.nearest_station_name
      ? cell.nearest_station_name
      : "London destination";
  container.appendChild(place);

  const average = document.createElement("div");
  average.className = "travel-tooltip-average";
  const averageValue = numericCellField(cell, "mean_minutes", valueForCell(cell, "model_score_minutes"));
  average.textContent = `Average travel time ${averageValue.toFixed(1)} min`;
  container.appendChild(average);

  const journeys = document.createElement("div");
  journeys.className = "travel-tooltip-journeys";
  includedFriendIndexes.forEach((friendIndex) => {
    const row = document.createElement("span");
    const marker = document.createElement("i");
    marker.style.background = markerColours[friendIndex % markerColours.length];
    const name = document.createElement("b");
    name.textContent = friends[friendIndex]?.name || `Friend ${friendIndex + 1}`;
    const duration = document.createElement("em");
    duration.textContent = `${numericCellField(cell, `friend_${friendIndex}_model_minutes`).toFixed(1)} min`;
    row.append(marker, name, duration);
    journeys.appendChild(row);
  });
  container.appendChild(journeys);
  return container;
}

function gridSignatureForCells(cells: SurfaceCell[]): string {
  if (!cells.length) {
    return "empty";
  }
  const first = cells[0];
  const middle = cells[Math.floor(cells.length / 2)];
  const last = cells[cells.length - 1];
  return [
    cells.length,
    first.destination_id,
    middle.destination_id,
    last.destination_id,
    first.h3_resolution ?? "",
    last.h3_resolution ?? "",
    first.grid_band ?? "",
    last.grid_band ?? ""
  ].join("|");
}

export default function MapView({
  friends,
  includedFriendIndexes,
  cells,
  selectedCell,
  valueKey,
  valueLabel,
  palette,
  customColorMap,
  colorScale,
  isLoading,
  loadingLabel,
  onSelectCell,
  variant = "developer",
  activeFriendIndex = null,
  placingFriendIndex = null,
  onMoveFriend,
  onPlaceFriend
}: MapViewProps) {
  const mapRef = useRef<L.Map | null>(null);
  const mapNodeRef = useRef<HTMLDivElement | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const markerLayerRef = useRef<L.LayerGroup | null>(null);
  const tubeLayerRef = useRef<L.LayerGroup | null>(null);
  const selectionRef = useRef<L.Layer | null>(null);
  const lastFittedGridSignatureRef = useRef<string | null>(null);
  const [tubeLines, setTubeLines] = useState<TubeLine[]>([]);
  const [showTubeLines, setShowTubeLines] = useState(true);

  const gridSignature = useMemo(() => gridSignatureForCells(cells), [cells]);
  const isZeroCentered = zeroCenteredValueKeys.has(valueKey);
  const paletteConfig = useMemo(() => {
    if (isZeroCentered) {
      return palettes.residual;
    }
    if (palette === "custom") {
      return customPaletteConfig(customColorMap);
    }
    return palettes[palette as RenderPaletteMode];
  }, [customColorMap, isZeroCentered, palette]);

  const domain = useMemo(() => {
    const values = cells.map((cell) => valueForCell(cell, valueKey)).filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
    if (isZeroCentered) {
      const absValues = values.map((value) => Math.abs(value)).sort((a, b) => a - b);
      const neutralLimit = absValues.length ? quantile(absValues, colorScale.lowerPercentile / 100) : 0;
      const limit = absValues.length ? Math.max(quantile(absValues, colorScale.upperPercentile / 100), 0.1) : 1;
      return {
        min: -limit,
        max: limit,
        p50: 0,
        p90: limit,
        centered: true,
        neutralLimit
      };
    }
    const lower = quantile(values, colorScale.lowerPercentile / 100);
    const upper = quantile(values, colorScale.upperPercentile / 100);
    return {
      min: values.length ? lower : 0,
      max: values.length ? Math.max(upper, lower + 0.1) : 1,
      p50: quantile(values, 0.5),
      p90: quantile(values, 0.9),
      centered: false,
      neutralLimit: 0
    };
  }, [cells, colorScale.lowerPercentile, colorScale.upperPercentile, isZeroCentered, valueKey]);

  useEffect(() => {
    if (!mapNodeRef.current || mapRef.current) {
      return;
    }

    const map = L.map(mapNodeRef.current, {
      inertia: false,
      zoomControl: false,
      preferCanvas: true
    }).setView([51.5074, -0.1278], 11);

    L.control.zoom({ position: "bottomright" }).addTo(map);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);

    const surfacePane = map.createPane(surfacePaneName);
    surfacePane.style.zIndex = "410";
    const tubePane = map.createPane(tubePaneName);
    tubePane.style.zIndex = "480";
    tubePane.style.pointerEvents = "none";
    const placeLabelsPane = map.createPane(placeLabelsPaneName);
    placeLabelsPane.style.zIndex = "580";
    placeLabelsPane.style.pointerEvents = "none";
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png", {
      pane: placeLabelsPaneName,
      attribution: '&copy; <a href="https://carto.com/attributions">CARTO</a>'
    }).addTo(map);
    const selectionPane = map.createPane(selectionPaneName);
    selectionPane.style.zIndex = "620";
    selectionPane.style.pointerEvents = "none";
    const friendMarkerPane = map.createPane(friendMarkerPaneName);
    friendMarkerPane.style.zIndex = "650";
    friendMarkerPane.style.pointerEvents = variant === "product" ? "auto" : "none";

    const updateMapNodeView = () => {
      if (!mapNodeRef.current) {
        return;
      }
      const center = map.getCenter();
      mapNodeRef.current.dataset.mapCenter = `${center.lat.toFixed(6)},${center.lng.toFixed(6)}`;
      mapNodeRef.current.dataset.mapZoom = String(map.getZoom());
    };

    map.on("moveend zoomend", updateMapNodeView);
    updateMapNodeView();

    layerRef.current = L.layerGroup().addTo(map);
    tubeLayerRef.current = L.layerGroup().addTo(map);
    markerLayerRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    return () => {
      map.off("moveend zoomend", updateMapNodeView);
      map.remove();
      mapRef.current = null;
    };
  }, [variant]);

  useEffect(() => {
    let cancelled = false;
    void fetch("/map/tube-lines.json", { cache: "force-cache" })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Tube overlay unavailable");
        }
        return response.json() as Promise<{ lines?: TubeLine[] }>;
      })
      .then((payload) => {
        if (!cancelled) {
          setTubeLines(payload.lines ?? []);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const layer = tubeLayerRef.current;
    if (!layer) {
      return;
    }
    layer.clearLayers();
    if (!showTubeLines) {
      return;
    }
    tubeLines.forEach((line) => {
      line.segments.forEach((segment) => {
        L.polyline(segment, {
          pane: tubePaneName,
          color: "#ffffff",
          opacity: 0.8,
          weight: 5,
          interactive: false
        }).addTo(layer);
        L.polyline(segment, {
          pane: tubePaneName,
          color: line.color,
          opacity: 0.94,
          weight: 2.8,
          interactive: false
        }).addTo(layer);
      });
    });
  }, [showTubeLines, tubeLines]);

  useEffect(() => {
    const layer = layerRef.current;
    const map = mapRef.current;
    if (!layer || !map) {
      return;
    }

    layer.clearLayers();

    if (!cells.length) {
      return;
    }

    const shouldFitBounds = lastFittedGridSignatureRef.current !== gridSignature;
    const retainedCenter = shouldFitBounds ? null : map.getCenter();
    const retainedZoom = shouldFitBounds ? null : map.getZoom();
    const sortedLats = Array.from(new Set(cells.map((cell) => cell.lat))).sort((a, b) => a - b);
    const sortedLngs = Array.from(new Set(cells.map((cell) => cell.lng))).sort((a, b) => a - b);
    const surfaceBounds = boundsFromCells(cells, sortedLats, sortedLngs);
    const renderCells = cells
      .slice()
      .sort((left, right) => numericCellField(right, "grid_priority") - numericCellField(left, "grid_priority"));

    renderCells.forEach((cell) => {
      const value = valueForCell(cell, valueKey);
      const polygon = cellLatLngs(cell);
      const color = scaleColor(value, domain, paletteConfig, colorScale.contrast);
      const edgeOpacity = surfaceBounds.isValid() ? edgeFadeOpacity(cell, surfaceBounds) : 1;
      const cellLayer = polygon
        ? L.polygon(polygon, {
            pane: surfacePaneName,
            color,
            fillColor: color,
            fillOpacity: 0.68 * edgeOpacity,
            opacity: 0.22 * edgeOpacity,
            weight: 0.8
          })
        : L.rectangle(cellBounds(cell, sortedLats, sortedLngs), {
            pane: surfacePaneName,
            color,
            fillColor: color,
            fillOpacity: 0.66 * edgeOpacity,
            opacity: 0.24 * edgeOpacity,
            weight: 1
          });
      cellLayer.on("click", () => onSelectCell(cell));
      cellLayer.bindTooltip(tooltipForCell(cell, friends, includedFriendIndexes), {
        className: "travel-cell-tooltip",
        direction: "top",
        offset: [0, -6],
        sticky: true
      });
      cellLayer.addTo(layer);
    });

    if (shouldFitBounds) {
      const bounds = boundsFromCells(cells, sortedLats, sortedLngs);
      if (bounds.isValid()) {
        map.fitBounds(bounds.pad(0.08), { animate: false });
        lastFittedGridSignatureRef.current = gridSignature;
      }
    } else if (retainedCenter && retainedZoom !== null) {
      map.setView(retainedCenter, retainedZoom, { animate: false });
    }
  }, [
    cells,
    colorScale.contrast,
    domain,
    friends,
    gridSignature,
    includedFriendIndexes,
    onSelectCell,
    paletteConfig,
    valueKey,
    valueLabel
  ]);

  useEffect(() => {
    const markerLayer = markerLayerRef.current;
    if (!markerLayer) {
      return;
    }
    markerLayer.clearLayers();
    friends.forEach((friend, index) => {
      const isIncluded = includedFriendIndexes.includes(index);
      if (variant === "product") {
        const marker = L.marker([friend.lat, friend.lng], {
          pane: friendMarkerPaneName,
          draggable: Boolean(onMoveFriend),
          keyboard: true,
          title: friend.name || `Friend ${index + 1}`,
          icon: L.divIcon({
            className: "friend-marker-wrap",
            html: `<span class="friend-map-marker${activeFriendIndex === index ? " active" : ""}${isIncluded ? "" : " excluded"}" style="--marker-colour:${markerColours[index % markerColours.length]}"><b>${index + 1}</b></span>`,
            iconSize: [34, 40],
            iconAnchor: [17, 36],
            tooltipAnchor: [0, -32]
          })
        });
        const tooltip = document.createElement("span");
        tooltip.textContent = `${friend.name || `Friend ${index + 1}`}${isIncluded ? "" : " (excluded)"}`;
        marker.bindTooltip(tooltip);
        marker.on("dragend", () => {
          const position = marker.getLatLng();
          onMoveFriend?.(index, position.lat, position.lng);
        });
        marker.addTo(markerLayer);
        return;
      }
      L.circleMarker([friend.lat, friend.lng], {
          pane: friendMarkerPaneName,
          interactive: false,
          radius: isIncluded ? 8 : 5,
          color: isIncluded ? "#0f172a" : "#64748b",
          weight: isIncluded ? 3 : 2,
          fillColor: isIncluded ? markerColours[index % markerColours.length] : "#cbd5e1",
          fillOpacity: isIncluded ? 0.96 : 0.74,
          opacity: isIncluded ? 1 : 0.72
        })
        .addTo(markerLayer);
    });
  }, [activeFriendIndex, friends, includedFriendIndexes, onMoveFriend, variant]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || placingFriendIndex === null || !onPlaceFriend) {
      return;
    }
    const handleMapClick = (event: L.LeafletMouseEvent) => {
      onPlaceFriend(placingFriendIndex, event.latlng.lat, event.latlng.lng);
    };
    map.on("click", handleMapClick);
    return () => {
      map.off("click", handleMapClick);
    };
  }, [onPlaceFriend, placingFriendIndex]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }
    selectionRef.current?.remove();
    selectionRef.current = null;
    if (!selectedCell) {
      return;
    }
    const sortedLats = Array.from(new Set(cells.map((cell) => cell.lat))).sort((a, b) => a - b);
    const sortedLngs = Array.from(new Set(cells.map((cell) => cell.lng))).sort((a, b) => a - b);
    const polygon = cellLatLngs(selectedCell);
    selectionRef.current = polygon
      ? L.polygon(polygon, {
          pane: selectionPaneName,
          color: "#0f172a",
          fillOpacity: 0,
          opacity: 0.95,
          weight: 2
        }).addTo(map)
      : L.rectangle(cellBounds(selectedCell, sortedLats, sortedLngs), {
          pane: selectionPaneName,
          color: "#0f172a",
          fillOpacity: 0,
          opacity: 0.95,
          weight: 2
        }).addTo(map);
  }, [cells, selectedCell]);

  return (
    <div className={`map-shell ${variant === "product" ? "product-map" : "developer-map"}`}>
      <div className="map-toolbar">
        <div className="map-toolbar-summary">
          <span>{cells.length.toLocaleString()} cells</span>
          <span>{domain.centered ? `±${domain.max.toFixed(1)} min` : `${domain.min.toFixed(1)}-${domain.max.toFixed(1)} min`}</span>
          <span>{valueLabel}</span>
        </div>
        <div className="map-legend" aria-label="Colour legend">
          <div className="legend-ramp" style={{ background: paletteGradient(paletteConfig) }} />
          <div className="legend-labels">
            <span>{domain.min.toFixed(0)}</span>
            <span>{domain.centered ? "0" : ((domain.min + domain.max) / 2).toFixed(0)}</span>
            <span>{domain.centered ? `+${domain.max.toFixed(0)} min` : `${domain.max.toFixed(0)} min`}</span>
          </div>
        </div>
      </div>
      {tubeLines.length ? (
        <div className="tube-overlay-control">
          <label>
            <input type="checkbox" checked={showTubeLines} onChange={(event) => setShowTubeLines(event.target.checked)} />
            <span aria-hidden="true" />
            Tube lines
          </label>
          {showTubeLines ? (
            <div className="tube-line-key" aria-label="Tube line colours">
              {tubeLines.map((line) => (
                <span key={line.id}><i style={{ background: line.color }} />{line.name}</span>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
      {isLoading ? (
        <div className="map-loading-overlay" aria-live="polite">
          <div className="loading-spinner" aria-hidden="true" />
          <strong>{loadingLabel}</strong>
        </div>
      ) : null}
      <div className="map-node" ref={mapNodeRef} />
    </div>
  );
}
