import { useEffect, useMemo, useRef } from "react";
import L, { type LatLngBoundsExpression, type Rectangle } from "leaflet";
import type { Friend, SurfaceCell } from "./types";

type MapViewProps = {
  friends: Friend[];
  cells: SurfaceCell[];
  selectedCell: SurfaceCell | null;
  onSelectCell: (cell: SurfaceCell | null) => void;
};

function valueForCell(cell: SurfaceCell): number {
  return cell.score_minutes ?? cell.travel_time_minutes ?? 0;
}

function scaleColor(value: number, min: number, max: number): string {
  const span = Math.max(max - min, 1);
  const t = Math.min(1, Math.max(0, (value - min) / span));
  if (t < 0.5) {
    const local = t / 0.5;
    return `rgb(${Math.round(41 + local * 206)}, ${Math.round(142 + local * 49)}, ${Math.round(108 - local * 73)})`;
  }
  const local = (t - 0.5) / 0.5;
  return `rgb(${Math.round(247 - local * 192)}, ${Math.round(191 - local * 126)}, ${Math.round(35 - local * 46)})`;
}

function cellBounds(cell: SurfaceCell, sortedLats: number[], sortedLngs: number[]): LatLngBoundsExpression {
  const latIndex = sortedLats.indexOf(cell.lat);
  const lngIndex = sortedLngs.indexOf(cell.lng);
  const latStep =
    sortedLats.length > 1
      ? Math.abs(sortedLats[Math.min(latIndex + 1, sortedLats.length - 1)] - sortedLats[Math.max(latIndex - 1, 0)]) / (latIndex === 0 || latIndex === sortedLats.length - 1 ? 1 : 2)
      : 0.01;
  const lngStep =
    sortedLngs.length > 1
      ? Math.abs(sortedLngs[Math.min(lngIndex + 1, sortedLngs.length - 1)] - sortedLngs[Math.max(lngIndex - 1, 0)]) / (lngIndex === 0 || lngIndex === sortedLngs.length - 1 ? 1 : 2)
      : 0.01;

  return [
    [cell.lat - latStep / 2, cell.lng - lngStep / 2],
    [cell.lat + latStep / 2, cell.lng + lngStep / 2]
  ];
}

export default function MapView({ friends, cells, selectedCell, onSelectCell }: MapViewProps) {
  const mapRef = useRef<L.Map | null>(null);
  const mapNodeRef = useRef<HTMLDivElement | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const markerLayerRef = useRef<L.LayerGroup | null>(null);
  const selectionRef = useRef<Rectangle | null>(null);

  const stats = useMemo(() => {
    const values = cells.map(valueForCell).filter((value) => Number.isFinite(value));
    return {
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 1
    };
  }, [cells]);

  useEffect(() => {
    if (!mapNodeRef.current || mapRef.current) {
      return;
    }

    const map = L.map(mapNodeRef.current, {
      zoomControl: false,
      preferCanvas: true
    }).setView([51.5074, -0.1278], 11);

    L.control.zoom({ position: "bottomright" }).addTo(map);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);

    layerRef.current = L.layerGroup().addTo(map);
    markerLayerRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const layer = layerRef.current;
    const map = mapRef.current;
    if (!layer || !map) {
      return;
    }

    layer.clearLayers();
    selectionRef.current?.remove();
    selectionRef.current = null;

    if (!cells.length) {
      return;
    }

    const sortedLats = Array.from(new Set(cells.map((cell) => cell.lat))).sort((a, b) => a - b);
    const sortedLngs = Array.from(new Set(cells.map((cell) => cell.lng))).sort((a, b) => a - b);
    const allBounds: LatLngBoundsExpression[] = [];

    cells.forEach((cell) => {
      const value = valueForCell(cell);
      const bounds = cellBounds(cell, sortedLats, sortedLngs);
      allBounds.push(bounds);
      const rectangle = L.rectangle(bounds, {
        color: scaleColor(value, stats.min, stats.max),
        fillColor: scaleColor(value, stats.min, stats.max),
        fillOpacity: 0.55,
        opacity: 0.18,
        weight: 1
      });
      rectangle.on("click", () => onSelectCell(cell));
      rectangle.bindTooltip(`${value.toFixed(1)} min`, { sticky: true });
      rectangle.addTo(layer);
    });

    const bounds = L.latLngBounds(allBounds.flat() as [number, number][]);
    if (bounds.isValid()) {
      map.fitBounds(bounds.pad(0.08), { animate: false });
    }
  }, [cells, onSelectCell, stats.max, stats.min]);

  useEffect(() => {
    const markerLayer = markerLayerRef.current;
    if (!markerLayer) {
      return;
    }
    markerLayer.clearLayers();
    friends.forEach((friend, index) => {
      L.circleMarker([friend.lat, friend.lng], {
        radius: 7,
        color: "#0f172a",
        weight: 2,
        fillColor: index % 2 === 0 ? "#38bdf8" : "#f97316",
        fillOpacity: 0.95
      })
        .bindTooltip(friend.name || `Friend ${index + 1}`)
        .addTo(markerLayer);
    });
  }, [friends]);

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
    selectionRef.current = L.rectangle(cellBounds(selectedCell, sortedLats, sortedLngs), {
      color: "#0f172a",
      fillOpacity: 0,
      opacity: 0.9,
      weight: 2
    }).addTo(map);
  }, [cells, selectedCell]);

  return (
    <div className="map-shell">
      <div className="map-toolbar">
        <span>{cells.length.toLocaleString()} cells</span>
        <span>{stats.min.toFixed(1)}-{stats.max.toFixed(1)} min</span>
      </div>
      <div ref={mapNodeRef} className="map-node" />
    </div>
  );
}
