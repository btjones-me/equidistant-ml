"""Export a lightweight, station-derived Tube line overlay for the browser map."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "london_tubes.csv"
OUTPUT = ROOT / "frontend" / "public" / "map" / "tube-lines.json"

LINE_COLOURS = {
    "Bakerloo": "#B36305",
    "Central": "#E32017",
    "Circle": "#FFD300",
    "District": "#00782A",
    "Hammersmith & City": "#F3A9BB",
    "Jubilee": "#7B868C",
    "Metropolitan": "#9B0056",
    "Northern": "#111111",
    "Piccadilly": "#003688",
    "Victoria": "#0098D4",
    "Waterloo & City": "#76D0BD",
}


def distance_squared(left: tuple[float, float], right: tuple[float, float]) -> float:
    mean_latitude = math.radians((left[0] + right[0]) / 2)
    delta_latitude = left[0] - right[0]
    delta_longitude = (left[1] - right[1]) * math.cos(mean_latitude)
    return delta_latitude**2 + delta_longitude**2


def minimum_spanning_segments(
    points: list[tuple[float, float]],
) -> list[list[list[float]]]:
    if len(points) < 2:
        return []
    connected = {0}
    remaining = set(range(1, len(points)))
    segments: list[list[list[float]]] = []
    while remaining:
        left_index, right_index = min(
            ((left, right) for left in connected for right in remaining),
            key=lambda pair: distance_squared(points[pair[0]], points[pair[1]]),
        )
        left = points[left_index]
        right = points[right_index]
        segments.append(
            [
                [round(left[0], 6), round(left[1], 6)],
                [round(right[0], 6), round(right[1], 6)],
            ]
        )
        connected.add(right_index)
        remaining.remove(right_index)
    return segments


def main() -> None:
    stations_by_line: dict[str, dict[str, tuple[float, float]]] = {
        line: {} for line in LINE_COLOURS
    }
    with SOURCE.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            try:
                point = (float(row["y"]), float(row["x"]))
            except (KeyError, TypeError, ValueError):
                continue
            for line in (part.strip() for part in row.get("LINES", "").split(",")):
                if line in stations_by_line:
                    stations_by_line[line].setdefault(row["NAME"], point)

    payload = {
        "version": 1,
        "source": "Station-derived schematic from the local London transport catalogue",
        "lines": [
            {
                "id": name.lower().replace(" & ", "-").replace(" ", "-"),
                "name": name,
                "color": colour,
                "segments": minimum_spanning_segments(
                    list(stations_by_line[name].values())
                ),
            }
            for name, colour in LINE_COLOURS.items()
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
