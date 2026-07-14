"""Export TfL's geographic Tube line paths for the browser map."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "frontend" / "public" / "map" / "tube-lines.json"
TFL_ROUTE_URL = "https://api.tfl.gov.uk/Line/{line_id}/Route/Sequence/outbound"

LINE_COLOURS = {
    "bakerloo": ("Bakerloo", "#B36305"),
    "central": ("Central", "#E32017"),
    "circle": ("Circle", "#FFD300"),
    "district": ("District", "#00782A"),
    "hammersmith-city": ("Hammersmith & City", "#F3A9BB"),
    "jubilee": ("Jubilee", "#7B868C"),
    "metropolitan": ("Metropolitan", "#9B0056"),
    "northern": ("Northern", "#111111"),
    "piccadilly": ("Piccadilly", "#003688"),
    "victoria": ("Victoria", "#0098D4"),
    "waterloo-city": ("Waterloo & City", "#76D0BD"),
}


def _is_coordinate(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and all(isinstance(part, (int, float)) for part in value[:2])
    )


def decode_line_paths(encoded_line: str) -> list[list[list[float]]]:
    """Convert TfL's stringified GeoJSON-style lng/lat paths to Leaflet lat/lng."""
    decoded = json.loads(encoded_line)
    paths: list[list[list[float]]] = []

    def collect(value: Any) -> None:
        if not isinstance(value, list) or not value:
            return
        if all(_is_coordinate(point) for point in value):
            paths.append(
                [
                    [round(float(point[1]), 6), round(float(point[0]), 6)]
                    for point in value
                ]
            )
            return
        for child in value:
            collect(child)

    collect(decoded)
    return [path for path in paths if len(path) >= 2]


def fetch_line_paths(line_id: str) -> list[list[list[float]]]:
    request = Request(
        TFL_ROUTE_URL.format(line_id=line_id),
        headers={"User-Agent": "equidistant-london/1.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    return [
        path
        for encoded_line in payload.get("lineStrings", [])
        for path in decode_line_paths(encoded_line)
    ]


def main() -> None:
    payload = {
        "version": 2,
        "source": "TfL Unified API geographic line strings",
        "source_url": "https://tfl.gov.uk/info-for/open-data-users/unified-api",
        "lines": [
            {
                "id": line_id,
                "name": name,
                "color": colour,
                "paths": fetch_line_paths(line_id),
            }
            for line_id, (name, colour) in LINE_COLOURS.items()
        ],
    }
    missing = [line["name"] for line in payload["lines"] if not line["paths"]]
    if missing:
        raise RuntimeError(f"TfL returned no route geometry for: {', '.join(missing)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
