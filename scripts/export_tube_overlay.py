"""Export TfL's geographic Tube line paths for the browser map."""

from __future__ import annotations

import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "frontend" / "public" / "map" / "tube-lines.json"
TFL_ROUTE_URL = "https://api.tfl.gov.uk/Line/{line_id}/Route/Sequence/outbound"
CORRIDOR_MERGE_DISTANCE_METRES = 1_000.0

Coordinate = list[float]
RoutePath = list[Coordinate]
PointKey = tuple[float, float]

LINE_COLOURS = {
    "bakerloo": ("Bakerloo", "#B36305"),
    "central": ("Central", "#E32017"),
    "circle": ("Circle", "#FFD300"),
    "district": ("District", "#00782A"),
    "elizabeth": ("Elizabeth", "#6950A1"),
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


def decode_line_paths(encoded_line: str) -> list[RoutePath]:
    """Convert TfL's stringified GeoJSON-style lng/lat paths to Leaflet lat/lng."""
    decoded = json.loads(encoded_line)
    paths: list[RoutePath] = []

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


def smooth_path(path: RoutePath, iterations: int = 2) -> RoutePath:
    """Round sparse route corners without moving branch endpoints."""
    smoothed = path
    for _ in range(iterations):
        if len(smoothed) < 3:
            break
        next_path = [smoothed[0]]
        for start, end in zip(smoothed, smoothed[1:]):
            next_path.extend(
                [
                    [
                        round(start[0] * 0.75 + end[0] * 0.25, 6),
                        round(start[1] * 0.75 + end[1] * 0.25, 6),
                    ],
                    [
                        round(start[0] * 0.25 + end[0] * 0.75, 6),
                        round(start[1] * 0.25 + end[1] * 0.75, 6),
                    ],
                ]
            )
        next_path.append(smoothed[-1])
        smoothed = next_path
    return smoothed


def _point_key(point: Coordinate) -> PointKey:
    return (round(point[0], 6), round(point[1], 6))


def _path_length_metres(path: RoutePath) -> float:
    return sum(
        _coordinate_distance_metres(start, end) for start, end in zip(path, path[1:])
    )


def _coordinate_distance_metres(start: Coordinate, end: Coordinate) -> float:
    mean_latitude = math.radians((start[0] + end[0]) / 2)
    north = (end[0] - start[0]) * 111_320.0
    east = (end[1] - start[1]) * 111_320.0 * math.cos(mean_latitude)
    return math.hypot(east, north)


def _project_coordinate(
    point: Coordinate, reference_latitude: float
) -> tuple[float, float]:
    return (
        point[1] * 111_320.0 * math.cos(math.radians(reference_latitude)),
        point[0] * 111_320.0,
    )


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * segment_x + (point[1] - start[1]) * segment_y)
            / length_squared,
        ),
    )
    nearest = (start[0] + fraction * segment_x, start[1] + fraction * segment_y)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def _directed_path_distance_metres(path: RoutePath, target: RoutePath) -> float:
    reference_latitude = sum(point[0] for point in path + target) / (
        len(path) + len(target)
    )
    projected_path = [_project_coordinate(point, reference_latitude) for point in path]
    projected_target = [
        _project_coordinate(point, reference_latitude) for point in target
    ]
    target_segments = list(zip(projected_target, projected_target[1:]))
    return max(
        min(
            _point_to_segment_distance(point, start, end)
            for start, end in target_segments
        )
        for point in projected_path
    )


def _path_distance_metres(first: RoutePath, second: RoutePath) -> float:
    return max(
        _directed_path_distance_metres(first, second),
        _directed_path_distance_metres(second, first),
    )


def _split_at_shared_points(paths: list[RoutePath]) -> list[RoutePath]:
    appearances: Counter[PointKey] = Counter()
    for path in paths:
        appearances.update(set(_point_key(point) for point in path))

    shared_points = {key for key, count in appearances.items() if count > 1}
    sections: list[RoutePath] = []
    for path in paths:
        anchor_indexes = [
            index
            for index, point in enumerate(path)
            if index in {0, len(path) - 1} or _point_key(point) in shared_points
        ]
        for start, end in zip(anchor_indexes, anchor_indexes[1:]):
            if end > start:
                sections.append(path[start : end + 1])
    return sections


def _canonical_sections(
    paths: list[RoutePath], corridor_distance_metres: float
) -> list[RoutePath]:
    sections_by_endpoints: dict[tuple[PointKey, PointKey], list[RoutePath]] = (
        defaultdict(list)
    )
    for section in _split_at_shared_points(paths):
        endpoints = tuple(sorted((_point_key(section[0]), _point_key(section[-1]))))
        sections_by_endpoints[endpoints].append(section)

    selected: list[RoutePath] = []
    for candidates in sections_by_endpoints.values():
        unique_candidates: dict[tuple[PointKey, ...], RoutePath] = {}
        for candidate in candidates:
            keys = tuple(_point_key(point) for point in candidate)
            canonical_key = min(keys, tuple(reversed(keys)))
            unique_candidates[canonical_key] = candidate

        preferred = sorted(
            unique_candidates.values(),
            key=lambda path: (len(path), _path_length_metres(path)),
            reverse=True,
        )
        representatives: list[RoutePath] = []
        for candidate in preferred:
            if any(
                _path_distance_metres(candidate, representative)
                <= corridor_distance_metres
                for representative in representatives
            ):
                continue
            representatives.append(candidate)
        selected.extend(representatives)
    return selected


def _shortest_alternative_path(
    sections: list[RoutePath], skipped_edge: int
) -> RoutePath | None:
    start_key = _point_key(sections[skipped_edge][0])
    end_key = _point_key(sections[skipped_edge][-1])
    adjacency: dict[PointKey, list[tuple[int, PointKey]]] = defaultdict(list)
    for edge_index, section in enumerate(sections):
        if edge_index == skipped_edge:
            continue
        first = _point_key(section[0])
        last = _point_key(section[-1])
        adjacency[first].append((edge_index, last))
        adjacency[last].append((edge_index, first))

    distances = {start_key: 0.0}
    previous: dict[PointKey, tuple[PointKey, int]] = {}
    queue: list[tuple[float, PointKey]] = [(0.0, start_key)]
    while queue:
        distance, node = heapq.heappop(queue)
        if node == end_key:
            break
        if distance > distances.get(node, math.inf):
            continue
        for edge_index, neighbour in adjacency[node]:
            candidate = distance + _path_length_metres(sections[edge_index])
            if candidate >= distances.get(neighbour, math.inf):
                continue
            distances[neighbour] = candidate
            previous[neighbour] = (node, edge_index)
            heapq.heappush(queue, (candidate, neighbour))

    if end_key not in previous:
        return None

    route: list[tuple[PointKey, int, PointKey]] = []
    node = end_key
    while node != start_key:
        parent, edge_index = previous[node]
        route.append((parent, edge_index, node))
        node = parent
    route.reverse()

    path: RoutePath = []
    for parent, edge_index, _ in route:
        section = sections[edge_index]
        if _point_key(section[0]) != parent:
            section = list(reversed(section))
        path.extend(section if not path else section[1:])
    return path


def _remove_nearby_shortcuts(
    sections: list[RoutePath], corridor_distance_metres: float
) -> list[RoutePath]:
    retained = list(sections)
    candidate_indexes = sorted(
        range(len(sections)),
        key=lambda index: _path_length_metres(sections[index]),
        reverse=True,
    )
    for original_index in candidate_indexes:
        candidate = sections[original_index]
        try:
            current_index = retained.index(candidate)
        except ValueError:
            continue
        alternative = _shortest_alternative_path(retained, current_index)
        if alternative is None or len(alternative) <= len(candidate):
            continue
        candidate_length = _path_length_metres(candidate)
        if _path_length_metres(alternative) > candidate_length * 1.6:
            continue
        if _path_distance_metres(candidate, alternative) <= corridor_distance_metres:
            retained.pop(current_index)
    return retained


def _merge_connected_sections(sections: list[RoutePath]) -> list[RoutePath]:
    adjacency: dict[PointKey, list[int]] = defaultdict(list)
    for edge_index, section in enumerate(sections):
        adjacency[_point_key(section[0])].append(edge_index)
        adjacency[_point_key(section[-1])].append(edge_index)

    used: set[int] = set()

    def walk(start_key: PointKey, first_edge: int) -> RoutePath:
        path: RoutePath = []
        current_key = start_key
        edge_index = first_edge
        while edge_index not in used:
            used.add(edge_index)
            section = sections[edge_index]
            if _point_key(section[0]) != current_key:
                section = list(reversed(section))
            path.extend(section if not path else section[1:])
            current_key = _point_key(section[-1])
            available = [index for index in adjacency[current_key] if index not in used]
            if len(adjacency[current_key]) != 2 or not available:
                break
            next_edge = sections[available[0]]
            next_endpoint = (
                _point_key(next_edge[-1])
                if _point_key(next_edge[0]) == current_key
                else _point_key(next_edge[0])
            )
            if next_endpoint == start_key:
                break
            edge_index = available[0]
        return path

    merged: list[RoutePath] = []
    for node, edge_indexes in adjacency.items():
        if len(edge_indexes) == 2:
            continue
        for edge_index in edge_indexes:
            if edge_index not in used:
                merged.append(walk(node, edge_index))

    for edge_index, section in enumerate(sections):
        if edge_index not in used:
            merged.append(walk(_point_key(section[0]), edge_index))
    return merged


def canonicalize_line_paths(
    paths: list[RoutePath],
    corridor_distance_metres: float = CORRIDOR_MERGE_DISTANCE_METRES,
    smoothing_iterations: int = 2,
) -> list[RoutePath]:
    """Collapse route variants into a single readable same-line network."""
    sections = _canonical_sections(paths, corridor_distance_metres)
    sections = _remove_nearby_shortcuts(sections, corridor_distance_metres)
    return [
        smooth_path(path, iterations=smoothing_iterations)
        for path in _merge_connected_sections(sections)
        if len(path) >= 2
    ]


def fetch_line_paths(line_id: str) -> list[RoutePath]:
    request = Request(
        TFL_ROUTE_URL.format(line_id=line_id),
        headers={"User-Agent": "equidistant-london/1.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    paths = [
        path
        for encoded_line in payload.get("lineStrings", [])
        for path in decode_line_paths(encoded_line)
    ]
    return canonicalize_line_paths(paths)


def main() -> None:
    payload = {
        "version": 4,
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
