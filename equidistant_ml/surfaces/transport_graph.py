"""Transport graph ingestion and feature generation.

The graph is intentionally approximate. TravelTime remains the label source; these
features give the offline model a topology prior so it can learn rail and transit
corridor effects that pure latitude/longitude features cannot express.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from pyprojroot import here

from equidistant_ml.surfaces.geo import haversine_m, safe_feature_name

GRAPH_NUMERIC_COLUMNS = [
    "graph_has_path",
    "graph_total_seconds",
    "graph_path_seconds",
    "graph_access_seconds",
    "graph_egress_seconds",
    "graph_interchanges",
    "graph_mode_count",
    "graph_rail_advantage_seconds",
    "graph_uses_bus",
    "graph_uses_tube",
    "graph_uses_rail",
    "graph_uses_overground",
    "graph_uses_elizabeth_line",
    "graph_uses_thameslink",
    "graph_uses_national_rail",
    "origin_nearest_heavy_rail_distance_m",
    "destination_nearest_heavy_rail_distance_m",
    "origin_bus_stop_density",
    "destination_bus_stop_density",
    "origin_bus_route_count",
    "destination_bus_route_count",
    "same_graph_corridor",
]

GRAPH_TEXT_COLUMNS = ["graph_modes", "nearest_corridors"]

RAIL_MODES = {"tube", "overground", "elizabeth-line", "dlr", "tram", "national_rail"}
HEAVY_RAIL_MODES = {"overground", "elizabeth-line", "national_rail"}

DEFAULT_SPEED_KPH = {
    "tube": 28.0,
    "overground": 33.0,
    "elizabeth-line": 42.0,
    "dlr": 26.0,
    "tram": 23.0,
    "national_rail": 45.0,
    "bus": 14.0,
    "transfer": 4.8,
}

CURATED_CORRIDORS: list[dict[str, Any]] = [
    {
        "line": "Thameslink",
        "mode": "national_rail",
        "operator": "National Rail",
        "stops": [
            ("West Hampstead Thameslink", 51.5486, -0.1910),
            ("Kentish Town", 51.5500, -0.1407),
            ("London St Pancras International", 51.5319, -0.1261),
            ("Farringdon", 51.5200, -0.1049),
            ("City Thameslink", 51.5139, -0.1036),
            ("Blackfriars", 51.5116, -0.1030),
            ("London Bridge", 51.5050, -0.0860),
            ("Elephant & Castle", 51.4943, -0.1003),
        ],
    },
    {
        "line": "London Overground North London",
        "mode": "overground",
        "operator": "TfL",
        "stops": [
            ("West Hampstead", 51.5475, -0.1912),
            ("Finchley Road & Frognal", 51.5503, -0.1831),
            ("Hampstead Heath", 51.5552, -0.1657),
            ("Gospel Oak", 51.5554, -0.1513),
            ("Kentish Town West", 51.5465, -0.1469),
            ("Camden Road", 51.5419, -0.1392),
            ("Highbury & Islington", 51.5464, -0.1039),
            ("Canonbury", 51.5482, -0.0923),
            ("Dalston Junction", 51.5461, -0.0751),
            ("Stratford", 51.5413, -0.0033),
        ],
    },
    {
        "line": "Elizabeth line",
        "mode": "elizabeth-line",
        "operator": "TfL",
        "stops": [
            ("Paddington", 51.5160, -0.1762),
            ("Bond Street", 51.5142, -0.1494),
            ("Tottenham Court Road", 51.5162, -0.1309),
            ("Farringdon", 51.5200, -0.1049),
            ("Liverpool Street", 51.5178, -0.0817),
            ("Whitechapel", 51.5195, -0.0598),
            ("Canary Wharf", 51.5054, -0.0235),
            ("Custom House", 51.5097, 0.0265),
            ("Woolwich", 51.4916, 0.0716),
        ],
    },
    {
        "line": "National Rail central terminals",
        "mode": "national_rail",
        "operator": "National Rail",
        "stops": [
            ("Clapham Junction", 51.4642, -0.1703),
            ("London Victoria", 51.4965, -0.1447),
            ("London Waterloo", 51.5032, -0.1136),
            ("London Bridge", 51.5050, -0.0860),
            ("Liverpool Street", 51.5178, -0.0817),
        ],
    },
]


@dataclass(frozen=True)
class TransportGraph:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    adjacency: dict[str, list[dict[str, Any]]]


def empty_graph_features(row_count: int) -> pd.DataFrame:
    data: dict[str, Any] = {
        column: np.zeros(row_count) for column in GRAPH_NUMERIC_COLUMNS
    }
    data["graph_total_seconds"] = np.nan
    data["graph_path_seconds"] = np.nan
    data["origin_nearest_heavy_rail_distance_m"] = np.nan
    data["destination_nearest_heavy_rail_distance_m"] = np.nan
    for column in GRAPH_TEXT_COLUMNS:
        data[column] = [""] * row_count
    return pd.DataFrame(data)


def _split_lines(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _corridor_flags(mode: str, line: str, name: str = "") -> set[str]:
    text = f"{mode} {line} {name}".lower()
    flags: set[str] = set()
    if "thameslink" in text:
        flags.add("thameslink")
    if "overground" in text:
        flags.add("overground")
    if "elizabeth" in text or "tflrail" in text:
        flags.add("elizabeth_line")
    if "national rail" in text or "national_rail" in text:
        flags.add("national_rail")
    if mode in RAIL_MODES:
        flags.add("rail")
    if mode == "tube" or "underground" in text:
        flags.add("tube")
    if mode == "bus":
        flags.add("bus")
    return flags


def _normalise_mode(value: str) -> str:
    mode = safe_feature_name(value).replace("_", "-")
    if mode in {"london-underground", "underground"}:
        return "tube"
    if mode in {"national-rail", "rail"}:
        return "national_rail"
    if mode in {"elizabeth-line", "tflrail"}:
        return "elizabeth-line"
    return mode or "unknown"


def _edge_seconds(
    lat_a: float,
    lng_a: float,
    lat_b: float,
    lng_b: float,
    mode: str,
    *,
    dwell_seconds: float,
) -> float:
    distance_m = float(haversine_m(lat_a, lng_a, lat_b, lng_b))
    speed_mps = DEFAULT_SPEED_KPH.get(mode, 25.0) * 1000 / 3600
    return max(35.0, distance_m / max(speed_mps, 0.1) + dwell_seconds)


def _node_id(prefix: str, name: str, suffix: str | int) -> str:
    return f"{prefix}_{safe_feature_name(name)}_{safe_feature_name(str(suffix))}"


def build_reference_from_station_catalog(
    stations: pd.DataFrame,
    *,
    include_curated_corridors: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a deterministic graph reference from the existing station file."""
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for index, station in stations.reset_index(drop=True).iterrows():
        name = str(station["station_name"])
        lines = _split_lines(station.get("lines"))
        mode = "tube"
        flags = sorted(
            set().union(*(_corridor_flags(mode, line, name) for line in lines))
            if lines
            else _corridor_flags(mode, "", name)
        )
        node_rows.append(
            {
                "node_id": _node_id("station", name, index),
                "name": name,
                "mode": mode,
                "operator": "TfL",
                "lat": float(station["lat"]),
                "lng": float(station["lng"]),
                "lines": ", ".join(lines),
                "corridor_flags": ", ".join(flags),
                "source": "station_catalog",
            }
        )

    nodes = pd.DataFrame(node_rows)
    for line in sorted(
        {line for lines in nodes["lines"] for line in _split_lines(lines)}
    ):
        line_nodes = nodes[nodes["lines"].str.contains(line, regex=False, na=False)]
        ordered = line_nodes.reset_index(drop=True)
        for index in range(len(ordered) - 1):
            left = ordered.iloc[index]
            right = ordered.iloc[index + 1]
            seconds = _edge_seconds(
                float(left["lat"]),
                float(left["lng"]),
                float(right["lat"]),
                float(right["lng"]),
                "tube",
                dwell_seconds=45.0,
            )
            edge_rows.extend(
                _bidirectional_edges(
                    left["node_id"],
                    right["node_id"],
                    mode="tube",
                    line=line,
                    travel_seconds=seconds,
                    source="station_catalog_order",
                )
            )

    if include_curated_corridors:
        _append_curated_corridors(node_rows, edge_rows)
        nodes = pd.DataFrame(node_rows)

    edges = pd.DataFrame(edge_rows)
    return _normalise_nodes(nodes), _normalise_edges(edges)


def _append_curated_corridors(
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> None:
    for corridor in CURATED_CORRIDORS:
        line = str(corridor["line"])
        mode = str(corridor["mode"])
        previous: dict[str, Any] | None = None
        for index, (name, lat, lng) in enumerate(corridor["stops"]):
            node = {
                "node_id": _node_id("corridor", f"{line}_{name}", index),
                "name": name,
                "mode": mode,
                "operator": corridor["operator"],
                "lat": float(lat),
                "lng": float(lng),
                "lines": line,
                "corridor_flags": ", ".join(sorted(_corridor_flags(mode, line, name))),
                "source": "curated_corridor",
            }
            node_rows.append(node)
            if previous is not None:
                seconds = _edge_seconds(
                    float(previous["lat"]),
                    float(previous["lng"]),
                    float(node["lat"]),
                    float(node["lng"]),
                    mode,
                    dwell_seconds=35.0,
                )
                edge_rows.extend(
                    _bidirectional_edges(
                        previous["node_id"],
                        node["node_id"],
                        mode=mode,
                        line=line,
                        travel_seconds=seconds,
                        source="curated_corridor",
                    )
                )
            previous = node


def fetch_tfl_transport_reference(
    modes: Iterable[str],
    *,
    bus_route_limit: int,
    app_key: str | None = None,
    timeout_seconds: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch route topology from TfL Unified API.

    The parser is defensive because the TfL response shape differs slightly by
    mode. If a mode fails, the caller can still merge the returned partial graph
    with deterministic local references.
    """
    session = requests.Session()
    node_by_id: dict[str, dict[str, Any]] = {}
    edge_rows: list[dict[str, Any]] = []
    params = {"app_key": app_key} if app_key else None

    mode_list = [mode for mode in modes if mode]
    if not mode_list:
        return pd.DataFrame(), pd.DataFrame()
    lines_url = f"https://api.tfl.gov.uk/Line/Mode/{','.join(mode_list)}"
    lines_response = session.get(lines_url, params=params, timeout=timeout_seconds)
    lines_response.raise_for_status()
    lines = lines_response.json()

    bus_seen = 0
    for line in lines:
        line_id = str(line.get("id") or line.get("name") or "")
        if not line_id:
            continue
        line_mode = _normalise_mode(str(line.get("modeName") or ""))
        if line_mode == "bus":
            bus_seen += 1
            if bus_seen > bus_route_limit:
                continue
        line_name = str(line.get("name") or line_id)
        for direction in ["inbound", "outbound"]:
            sequence_url = (
                f"https://api.tfl.gov.uk/Line/{line_id}/Route/Sequence/{direction}"
            )
            response = session.get(sequence_url, params=params, timeout=timeout_seconds)
            if response.status_code >= 400:
                continue
            for stop_sequence in _iter_tfl_stop_sequences(response.json()):
                previous_node_id: str | None = None
                previous_stop: dict[str, Any] | None = None
                for stop in stop_sequence:
                    stop_id = str(stop.get("id") or stop.get("naptanId") or "")
                    lat = stop.get("lat")
                    lng = stop.get("lon", stop.get("lng"))
                    if not stop_id or lat is None or lng is None:
                        continue
                    name = str(stop.get("name") or stop.get("commonName") or stop_id)
                    existing = node_by_id.get(stop_id)
                    if existing:
                        lines_value = set(_split_lines(existing["lines"]))
                        lines_value.add(line_name)
                        existing["lines"] = ", ".join(sorted(lines_value))
                        flags = set(_split_lines(existing["corridor_flags"]))
                        flags.update(_corridor_flags(line_mode, line_name, name))
                        existing["corridor_flags"] = ", ".join(sorted(flags))
                    else:
                        node_by_id[stop_id] = {
                            "node_id": stop_id,
                            "name": name,
                            "mode": line_mode,
                            "operator": "TfL",
                            "lat": float(lat),
                            "lng": float(lng),
                            "lines": line_name,
                            "corridor_flags": ", ".join(
                                sorted(_corridor_flags(line_mode, line_name, name))
                            ),
                            "source": "tfl_unified_api",
                        }
                    if previous_node_id and previous_stop:
                        seconds = _edge_seconds(
                            float(previous_stop["lat"]),
                            float(previous_stop["lng"]),
                            float(lat),
                            float(lng),
                            line_mode,
                            dwell_seconds=35.0 if line_mode != "bus" else 25.0,
                        )
                        edge_rows.extend(
                            _bidirectional_edges(
                                previous_node_id,
                                stop_id,
                                mode=line_mode,
                                line=line_name,
                                travel_seconds=seconds,
                                source="tfl_unified_api",
                            )
                        )
                    previous_node_id = stop_id
                    previous_stop = {"lat": float(lat), "lng": float(lng)}

    return _normalise_nodes(pd.DataFrame(node_by_id.values())), _normalise_edges(
        pd.DataFrame(edge_rows)
    )


def _iter_tfl_stop_sequences(payload: dict[str, Any]) -> Iterable[list[dict[str, Any]]]:
    for sequence in payload.get("stopPointSequences", []) or []:
        points = sequence.get("stopPoint") or sequence.get("stopPoints") or []
        if points:
            yield points
    for route in payload.get("orderedLineRoutes", []) or []:
        points = route.get("naptanIds") or []
        if points:
            yield [{"id": point} for point in points]
    stations = payload.get("stations") or []
    if stations:
        yield stations


def read_naptan_nodes(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_absolute():
        source = here() / source
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source)
    columns = {column.lower(): column for column in frame.columns}
    lat_col = columns.get("latitude") or columns.get("lat")
    lng_col = columns.get("longitude") or columns.get("lon") or columns.get("lng")
    name_col = (
        columns.get("commonname") or columns.get("name") or columns.get("stopname")
    )
    id_col = columns.get("atcocode") or columns.get("naptancode") or columns.get("id")
    type_col = columns.get("stoptype") or columns.get("type")
    if not all([lat_col, lng_col, name_col, id_col]):
        return pd.DataFrame()

    rows = []
    for _, row in frame.iterrows():
        stop_type = str(row[type_col]).lower() if type_col else ""
        mode = "national_rail" if "rail" in stop_type else "bus"
        rows.append(
            {
                "node_id": f"naptan_{row[id_col]}",
                "name": str(row[name_col]),
                "mode": mode,
                "operator": "NaPTAN",
                "lat": float(row[lat_col]),
                "lng": float(row[lng_col]),
                "lines": "",
                "corridor_flags": ", ".join(
                    sorted(_corridor_flags(mode, "", str(row[name_col])))
                ),
                "source": "naptan",
            }
        )
    return _normalise_nodes(pd.DataFrame(rows))


def merge_references(
    references: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_frames = [nodes for nodes, _ in references if not nodes.empty]
    edge_frames = [edges for _, edges in references if not edges.empty]
    nodes = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    if not nodes.empty:
        nodes = nodes.drop_duplicates("node_id", keep="first").reset_index(drop=True)
    if not edges.empty:
        edges = edges.drop_duplicates(
            ["from_node_id", "to_node_id", "mode", "line"], keep="first"
        ).reset_index(drop=True)
    return _normalise_nodes(nodes), _normalise_edges(edges)


def build_transport_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    transfer_radius_m: float,
    transfer_penalty_seconds: float,
    walking_speed_mps: float,
) -> TransportGraph:
    nodes = _normalise_nodes(nodes)
    edges = _normalise_edges(edges)
    transfer_edges = _build_transfer_edges(
        nodes,
        transfer_radius_m=transfer_radius_m,
        transfer_penalty_seconds=transfer_penalty_seconds,
        walking_speed_mps=walking_speed_mps,
    )
    if not transfer_edges.empty:
        edges = pd.concat([edges, transfer_edges], ignore_index=True)
        edges = edges.drop_duplicates(
            ["from_node_id", "to_node_id", "mode", "line"], keep="first"
        )
    adjacency: dict[str, list[dict[str, Any]]] = {
        node_id: [] for node_id in nodes["node_id"]
    }
    for _, edge in edges.iterrows():
        adjacency.setdefault(str(edge["from_node_id"]), []).append(edge.to_dict())
    return TransportGraph(
        nodes=nodes.reset_index(drop=True),
        edges=edges.reset_index(drop=True),
        adjacency=adjacency,
    )


def write_graph_artifacts(
    graph: TransportGraph,
    *,
    nodes_path: str | Path,
    edges_path: str | Path,
) -> None:
    resolved_nodes = _resolve_path(nodes_path)
    resolved_edges = _resolve_path(edges_path)
    resolved_nodes.parent.mkdir(parents=True, exist_ok=True)
    resolved_edges.parent.mkdir(parents=True, exist_ok=True)
    graph.nodes.to_parquet(resolved_nodes, index=False)
    graph.edges.to_parquet(resolved_edges, index=False)


def read_graph_artifacts(
    *,
    nodes_path: str | Path,
    edges_path: str | Path,
    fallback_stations: pd.DataFrame | None = None,
    transfer_radius_m: float = 180.0,
    transfer_penalty_seconds: float = 120.0,
    walking_speed_mps: float = 1.35,
) -> TransportGraph | None:
    resolved_nodes = _resolve_path(nodes_path)
    resolved_edges = _resolve_path(edges_path)
    if resolved_nodes.exists() and resolved_edges.exists():
        nodes = pd.read_parquet(resolved_nodes)
        edges = pd.read_parquet(resolved_edges)
        return build_transport_graph(
            nodes,
            edges,
            transfer_radius_m=transfer_radius_m,
            transfer_penalty_seconds=transfer_penalty_seconds,
            walking_speed_mps=walking_speed_mps,
        )
    if fallback_stations is None:
        return None
    nodes, edges = build_reference_from_station_catalog(fallback_stations)
    return build_transport_graph(
        nodes,
        edges,
        transfer_radius_m=transfer_radius_m,
        transfer_penalty_seconds=transfer_penalty_seconds,
        walking_speed_mps=walking_speed_mps,
    )


def add_graph_features(
    features: pd.DataFrame,
    graph: TransportGraph | None,
    *,
    walking_speed_mps: float,
    access_node_limit: int,
    max_access_distance_m: float,
    bus_density_radius_m: float,
) -> pd.DataFrame:
    if graph is None or graph.nodes.empty or graph.edges.empty:
        fallback = empty_graph_features(len(features))
        walking_seconds = features["haversine_distance_m"] / walking_speed_mps
        fallback_seconds = features.apply(
            lambda row: _transit_fallback_seconds(row, walking_speed_mps),
            axis=1,
        )
        fallback["graph_total_seconds"] = fallback_seconds
        fallback["graph_rail_advantage_seconds"] = walking_seconds - fallback_seconds
        fallback["graph_modes"] = "fallback"
        return pd.concat([features.reset_index(drop=True), fallback], axis=1)

    origin_points = (
        features[["origin_id", "origin_lat", "origin_lng"]]
        .drop_duplicates("origin_id")
        .rename(columns={"origin_lat": "lat", "origin_lng": "lng"})
    )
    destination_points = (
        features[["destination_id", "destination_lat", "destination_lng"]]
        .drop_duplicates("destination_id")
        .rename(columns={"destination_lat": "lat", "destination_lng": "lng"})
    )
    origin_nearest = _nearest_candidates(
        origin_points,
        graph.nodes,
        limit=access_node_limit,
        max_distance_m=max_access_distance_m,
        walking_speed_mps=walking_speed_mps,
    )
    destination_nearest = _nearest_candidates(
        destination_points,
        graph.nodes,
        limit=access_node_limit,
        max_distance_m=max_access_distance_m,
        walking_speed_mps=walking_speed_mps,
    )
    point_features = _point_access_features(
        origin_points,
        destination_points,
        graph.nodes,
        radius_m=bus_density_radius_m,
    )

    rows: list[dict[str, Any]] = []
    grouped = features.groupby("origin_id", sort=False)
    for origin_id, group in grouped:
        candidates = origin_nearest.get(str(origin_id), [])
        distances, previous, source_for_node = _multi_source_dijkstra(
            graph,
            candidates,
        )
        for _, row in group.iterrows():
            destination_id = str(row["destination_id"])
            destination_candidates = destination_nearest.get(destination_id, [])
            best = _best_destination_path(
                graph,
                destination_candidates,
                distances,
                previous,
                source_for_node,
            )
            walking_seconds = float(row["haversine_distance_m"]) / walking_speed_mps
            fallback_seconds = _transit_fallback_seconds(row, walking_speed_mps)
            if best is None or float(best["graph_total_seconds"]) > (
                fallback_seconds * 1.15
            ):
                graph_row = _fallback_graph_row(
                    fallback_seconds,
                    walking_seconds,
                )
            else:
                graph_row = best
                graph_row["graph_rail_advantage_seconds"] = walking_seconds - float(
                    graph_row["graph_total_seconds"]
                )
            graph_row.update(point_features["origin"].get(str(origin_id), {}))
            graph_row.update(point_features["destination"].get(destination_id, {}))
            origin_corridors = set(
                _split_lines(graph_row.get("origin_nearest_corridors", ""))
            )
            destination_corridors = set(
                _split_lines(graph_row.get("destination_nearest_corridors", ""))
            )
            graph_row["same_graph_corridor"] = float(
                bool(origin_corridors.intersection(destination_corridors))
            )
            graph_row["nearest_corridors"] = ", ".join(
                sorted(
                    set(_split_lines(str(graph_row.get("nearest_corridors", ""))))
                    | origin_corridors
                    | destination_corridors
                )
            )
            graph_row.pop("origin_nearest_corridors", None)
            graph_row.pop("destination_nearest_corridors", None)
            rows.append(graph_row)

    graph_features = pd.DataFrame(rows)
    graph_features = graph_features.reindex(
        columns=GRAPH_NUMERIC_COLUMNS + GRAPH_TEXT_COLUMNS
    )
    for column in GRAPH_NUMERIC_COLUMNS:
        graph_features[column] = pd.to_numeric(
            graph_features[column], errors="coerce"
        ).fillna(0.0)
    for column in GRAPH_TEXT_COLUMNS:
        graph_features[column] = graph_features[column].fillna("")
    return pd.concat(
        [features.reset_index(drop=True), graph_features.reset_index(drop=True)], axis=1
    )


def _normalise_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame(
            columns=[
                "node_id",
                "name",
                "mode",
                "operator",
                "lat",
                "lng",
                "lines",
                "corridor_flags",
                "source",
            ]
        )
    normalised = nodes.copy()
    normalised["node_id"] = normalised["node_id"].astype(str)
    normalised["name"] = normalised["name"].fillna("").astype(str)
    normalised["mode"] = normalised["mode"].fillna("unknown").map(_normalise_mode)
    normalised["operator"] = normalised["operator"].fillna("").astype(str)
    normalised["lat"] = pd.to_numeric(normalised["lat"], errors="coerce")
    normalised["lng"] = pd.to_numeric(normalised["lng"], errors="coerce")
    normalised["lines"] = normalised["lines"].fillna("").astype(str)
    normalised["corridor_flags"] = normalised["corridor_flags"].fillna("").astype(str)
    normalised["source"] = normalised.get("source", "").fillna("").astype(str)
    return normalised.dropna(subset=["lat", "lng"]).reset_index(drop=True)


def _normalise_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "from_node_id",
                "to_node_id",
                "mode",
                "line",
                "travel_seconds",
                "source",
            ]
        )
    normalised = edges.copy()
    normalised["from_node_id"] = normalised["from_node_id"].astype(str)
    normalised["to_node_id"] = normalised["to_node_id"].astype(str)
    normalised["mode"] = normalised["mode"].fillna("unknown").map(_normalise_mode)
    normalised["line"] = normalised["line"].fillna("").astype(str)
    normalised["travel_seconds"] = pd.to_numeric(
        normalised["travel_seconds"], errors="coerce"
    ).fillna(0.0)
    normalised["source"] = normalised.get("source", "").fillna("").astype(str)
    return normalised[normalised["travel_seconds"] > 0].reset_index(drop=True)


def _bidirectional_edges(
    left: str,
    right: str,
    *,
    mode: str,
    line: str,
    travel_seconds: float,
    source: str,
) -> list[dict[str, Any]]:
    return [
        {
            "from_node_id": left,
            "to_node_id": right,
            "mode": mode,
            "line": line,
            "travel_seconds": float(travel_seconds),
            "source": source,
        },
        {
            "from_node_id": right,
            "to_node_id": left,
            "mode": mode,
            "line": line,
            "travel_seconds": float(travel_seconds),
            "source": source,
        },
    ]


def _build_transfer_edges(
    nodes: pd.DataFrame,
    *,
    transfer_radius_m: float,
    transfer_penalty_seconds: float,
    walking_speed_mps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left_index, left in nodes.iterrows():
        distances = haversine_m(
            float(left["lat"]),
            float(left["lng"]),
            nodes["lat"].to_numpy(),
            nodes["lng"].to_numpy(),
        )
        for right_index in np.where((distances > 0) & (distances <= transfer_radius_m))[
            0
        ]:
            if int(right_index) <= int(left_index):
                continue
            right = nodes.iloc[int(right_index)]
            seconds = float(
                distances[int(right_index)] / walking_speed_mps
                + transfer_penalty_seconds
            )
            rows.extend(
                _bidirectional_edges(
                    str(left["node_id"]),
                    str(right["node_id"]),
                    mode="transfer",
                    line="walk_transfer",
                    travel_seconds=seconds,
                    source="transfer_radius",
                )
            )
    return _normalise_edges(pd.DataFrame(rows))


def _nearest_candidates(
    points: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    limit: int,
    max_distance_m: float,
    walking_speed_mps: float,
) -> dict[str, list[dict[str, Any]]]:
    node_lats = nodes["lat"].to_numpy()
    node_lngs = nodes["lng"].to_numpy()
    result: dict[str, list[dict[str, Any]]] = {}
    point_id_column = "origin_id" if "origin_id" in points.columns else "destination_id"
    for _, point in points.iterrows():
        distances = haversine_m(
            float(point["lat"]), float(point["lng"]), node_lats, node_lngs
        )
        order = np.argsort(distances)
        candidates: list[dict[str, Any]] = []
        for node_index in order[: max(limit * 4, limit)]:
            distance = float(distances[int(node_index)])
            if distance > max_distance_m and candidates:
                continue
            node = nodes.iloc[int(node_index)]
            candidates.append(
                {
                    "node_id": str(node["node_id"]),
                    "distance_m": distance,
                    "access_seconds": distance / walking_speed_mps,
                    "corridor_flags": str(node["corridor_flags"]),
                }
            )
            if len(candidates) >= limit:
                break
        result[str(point[point_id_column])] = candidates
    return result


def _multi_source_dijkstra(
    graph: TransportGraph,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, tuple[str, dict[str, Any]]], dict[str, str]]:
    distances: dict[str, float] = {}
    previous: dict[str, tuple[str, dict[str, Any]]] = {}
    source_for_node: dict[str, str] = {}
    heap: list[tuple[float, str]] = []
    for candidate in candidates:
        node_id = str(candidate["node_id"])
        cost = float(candidate["access_seconds"])
        if node_id not in distances or cost < distances[node_id]:
            distances[node_id] = cost
            source_for_node[node_id] = node_id
            heapq.heappush(heap, (cost, node_id))
    while heap:
        cost, node_id = heapq.heappop(heap)
        if cost > distances.get(node_id, float("inf")):
            continue
        for edge in graph.adjacency.get(node_id, []):
            next_node = str(edge["to_node_id"])
            next_cost = cost + float(edge["travel_seconds"])
            if next_cost < distances.get(next_node, float("inf")):
                distances[next_node] = next_cost
                previous[next_node] = (node_id, edge)
                source_for_node[next_node] = source_for_node.get(node_id, node_id)
                heapq.heappush(heap, (next_cost, next_node))
    return distances, previous, source_for_node


def _best_destination_path(
    graph: TransportGraph,
    destination_candidates: list[dict[str, Any]],
    distances: dict[str, float],
    previous: dict[str, tuple[str, dict[str, Any]]],
    source_for_node: dict[str, str],
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in destination_candidates:
        node_id = str(candidate["node_id"])
        if node_id not in distances:
            continue
        source_node = source_for_node.get(node_id)
        if source_node is None:
            continue
        total = float(distances[node_id] + candidate["access_seconds"])
        access_seconds = float(distances.get(source_node, 0.0))
        path_edges = _reconstruct_edges(node_id, previous)
        row = _path_feature_row(
            graph,
            path_edges,
            total_seconds=total,
            access_seconds=access_seconds,
            egress_seconds=float(candidate["access_seconds"]),
        )
        if best is None or total < best[0]:
            best = (total, row)
    return None if best is None else best[1]


def _reconstruct_edges(
    destination_node_id: str,
    previous: dict[str, tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    current = destination_node_id
    while current in previous:
        parent, edge = previous[current]
        edges.append(edge)
        current = parent
    edges.reverse()
    return edges


def _path_feature_row(
    graph: TransportGraph,
    path_edges: list[dict[str, Any]],
    *,
    total_seconds: float,
    access_seconds: float,
    egress_seconds: float,
) -> dict[str, Any]:
    transit_edges = [edge for edge in path_edges if edge["mode"] != "transfer"]
    modes = sorted({str(edge["mode"]) for edge in transit_edges})
    lines = [str(edge["line"]) for edge in transit_edges if str(edge["line"])]
    interchanges = 0
    previous_line: str | None = None
    for line in lines:
        if previous_line is not None and line != previous_line:
            interchanges += 1
        previous_line = line
    corridor_flags = set()
    for edge in transit_edges:
        corridor_flags.update(_corridor_flags(str(edge["mode"]), str(edge["line"])))
    return {
        "graph_has_path": 1.0,
        "graph_total_seconds": float(total_seconds),
        "graph_path_seconds": float(
            max(total_seconds - access_seconds - egress_seconds, 0.0)
        ),
        "graph_access_seconds": float(access_seconds),
        "graph_egress_seconds": float(egress_seconds),
        "graph_interchanges": float(interchanges),
        "graph_mode_count": float(len(modes)),
        "graph_rail_advantage_seconds": 0.0,
        "graph_uses_bus": float("bus" in modes),
        "graph_uses_tube": float("tube" in modes),
        "graph_uses_rail": float(any(mode in RAIL_MODES for mode in modes)),
        "graph_uses_overground": float(
            "overground" in modes or "overground" in corridor_flags
        ),
        "graph_uses_elizabeth_line": float(
            "elizabeth-line" in modes or "elizabeth_line" in corridor_flags
        ),
        "graph_uses_thameslink": float("thameslink" in corridor_flags),
        "graph_uses_national_rail": float(
            "national_rail" in modes or "national_rail" in corridor_flags
        ),
        "graph_modes": ", ".join(modes),
        "nearest_corridors": ", ".join(sorted(corridor_flags)),
    }


def _transit_fallback_seconds(row: pd.Series, walking_speed_mps: float) -> float:
    station_access_seconds = (
        float(row.get("origin_station_1_distance_m", 0.0))
        + float(row.get("destination_station_1_distance_m", 0.0))
    ) / walking_speed_mps
    in_vehicle_seconds = float(row.get("haversine_distance_m", 0.0)) / 8.5
    return max(300.0 + station_access_seconds + in_vehicle_seconds, 0.0)


def _fallback_graph_row(total_seconds: float, walking_seconds: float) -> dict[str, Any]:
    row: dict[str, Any] = {column: 0.0 for column in GRAPH_NUMERIC_COLUMNS}
    row.update(
        {
            "graph_has_path": 0.0,
            "graph_total_seconds": float(total_seconds),
            "graph_path_seconds": 0.0,
            "graph_access_seconds": 0.0,
            "graph_egress_seconds": 0.0,
            "graph_rail_advantage_seconds": float(walking_seconds - total_seconds),
            "graph_modes": "fallback",
            "nearest_corridors": "",
        }
    )
    return row


def _point_access_features(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    radius_m: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    bus_nodes = nodes[nodes["mode"] == "bus"].reset_index(drop=True)
    heavy_nodes = nodes[nodes["mode"].isin(HEAVY_RAIL_MODES)].reset_index(drop=True)
    return {
        "origin": _access_features_for_points(
            origins, nodes, bus_nodes, heavy_nodes, "origin", radius_m
        ),
        "destination": _access_features_for_points(
            destinations, nodes, bus_nodes, heavy_nodes, "destination", radius_m
        ),
    }


def _access_features_for_points(
    points: pd.DataFrame,
    nodes: pd.DataFrame,
    bus_nodes: pd.DataFrame,
    heavy_nodes: pd.DataFrame,
    prefix: str,
    radius_m: float,
) -> dict[str, dict[str, Any]]:
    point_id_column = "origin_id" if "origin_id" in points.columns else "destination_id"
    result: dict[str, dict[str, Any]] = {}
    for _, point in points.iterrows():
        row: dict[str, Any] = {}
        row[f"{prefix}_bus_stop_density"] = _density(point, bus_nodes, radius_m)
        row[f"{prefix}_bus_route_count"] = _route_count(point, bus_nodes, radius_m)
        row[f"{prefix}_nearest_heavy_rail_distance_m"] = _nearest_distance(
            point, heavy_nodes
        )
        row[f"{prefix}_nearest_corridors"] = _nearest_corridors(point, nodes, radius_m)
        result[str(point[point_id_column])] = row
    return result


def _density(point: pd.Series, nodes: pd.DataFrame, radius_m: float) -> float:
    if nodes.empty:
        return 0.0
    distances = haversine_m(
        float(point["lat"]),
        float(point["lng"]),
        nodes["lat"].to_numpy(),
        nodes["lng"].to_numpy(),
    )
    return float(np.sum(distances <= radius_m))


def _route_count(point: pd.Series, nodes: pd.DataFrame, radius_m: float) -> float:
    if nodes.empty:
        return 0.0
    distances = haversine_m(
        float(point["lat"]),
        float(point["lng"]),
        nodes["lat"].to_numpy(),
        nodes["lng"].to_numpy(),
    )
    lines: set[str] = set()
    for node_index in np.where(distances <= radius_m)[0]:
        lines.update(_split_lines(nodes.iloc[int(node_index)]["lines"]))
    return float(len(lines))


def _nearest_distance(point: pd.Series, nodes: pd.DataFrame) -> float:
    if nodes.empty:
        return 99_999.0
    distances = haversine_m(
        float(point["lat"]),
        float(point["lng"]),
        nodes["lat"].to_numpy(),
        nodes["lng"].to_numpy(),
    )
    return float(np.min(distances))


def _nearest_corridors(point: pd.Series, nodes: pd.DataFrame, radius_m: float) -> str:
    if nodes.empty:
        return ""
    distances = haversine_m(
        float(point["lat"]),
        float(point["lng"]),
        nodes["lat"].to_numpy(),
        nodes["lng"].to_numpy(),
    )
    flags: set[str] = set()
    for node_index in np.where(distances <= radius_m)[0]:
        flags.update(_split_lines(nodes.iloc[int(node_index)]["corridor_flags"]))
    if not flags:
        nearest = int(np.argmin(distances))
        flags.update(_split_lines(nodes.iloc[nearest]["corridor_flags"]))
    return ", ".join(sorted(flags))


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else here() / value
