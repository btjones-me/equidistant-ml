"""Offline travel-time surface prediction."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from equidistant_ml.surfaces.config import load_params, project_path
from equidistant_ml.surfaces.features import build_feature_frame
from equidistant_ml.surfaces.geo import haversine_m, read_station_catalog
from equidistant_ml.surfaces.grid import (
    bbox_from_params,
    build_destination_grid,
    build_h3_destination_grid,
)
from equidistant_ml.surfaces.models import (
    GroupCombine,
    ModelBundle,
    combine_surfaces,
    load_bundle,
    predict,
)
from equidistant_ml.surfaces.transport_graph import (
    GRAPH_NUMERIC_COLUMNS,
    GRAPH_TEXT_COLUMNS,
    add_graph_features,
    read_graph_artifacts,
)
from equidistant_ml.surfaces.traveltime import (
    TravelTimeClient,
    TravelTimeCredentials,
    fetch_origin_surfaces,
)

DEFAULT_PREDICTION_GRID = {
    "focus": "inner",
    "detail": "fine",
    "focuses": {
        "central": ["zone1_core"],
        "inner": ["zone1_core", "zone2_3_inner"],
        "wide": ["zone1_core", "zone2_3_inner", "outer_context"],
    },
    "details": {
        "fast": {"resolution_offset": -1},
        "fine": {"resolution_offset": 0},
    },
    "bands": {
        "zone1_core": {
            "label": "Zone 1 core",
            "north": 51.535,
            "south": 51.492,
            "west": -0.19,
            "east": -0.065,
            "resolution": 9,
            "buffer_degrees": 0.006,
        },
        "zone2_3_inner": {
            "label": "Zones 2-3",
            "north": 51.56,
            "south": 51.46,
            "west": -0.245,
            "east": 0.03,
            "resolution": 9,
            "buffer_degrees": 0.012,
        },
        "outer_context": {
            "label": "Outer context",
            "north": 51.65,
            "south": 51.37,
            "west": -0.43,
            "east": 0.18,
            "resolution": 8,
            "buffer_degrees": 0.018,
        },
    },
}


def _mtime(path: str) -> float:
    resolved = project_path(path)
    return resolved.stat().st_mtime if resolved.exists() else -1.0


@lru_cache(maxsize=4)
def _cached_params(path: str, mtime: float) -> dict:
    del mtime
    return load_params(path)


def _prediction_params() -> dict:
    return _cached_params("params.yaml", _mtime("params.yaml"))


@lru_cache(maxsize=8)
def _cached_station_catalog(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return read_station_catalog(path)


def _prediction_stations(params: dict) -> pd.DataFrame:
    path = str(params["features"]["stations_path"])
    return _cached_station_catalog(path, _mtime(path))


@lru_cache(maxsize=16)
def _cached_destination_grid(
    params_mtime: float,
    grid_mode: str,
    focus: str | None,
    detail: str | None,
    x_size: int | None,
    y_size: int | None,
) -> pd.DataFrame:
    params = _cached_params("params.yaml", params_mtime)
    return default_destination_grid(
        params,
        x_size,
        y_size,
        grid_mode=grid_mode,
        focus=focus,
        detail=detail,
    )


def _prediction_destination_grid(
    grid_mode: str,
    focus: str | None,
    detail: str | None,
    x_size: int | None,
    y_size: int | None,
) -> pd.DataFrame:
    return _cached_destination_grid(
        _mtime("params.yaml"),
        grid_mode,
        focus,
        detail,
        x_size,
        y_size,
    ).copy()


@lru_cache(maxsize=8)
def _cached_bundle(path: str, mtime: float) -> ModelBundle:
    del mtime
    return load_bundle(path)


def _load_bundle_if_exists(path: str) -> ModelBundle | None:
    resolved = project_path(path)
    if not resolved.exists():
        return None
    return _cached_bundle(str(resolved), resolved.stat().st_mtime)


@lru_cache(maxsize=4)
def _cached_graph_artifacts(
    nodes_path: str,
    nodes_mtime: float,
    edges_path: str,
    edges_mtime: float,
    fallback_stations_path: str,
    fallback_stations_mtime: float,
    transfer_radius_m: float,
    transfer_penalty_seconds: float,
    walking_speed_mps: float,
):
    del nodes_mtime, edges_mtime, fallback_stations_mtime
    fallback_stations = (
        _cached_station_catalog(fallback_stations_path, _mtime(fallback_stations_path))
        if fallback_stations_path
        else None
    )
    return read_graph_artifacts(
        nodes_path=nodes_path,
        edges_path=edges_path,
        fallback_stations=fallback_stations,
        transfer_radius_m=transfer_radius_m,
        transfer_penalty_seconds=transfer_penalty_seconds,
        walking_speed_mps=walking_speed_mps,
    )


def default_destination_grid(
    params: dict,
    x_size: int | None,
    y_size: int | None,
    *,
    grid_mode: str = "uniform",
    focus: str | None = None,
    detail: str | None = None,
) -> pd.DataFrame:
    if grid_mode == "h3":
        return h3_destination_grid(params, focus=focus, detail=detail)
    grid_params = params["grid"]
    return build_destination_grid(
        bbox_from_params(params),
        int(x_size or grid_params["x_size"]),
        int(y_size or grid_params["y_size"]),
    )


def h3_destination_grid(
    params: dict, *, focus: str | None = None, detail: str | None = None
) -> pd.DataFrame:
    grid_params = params.get("prediction_grid", DEFAULT_PREDICTION_GRID)
    focus_name = focus or grid_params.get("focus", "inner")
    detail_name = detail or grid_params.get("detail", "fine")
    focus_band_ids = grid_params["focuses"][focus_name]
    resolution_offset = int(
        grid_params.get("details", {}).get(detail_name, {}).get("resolution_offset", 0)
    )
    bands = []
    for band_id in focus_band_ids:
        band = dict(grid_params["bands"][band_id])
        band["id"] = band_id
        band["resolution"] = max(0, int(band["resolution"]) + resolution_offset)
        bands.append(band)
    return build_h3_destination_grid(bands)


def _labels_for_prediction(origin_id: str, destinations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_id": origin_id,
            "destination_id": destinations["destination_id"].astype(str),
            "reachable": True,
            "api_status": "PREDICT",
        }
    )


def _labels_for_origins(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
) -> pd.DataFrame:
    origin_ids = origins["origin_id"].astype(str).to_numpy()
    destination_ids = destinations["destination_id"].astype(str).to_numpy()
    return pd.DataFrame(
        {
            "origin_id": np.repeat(origin_ids, len(destination_ids)),
            "destination_id": np.tile(destination_ids, len(origin_ids)),
            "reachable": True,
            "api_status": "PREDICT",
        }
    )


def _heuristic_seconds(features: pd.DataFrame) -> np.ndarray:
    station_access = (
        features["origin_station_1_distance_m"]
        + features["destination_station_1_distance_m"]
    ) / 1.35
    transit = features["haversine_distance_m"] / 8.5
    return np.maximum(300 + station_access + transit, 0).to_numpy()


def _load_best_available_model(grid_mode: str = "uniform") -> ModelBundle | None:
    candidates = [
        "models/travel_time_graph_residual.joblib",
        "models/travel_time_model.joblib",
        "models/baseline_model.joblib",
    ]
    if grid_mode == "h3":
        candidates.insert(0, "models/travel_time_hillclimb_best.joblib")
    for path in candidates:
        bundle = _load_bundle_if_exists(path)
        if bundle is not None:
            return bundle
    return None


def _load_graph_baseline_model() -> ModelBundle | None:
    return _load_bundle_if_exists("models/graph_baseline_model.joblib")


def _add_prediction_graph_features(
    features: pd.DataFrame,
    params: dict,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    graph_params = params.get("transport_graph", {})
    feature_params = params.get("graph_features", {})
    del stations
    nodes_path = str(
        graph_params.get("nodes_path", "data/reference/transport_graph_nodes.parquet")
    )
    edges_path = str(
        graph_params.get("edges_path", "data/reference/transport_graph_edges.parquet")
    )
    fallback_stations_path = str(params.get("features", {}).get("stations_path", ""))
    transfer_radius_m = float(graph_params.get("transfer_radius_m", 180))
    transfer_penalty_seconds = float(graph_params.get("transfer_penalty_seconds", 120))
    walking_speed_mps = float(graph_params.get("walking_speed_mps", 1.35))
    graph = _cached_graph_artifacts(
        nodes_path,
        _mtime(nodes_path),
        edges_path,
        _mtime(edges_path),
        fallback_stations_path,
        _mtime(fallback_stations_path) if fallback_stations_path else -1.0,
        transfer_radius_m,
        transfer_penalty_seconds,
        walking_speed_mps,
    )
    return add_graph_features(
        features,
        graph,
        walking_speed_mps=walking_speed_mps,
        access_node_limit=int(feature_params.get("access_node_limit", 4)),
        max_access_distance_m=float(feature_params.get("max_access_distance_m", 1600)),
        bus_density_radius_m=float(feature_params.get("bus_density_radius_m", 500)),
    )


def _selected_friend_indexes(
    friend_count: int, included_friend_indexes: Iterable[int] | None
) -> list[int]:
    if included_friend_indexes is None:
        return list(range(friend_count))
    selected = sorted({int(index) for index in included_friend_indexes})
    if not selected:
        raise ValueError("At least one friend must be selected for the surface.")
    invalid = [index for index in selected if index < 0 or index >= friend_count]
    if invalid:
        raise ValueError(f"Selected friend indexes out of range: {invalid}")
    return selected


def _score_columns(
    frame: pd.DataFrame,
    columns: list[str],
    combine: GroupCombine,
) -> pd.Series:
    if not columns:
        raise ValueError("No surface columns available to combine.")
    return combine_surfaces(frame[columns], combine)


def _stable_origin_id(
    index: int,
    lat: float,
    lng: float,
    focus: str,
    detail: str,
) -> str:
    payload = json.dumps(
        {
            "index": index,
            "lat": round(float(lat), 7),
            "lng": round(float(lng), 7),
            "focus": focus,
            "detail": detail,
        },
        sort_keys=True,
    )
    return f"friend_{index}_{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _corr(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    if len(values_a) < 2 or np.allclose(values_a, values_a[0]):
        return None
    if np.allclose(values_b, values_b[0]):
        return None
    return float(np.corrcoef(values_a, values_b)[0, 1])


def predict_origin_surface(
    origin_lat: float,
    origin_lng: float,
    *,
    origin_id: str = "origin_0",
    x_size: int | None = None,
    y_size: int | None = None,
    grid_mode: str = "uniform",
    focus: str | None = None,
    detail: str | None = None,
) -> pd.DataFrame:
    params = _prediction_params()
    destinations = _prediction_destination_grid(
        grid_mode,
        focus,
        detail,
        x_size,
        y_size,
    )
    origins = pd.DataFrame(
        [{"origin_id": origin_id, "lat": float(origin_lat), "lng": float(origin_lng)}]
    )
    stations = _prediction_stations(params)
    return _predict_surfaces_for_origins(
        origins,
        destinations,
        params,
        stations,
        grid_mode=grid_mode,
    )


def _predict_surfaces_for_origins(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    params: dict,
    stations: pd.DataFrame,
    *,
    grid_mode: str,
) -> pd.DataFrame:
    labels = _labels_for_origins(origins, destinations)
    features = build_feature_frame(
        labels,
        origins,
        destinations,
        stations,
        nearest_station_count=int(params["features"]["nearest_station_count"]),
        density_radius_m=float(params["features"]["density_radius_m"]),
        include_target=False,
    )
    features = _add_prediction_graph_features(features, params, stations)
    bundle = _load_best_available_model(grid_mode)
    predictions = predict(bundle, features) if bundle else _heuristic_seconds(features)
    graph_baseline = _load_graph_baseline_model()
    if graph_baseline:
        graph_predictions = predict(graph_baseline, features)
    elif "graph_total_seconds" in features:
        graph_predictions = features["graph_total_seconds"].to_numpy()
    else:
        graph_predictions = _heuristic_seconds(features)
    result = features[["origin_id", "destination_id"]].copy()
    result = result.merge(destinations, on="destination_id", how="left")
    result["travel_time_seconds"] = predictions
    result["travel_time_minutes"] = result["travel_time_seconds"] / 60
    result["graph_score_seconds"] = graph_predictions
    result["graph_score_minutes"] = result["graph_score_seconds"] / 60
    result["model_residual_seconds"] = (
        result["travel_time_seconds"] - result["graph_score_seconds"]
    )
    result["model_residual_minutes"] = result["model_residual_seconds"] / 60
    graph_columns = GRAPH_NUMERIC_COLUMNS + GRAPH_TEXT_COLUMNS
    for column in graph_columns:
        if column in features:
            result[column] = features[column].to_numpy()
    return result


def predict_group_surface(
    friends: Iterable[dict],
    *,
    combine: GroupCombine = "balanced",
    included_friend_indexes: Iterable[int] | None = None,
    x_size: int | None = None,
    y_size: int | None = None,
    grid_mode: str = "uniform",
    focus: str | None = None,
    detail: str | None = None,
) -> pd.DataFrame:
    friend_list = list(friends)
    selected_indexes = _selected_friend_indexes(
        len(friend_list),
        included_friend_indexes,
    )
    params = _prediction_params()
    destinations = _prediction_destination_grid(
        grid_mode,
        focus,
        detail,
        x_size,
        y_size,
    )
    stations = _prediction_stations(params)
    origins = pd.DataFrame(
        [
            {
                "origin_id": f"friend_{index}",
                "lat": float(friend["lat"]),
                "lng": float(friend["lng"]),
            }
            for index, friend in enumerate(friend_list)
        ]
    )
    batch_surfaces = _predict_surfaces_for_origins(
        origins,
        destinations,
        params,
        stations,
        grid_mode=grid_mode,
    )
    surfaces = []
    for index, friend in enumerate(friend_list):
        surface = batch_surfaces[batch_surfaces["origin_id"] == f"friend_{index}"]
        grid_columns = [
            column
            for column in [
                "destination_id",
                "lat",
                "lng",
                "x_index",
                "y_index",
                "south",
                "north",
                "west",
                "east",
                "h3_cell",
                "h3_resolution",
                "boundary",
                "grid_band",
                "grid_priority",
                "cell_area_km2",
                "travel_time_seconds",
                "graph_score_seconds",
                "model_residual_seconds",
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
                "graph_modes",
                "nearest_corridors",
            ]
            if column in surface.columns
        ]
        renamed = surface[grid_columns].rename(
            columns={
                "travel_time_seconds": f"friend_{index}_seconds",
                "graph_score_seconds": f"friend_{index}_graph_seconds",
                "model_residual_seconds": f"friend_{index}_model_residual_seconds",
                "graph_modes": f"friend_{index}_graph_modes",
                "nearest_corridors": f"friend_{index}_nearest_corridors",
                "graph_interchanges": f"friend_{index}_graph_interchanges",
                "graph_access_seconds": f"friend_{index}_graph_access_seconds",
                "graph_egress_seconds": f"friend_{index}_graph_egress_seconds",
            }
        )
        if index > 0:
            friend_specific = [
                column
                for column in renamed.columns
                if column == "destination_id" or column.startswith(f"friend_{index}_")
            ]
            renamed = renamed[friend_specific]
        surfaces.append(renamed)
    combined = surfaces[0]
    for surface in surfaces[1:]:
        combined = combined.merge(
            surface,
            on="destination_id",
        )
    all_friend_columns = [
        column for column in combined.columns if column.endswith("_seconds")
    ]
    friend_columns = [f"friend_{index}_seconds" for index in selected_indexes]
    graph_friend_columns = [
        f"friend_{index}_graph_seconds"
        for index in selected_indexes
        if f"friend_{index}_graph_seconds" in combined
    ]
    for column in all_friend_columns:
        combined[column.replace("_seconds", "_minutes")] = combined[column] / 60
    for column in [
        column for column in combined.columns if column.endswith("_graph_seconds")
    ]:
        combined[column.replace("_seconds", "_minutes")] = combined[column] / 60
    for column in [
        column
        for column in combined.columns
        if column.endswith("_model_residual_seconds")
    ]:
        combined[column.replace("_seconds", "_minutes")] = combined[column] / 60
    combined["score_seconds"] = _score_columns(combined, friend_columns, combine)
    combined["score_minutes"] = combined["score_seconds"] / 60
    if graph_friend_columns:
        combined["graph_score_seconds"] = _score_columns(
            combined, graph_friend_columns, combine
        )
    else:
        combined["graph_score_seconds"] = np.nan
    combined["graph_score_minutes"] = combined["graph_score_seconds"] / 60
    combined["max_seconds"] = combined[friend_columns].max(axis=1)
    combined["mean_seconds"] = combined[friend_columns].mean(axis=1)
    combined["fairness_seconds"] = combined[friend_columns].std(axis=1).fillna(0)
    combined["max_minutes"] = combined["max_seconds"] / 60
    combined["mean_minutes"] = combined["mean_seconds"] / 60
    combined["fairness_minutes"] = combined["fairness_seconds"] / 60
    combined["model_residual_seconds"] = (
        combined["score_seconds"] - combined["graph_score_seconds"]
    )
    combined["model_residual_minutes"] = combined["model_residual_seconds"] / 60
    for index, friend in enumerate(friend_list):
        combined[f"friend_{index}_name"] = friend.get("name") or f"Friend {index + 1}"
        if f"friend_{index}_seconds" in combined:
            combined[f"friend_{index}_model_seconds"] = combined[
                f"friend_{index}_seconds"
            ]
            combined[f"friend_{index}_model_minutes"] = combined[
                f"friend_{index}_minutes"
            ]
        if f"friend_{index}_graph_seconds" in combined:
            combined[f"friend_{index}_graph_minutes"] = (
                combined[f"friend_{index}_graph_seconds"] / 60
            )
    combined["model_score_seconds"] = combined["score_seconds"]
    combined["model_score_minutes"] = combined["score_minutes"]
    combined["included_friend_indexes"] = ",".join(
        str(index) for index in selected_indexes
    )
    return combined


def reference_group_surface(
    friends: Iterable[dict],
    *,
    combine: GroupCombine = "balanced",
    included_friend_indexes: Iterable[int] | None = None,
    x_size: int | None = None,
    y_size: int | None = None,
    grid_mode: str = "h3",
    focus: str | None = None,
    detail: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    if grid_mode != "h3":
        raise ValueError("TravelTime reference comparison currently expects h3 grids.")
    load_dotenv(project_path(".env"))
    params = load_params()
    focus_name = str(focus or DEFAULT_PREDICTION_GRID["focus"])
    detail_name = str(detail or DEFAULT_PREDICTION_GRID["detail"])
    friend_list = list(friends)
    selected_indexes = _selected_friend_indexes(
        len(friend_list),
        included_friend_indexes,
    )
    destinations = default_destination_grid(
        params,
        x_size,
        y_size,
        grid_mode=grid_mode,
        focus=focus_name,
        detail=detail_name,
    )
    model_surface = predict_group_surface(
        friend_list,
        combine=combine,
        included_friend_indexes=selected_indexes,
        x_size=x_size,
        y_size=y_size,
        grid_mode=grid_mode,
        focus=focus_name,
        detail=detail_name,
    )
    travel_params = params["traveltime"]
    client = TravelTimeClient(
        TravelTimeCredentials.from_env(),
        timeout_seconds=int(travel_params["timeout_seconds"]),
        sleep_seconds=0.0,
    )
    cache_dir = project_path(
        f"data/holdout/ui_reference/{focus_name}_{detail_name}/labels.parts"
    )

    result = model_surface.copy()
    fetched_origin_count = 0
    reference_columns: list[str] = []
    for index in selected_indexes:
        friend = friend_list[index]
        origin_id = _stable_origin_id(
            index,
            float(friend["lat"]),
            float(friend["lng"]),
            focus_name,
            detail_name,
        )
        origins = pd.DataFrame(
            [
                {
                    "origin_id": origin_id,
                    "lat": float(friend["lat"]),
                    "lng": float(friend["lng"]),
                }
            ]
        )
        labels = fetch_origin_surfaces(
            origins,
            destinations,
            client,
            transportation_type=travel_params["transportation_type"],
            arrival_time_period=travel_params["arrival_time_period"],
            travel_time_seconds=int(travel_params["travel_time_seconds"]),
            unreachable_penalty_seconds=int(
                travel_params["unreachable_penalty_seconds"]
            ),
            properties=travel_params["properties"],
            checkpoint_dir=cache_dir,
        )
        fetched_origin_count += 1
        labels = labels.rename(
            columns={
                "target_travel_time_seconds": f"friend_{index}_reference_seconds",
                "reachable": f"friend_{index}_reference_reachable",
            }
        )[
            [
                "destination_id",
                f"friend_{index}_reference_seconds",
                f"friend_{index}_reference_reachable",
            ]
        ]
        result = result.merge(labels, on="destination_id", how="left")
        result[f"friend_{index}_reference_minutes"] = (
            result[f"friend_{index}_reference_seconds"] / 60
        )
        result[f"friend_{index}_error_minutes"] = (
            result[f"friend_{index}_model_minutes"]
            - result[f"friend_{index}_reference_minutes"]
        )
        reference_columns.append(f"friend_{index}_reference_seconds")

    result["reference_score_seconds"] = _score_columns(
        result,
        reference_columns,
        combine,
    )
    result["reference_score_minutes"] = result["reference_score_seconds"] / 60
    result["signed_error_minutes"] = (
        result["model_score_minutes"] - result["reference_score_minutes"]
    )
    result["abs_error_minutes"] = result["signed_error_minutes"].abs()

    abs_error = result["abs_error_minutes"].replace([np.inf, -np.inf], np.nan).dropna()
    signed_error = (
        result["signed_error_minutes"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )
    model_distance_corrs = []
    reference_distance_corrs = []
    for index in selected_indexes:
        friend = friend_list[index]
        distances = haversine_m(
            float(friend["lat"]),
            float(friend["lng"]),
            result["lat"].to_numpy(),
            result["lng"].to_numpy(),
        )
        model_corr = _corr(
            distances,
            result[f"friend_{index}_model_minutes"].to_numpy(),
        )
        reference_corr = _corr(
            distances,
            result[f"friend_{index}_reference_minutes"].to_numpy(),
        )
        if model_corr is not None:
            model_distance_corrs.append(model_corr)
        if reference_corr is not None:
            reference_distance_corrs.append(reference_corr)

    metrics = {
        "reference_origin_count": fetched_origin_count,
        "included_friend_indexes": selected_indexes,
        "mae_minutes": float(abs_error.mean()) if not abs_error.empty else None,
        "median_abs_error_minutes": (
            float(abs_error.median()) if not abs_error.empty else None
        ),
        "p90_abs_error_minutes": (
            float(abs_error.quantile(0.90)) if not abs_error.empty else None
        ),
        "within_5_min_pct": (
            float((abs_error <= 5).mean() * 100) if not abs_error.empty else None
        ),
        "within_10_min_pct": (
            float((abs_error <= 10).mean() * 100) if not abs_error.empty else None
        ),
        "mean_signed_error_minutes": (
            float(signed_error.mean()) if not signed_error.empty else None
        ),
        "model_distance_correlation": (
            float(np.mean(model_distance_corrs)) if model_distance_corrs else None
        ),
        "reference_distance_correlation": (
            float(np.mean(reference_distance_corrs))
            if reference_distance_corrs
            else None
        ),
    }
    return result, metrics


def surface_to_grid_response(df: pd.DataFrame, value_column: str) -> dict:
    is_h3 = "h3_cell" in df.columns
    rectangular = not is_h3 and not df[["y_index", "x_index"]].duplicated().any()
    if rectangular:
        pivot = df.pivot(
            index="y_index", columns="x_index", values=value_column
        ).sort_index()
        complete = pivot.notna().all().all()
    else:
        pivot = pd.DataFrame()
        complete = False
    if rectangular and complete:
        lats = (
            df.sort_values("y_index")
            .drop_duplicates("y_index")
            .sort_values("y_index")["lat"]
            .round(7)
            .tolist()
        )
        lngs = (
            df.sort_values("x_index")
            .drop_duplicates("x_index")
            .sort_values("x_index")["lng"]
            .round(7)
            .tolist()
        )
        z_values = pivot.to_numpy().round(3).tolist()
    else:
        lats = []
        lngs = []
        z_values = []
    values = df[value_column].replace([np.inf, -np.inf], np.nan).dropna()
    friend_columns = sorted(
        column
        for column in df.columns
        if column.startswith("friend_") and column.endswith("_minutes")
    )
    geometry_columns = {
        "destination_id",
        "lat",
        "lng",
        "x_index",
        "y_index",
        "south",
        "north",
        "west",
        "east",
        "boundary",
        "h3_cell",
        "h3_resolution",
        "grid_band",
        "grid_priority",
        "cell_area_km2",
        "included_friend_indexes",
    }
    diagnostic_columns = {
        "graph_modes",
        "nearest_corridors",
        "graph_interchanges",
        "graph_access_seconds",
        "graph_egress_seconds",
    }
    response_columns = [
        column
        for column in df.columns
        if column in geometry_columns
        or column in diagnostic_columns
        or column.endswith("_minutes")
        or (column.startswith("friend_") and column.endswith("_name"))
    ]
    cells = df[response_columns].replace([np.inf, -np.inf], np.nan)
    cells = cells.astype(object).where(pd.notna(cells), None)
    return {
        "lats": lats,
        "lngs": lngs,
        "Z": z_values,
        "cells": cells.to_dict(orient="records"),
        "metadata": {
            "value_column": value_column,
            "cell_count": int(len(df)),
            "grid_type": "h3" if "h3_cell" in df.columns else "uniform",
            "friend_columns": friend_columns,
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
            "p10": float(values.quantile(0.10)) if not values.empty else None,
            "p50": float(values.quantile(0.50)) if not values.empty else None,
            "p90": float(values.quantile(0.90)) if not values.empty else None,
            "source": "api",
        },
    }
