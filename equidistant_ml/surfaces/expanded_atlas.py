"""Export the isolated Harrow-to-Sidcup candidate browser atlas."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from equidistant_ml.surfaces.config import load_params, project_path
from equidistant_ml.surfaces.export_atlas import _json_value
from equidistant_ml.surfaces.expanded_run import (
    _artifact_record,
    _atomic_json,
    _canonical_sha256,
    _file_sha256,
    _regression_report,
    _slice_reports,
    run_paths,
)
from equidistant_ml.surfaces.features import build_feature_frame
from equidistant_ml.surfaces.geo import haversine_m
from equidistant_ml.surfaces.models import load_bundle, predict
from equidistant_ml.surfaces.predict import _labels_for_origins
from equidistant_ml.surfaces.transport_graph import (
    add_graph_features,
    build_transport_graph,
)

ATLAS_STEP_MINUTES = 1.0
ATLAS_MAX_VALUE = 255

ATLAS_CELL_COLUMNS = [
    "destination_id",
    "lat",
    "lng",
    "boundary",
    "h3_cell",
    "h3_resolution",
    "grid_band",
    "grid_priority",
    "coverage_region",
    "cell_area_km2",
    "nearest_station_name",
    "nearest_station_lines",
]


def _atlas_cell_records(destinations: pd.DataFrame) -> list[dict]:
    """Return JSON-safe cell metadata, including nested NumPy boundaries."""
    return [
        {
            key: _json_value(row[key])
            for key in ATLAS_CELL_COLUMNS
            if key in destinations.columns
        }
        for _, row in destinations.iterrows()
    ]


def _wide_grid_axes(params: dict) -> tuple[np.ndarray, np.ndarray]:
    run = params["expanded_run"]
    atlas = run["atlas"]
    bounds = run["expanded_bounds"]
    inner_lats = np.linspace(
        float(atlas["inner_south"]),
        float(atlas["inner_north"]),
        int(atlas["inner_lat_count"]),
    )
    inner_lngs = np.linspace(
        float(atlas["inner_west"]),
        float(atlas["inner_east"]),
        int(atlas["inner_lng_count"]),
    )
    lat_spacing = float(inner_lats[1] - inner_lats[0]) * float(
        atlas["outer_spacing_multiplier"]
    )
    lng_spacing = float(inner_lngs[1] - inner_lngs[0]) * float(
        atlas["outer_spacing_multiplier"]
    )
    outer_lat_count = (
        round((float(bounds["north"]) - float(bounds["south"])) / lat_spacing) + 1
    )
    outer_lng_count = (
        round((float(bounds["east"]) - float(bounds["west"])) / lng_spacing) + 1
    )
    return (
        np.linspace(float(bounds["south"]), float(bounds["north"]), outer_lat_count),
        np.linspace(float(bounds["west"]), float(bounds["east"]), outer_lng_count),
    )


def _inside_inner(lat: float, lng: float, atlas: dict) -> bool:
    return float(atlas["inner_south"]) <= lat <= float(atlas["inner_north"]) and float(
        atlas["inner_west"]
    ) <= lng <= float(atlas["inner_east"])


def expanded_atlas_origins(params: dict) -> pd.DataFrame:
    """Keep the 560 central anchors and add the base twice-spaced outer grid."""
    run = params["expanded_run"]
    atlas = run["atlas"]
    inner_lats = np.linspace(
        float(atlas["inner_south"]),
        float(atlas["inner_north"]),
        int(atlas["inner_lat_count"]),
    )
    inner_lngs = np.linspace(
        float(atlas["inner_west"]),
        float(atlas["inner_east"]),
        int(atlas["inner_lng_count"]),
    )
    wide_lats, wide_lngs = _wide_grid_axes(params)
    rows = [
        {
            "lat": float(lat),
            "lng": float(lng),
            "anchor_region": "central",
            "anchor_source": "grid",
            "lat_index": lat_index,
            "lng_index": lng_index,
        }
        for lat_index, lat in enumerate(inner_lats)
        for lng_index, lng in enumerate(inner_lngs)
    ]
    for lat_index, lat in enumerate(wide_lats):
        for lng_index, lng in enumerate(wide_lngs):
            if not _inside_inner(float(lat), float(lng), atlas):
                rows.append(
                    {
                        "lat": float(lat),
                        "lng": float(lng),
                        "anchor_region": "outer",
                        "anchor_source": "grid",
                        "lat_index": lat_index,
                        "lng_index": lng_index,
                    }
                )
    origins = pd.DataFrame(rows)
    origins.insert(
        0,
        "origin_id",
        [
            f"atlas_{row.anchor_region}_{index:04d}"
            for index, row in enumerate(origins.itertuples(index=False))
        ],
    )
    return origins


def adaptive_atlas_probe_origins(params: dict) -> pd.DataFrame:
    """Return disjoint outer probes for anchor selection and validation.

    Quarter-cell probes exercise the gaps in the base outer grid. A checkerboard
    assignment keeps validation probes out of adaptive anchor selection, and the
    TravelTime train/tune/test origins are never consulted.
    """
    atlas = params["expanded_run"]["atlas"]
    wide_lats, wide_lngs = _wide_grid_axes(params)
    rows: list[dict] = []
    for lat_index in range(len(wide_lats) - 1):
        for lng_index in range(len(wide_lngs) - 1):
            for lat_quarter in (0.25, 0.75):
                for lng_quarter in (0.25, 0.75):
                    lat = float(
                        wide_lats[lat_index]
                        + (wide_lats[lat_index + 1] - wide_lats[lat_index])
                        * lat_quarter
                    )
                    lng = float(
                        wide_lngs[lng_index]
                        + (wide_lngs[lng_index + 1] - wide_lngs[lng_index])
                        * lng_quarter
                    )
                    if _inside_inner(lat, lng, atlas):
                        continue
                    parity = (
                        lat_index
                        + lng_index
                        + int(lat_quarter * 4)
                        + int(lng_quarter * 4)
                    ) % 2
                    rows.append(
                        {
                            "origin_id": f"atlas_probe_{len(rows):04d}",
                            "lat": lat,
                            "lng": lng,
                            "probe_role": "selection" if parity == 0 else "validation",
                            "lat_index": lat_index,
                            "lng_index": lng_index,
                        }
                    )
    return pd.DataFrame(rows)


def _station_catalog(nodes: pd.DataFrame) -> pd.DataFrame:
    stations = nodes[~nodes["mode"].isin(["bus", "transfer"])].copy()
    stations = stations.rename(columns={"name": "station_name"})
    stations["lines"] = stations["lines"].fillna("").astype(str)
    return stations[["station_name", "lat", "lng", "lines"]].drop_duplicates(
        ["station_name", "lat", "lng"]
    )


def _quantise(values_seconds: np.ndarray) -> np.ndarray:
    return (
        np.rint(values_seconds / 60 / ATLAS_STEP_MINUTES)
        .clip(0, ATLAS_MAX_VALUE)
        .astype(np.uint8)
    )


def _interpolate(
    atlas: np.ndarray,
    anchors: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    neighbours: int,
    discontinuity_threshold_minutes: float | None = None,
) -> np.ndarray:
    result = np.empty((len(origins), atlas.shape[1]), dtype=np.float32)
    anchor_lats = anchors["lat"].to_numpy()
    anchor_lngs = anchors["lng"].to_numpy()
    signatures = (
        anchors["surface_signature_minutes"].to_numpy(dtype=float)
        if "surface_signature_minutes" in anchors
        else None
    )
    for row_index, row in enumerate(origins.itertuples(index=False)):
        distances = haversine_m(row.lat, row.lng, anchor_lats, anchor_lngs)
        selected = _select_interpolation_neighbours(
            distances,
            neighbours,
            signatures=signatures,
            discontinuity_threshold_minutes=discontinuity_threshold_minutes,
        )
        if distances[selected].min() < 20:
            result[row_index] = atlas[selected[np.argmin(distances[selected])]]
            continue
        selected_distances = np.maximum(distances[selected], 40.0)
        weights = 1.0 / np.square(selected_distances)
        weights /= weights.sum()
        result[row_index] = np.sum(atlas[selected] * weights[:, None], axis=0)
    return result


def _select_interpolation_neighbours(
    distances: np.ndarray,
    neighbours: int,
    *,
    signatures: np.ndarray | None,
    discontinuity_threshold_minutes: float | None,
) -> np.ndarray:
    """Choose nearby anchors without blending incompatible time regimes."""
    count = min(max(int(neighbours), 1), len(distances))
    selected = np.argpartition(distances, count - 1)[:count]
    selected = selected[np.argsort(distances[selected])]
    if (
        discontinuity_threshold_minutes is None
        or signatures is None
        or len(selected) < 3
    ):
        return selected
    selected_signatures = signatures[selected]
    signature_order = np.argsort(selected_signatures)
    gaps = np.diff(selected_signatures[signature_order])
    if not len(gaps):
        return selected
    split = int(np.argmax(gaps))
    if float(gaps[split]) <= float(discontinuity_threshold_minutes):
        return selected
    groups = [signature_order[: split + 1], signature_order[split + 1 :]]
    if len(groups[0]) > len(groups[1]):
        compatible = groups[0]
    elif len(groups[1]) > len(groups[0]):
        compatible = groups[1]
    else:
        compatible = next(group for group in groups if 0 in group)
    return selected[compatible]


def _feature_batch(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    stations: pd.DataFrame,
    graph,
    access_nodes: pd.DataFrame,
    params: dict,
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
    graph_params = params["transport_graph"]
    feature_params = params["graph_features"]
    return add_graph_features(
        features,
        graph,
        walking_speed_mps=float(graph_params["walking_speed_mps"]),
        access_node_limit=int(feature_params["access_node_limit"]),
        max_access_distance_m=float(feature_params["max_access_distance_m"]),
        bus_density_radius_m=float(feature_params["bus_density_radius_m"]),
        access_nodes=access_nodes,
    )


def _predict_anchor_surfaces(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    stations: pd.DataFrame,
    graph,
    access_nodes: pd.DataFrame,
    selected,
    graph_baseline,
    params: dict,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(origins), len(destinations))
    model_values = np.empty(shape, dtype=np.uint8)
    graph_values = np.empty(shape, dtype=np.uint8)
    for start in range(0, len(origins), batch_size):
        batch = origins.iloc[start : start + batch_size]
        features = _feature_batch(
            batch, destinations, stations, graph, access_nodes, params
        )
        model_prediction = predict(selected, features).reshape(len(batch), -1)
        graph_prediction = predict(graph_baseline, features).reshape(len(batch), -1)
        model_values[start : start + len(batch)] = _quantise(model_prediction)
        graph_values[start : start + len(batch)] = _quantise(graph_prediction)
        print(f"Predicted {start + len(batch)}/{len(origins)} origins", flush=True)
    return model_values, graph_values


def _probe_error_report(
    direct_values: np.ndarray, interpolated_values: np.ndarray
) -> dict:
    absolute = np.abs(direct_values.astype(np.float32) - interpolated_values)
    per_origin = absolute.mean(axis=1)
    return {
        "origins": int(len(direct_values)),
        "rows": int(absolute.size),
        "mae_minutes": float(absolute.mean()),
        "median_absolute_error_minutes": float(np.median(absolute)),
        "p90_absolute_error_minutes": float(np.quantile(absolute, 0.9)),
        "p95_absolute_error_minutes": float(np.quantile(absolute, 0.95)),
        "origin_macro_mae_minutes": float(per_origin.mean()),
        "origin_p90_mae_minutes": float(np.quantile(per_origin, 0.9)),
        "worst_origin_mae_minutes": float(per_origin.max()),
    }


def _adaptive_anchors_pass_validation(before: dict, candidate: dict) -> bool:
    return (
        candidate["mae_minutes"] < before["mae_minutes"]
        and candidate["p90_absolute_error_minutes"]
        <= before["p90_absolute_error_minutes"]
    )


def _select_adaptive_anchors(
    probes: pd.DataFrame,
    direct_values: np.ndarray,
    interpolated_values: np.ndarray,
    *,
    count: int,
    min_separation_m: float,
    min_mae_minutes: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    errors = np.abs(direct_values.astype(np.float32) - interpolated_values).mean(axis=1)
    ranked = np.argsort(errors)[::-1]
    selected: list[int] = []
    for index in ranked:
        if float(errors[index]) < min_mae_minutes or len(selected) >= count:
            break
        if selected:
            distances = haversine_m(
                float(probes.iloc[index]["lat"]),
                float(probes.iloc[index]["lng"]),
                probes.iloc[selected]["lat"].to_numpy(),
                probes.iloc[selected]["lng"].to_numpy(),
            )
            if float(distances.min()) < min_separation_m:
                continue
        selected.append(int(index))
    chosen = probes.iloc[selected].copy().reset_index(drop=True)
    chosen["origin_id"] = [
        f"atlas_outer_adaptive_{index:04d}" for index in range(len(chosen))
    ]
    chosen["anchor_region"] = "outer"
    chosen["anchor_source"] = "adaptive_probe"
    chosen["adaptive_probe_mae_minutes"] = errors[selected]
    return chosen, np.asarray(selected, dtype=int)


def _attach_nearest_station(
    destinations: pd.DataFrame, stations: pd.DataFrame
) -> pd.DataFrame:
    result = destinations.copy()
    indexes = [
        int(
            np.argmin(
                haversine_m(
                    row.lat,
                    row.lng,
                    stations["lat"].to_numpy(),
                    stations["lng"].to_numpy(),
                )
            )
        )
        for row in result.itertuples(index=False)
    ]
    result["nearest_station_name"] = [
        stations.iloc[index]["station_name"] for index in indexes
    ]
    result["nearest_station_lines"] = [
        stations.iloc[index]["lines"] for index in indexes
    ]
    return result


def _ordered_test_frame(
    test: pd.DataFrame, origins: pd.DataFrame, destination_ids: list[str]
) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [origins["origin_id"].astype(str), destination_ids],
        names=["origin_id", "destination_id"],
    )
    ordered = test.set_index(["origin_id", "destination_id"]).reindex(index)
    if ordered["target_travel_time_seconds"].isna().any():
        raise ValueError("Test data is not complete for atlas evaluation.")
    return ordered.reset_index()


def _evaluate_candidate_atlas(
    model_atlas: np.ndarray,
    anchors: pd.DataFrame,
    test: pd.DataFrame,
    destinations: pd.DataFrame,
    selected,
    neighbours: int,
    discontinuity_threshold_minutes: float,
) -> dict:
    test_origins = (
        test[["origin_id", "origin_lat", "origin_lng"]]
        .drop_duplicates("origin_id")
        .rename(columns={"origin_lat": "lat", "origin_lng": "lng"})
    )
    destination_ids = destinations["destination_id"].astype(str).tolist()
    ordered = _ordered_test_frame(test, test_origins, destination_ids)
    interpolated_minutes = _interpolate(
        model_atlas.astype(np.float32) * ATLAS_STEP_MINUTES,
        anchors,
        test_origins,
        neighbours=neighbours,
        discontinuity_threshold_minutes=discontinuity_threshold_minutes,
    )
    atlas_seconds = interpolated_minutes.reshape(-1) * 60
    direct_seconds = predict(selected, ordered)
    direct_error_frame = ordered.copy()
    direct_error_frame["target_travel_time_seconds"] = direct_seconds
    direct_band_metrics = {}
    for focus, mask in {
        "central": ordered["grid_band"] == "Zone 1 core",
        "inner": ordered["coverage_region_destination"] == "original",
        "wide": pd.Series(True, index=ordered.index),
        "outer": ordered["coverage_region_destination"] == "outer",
    }.items():
        direct_band_metrics[focus] = _regression_report(
            direct_error_frame.loc[mask], atlas_seconds[mask.to_numpy()]
        )
    return {
        "atlas_vs_traveltime": {
            **_regression_report(ordered, atlas_seconds),
            "slices": _slice_reports(ordered, atlas_seconds),
        },
        "atlas_vs_direct": {
            **_regression_report(direct_error_frame, atlas_seconds),
            "slices": _slice_reports(direct_error_frame, atlas_seconds),
            "coverage_bands": direct_band_metrics,
        },
    }


def _evaluate_production_atlas(test: pd.DataFrame) -> dict | None:
    metadata_path = project_path("frontend/public/model/atlas.json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = metadata_path.parent / metadata["model_file"]
    if not model_path.exists():
        return None
    cell_ids = [str(cell["destination_id"]) for cell in metadata["cells"]]
    original = test[
        (test["coverage_region_origin"] == "original")
        & (test["coverage_region_destination"] == "original")
        & test["destination_id"].astype(str).isin(cell_ids)
    ].copy()
    origins = (
        original[["origin_id", "origin_lat", "origin_lng"]]
        .drop_duplicates("origin_id")
        .rename(columns={"origin_lat": "lat", "origin_lng": "lng"})
    )
    ordered = _ordered_test_frame(original, origins, cell_ids)
    anchors = pd.DataFrame(metadata["origins"])
    values = np.memmap(
        model_path,
        dtype=np.uint8,
        mode="r",
        shape=(int(metadata["origin_count"]), int(metadata["cell_count"])),
    )
    predicted = (
        _interpolate(
            np.asarray(values, dtype=np.float32)
            * float(metadata["quantisation_step_minutes"]),
            anchors,
            origins,
            neighbours=int(metadata["interpolation_neighbours"]),
        ).reshape(-1)
        * 60
    )
    return _regression_report(ordered, predicted)


def export_expanded_atlas(
    params_path: str = "params.yaml", *, batch_size: int = 128
) -> dict:
    started_at = time.perf_counter()
    params = load_params(params_path)
    run_id = str(params["expanded_run"]["id"])
    paths = run_paths(run_id)
    required = [
        paths["destinations"],
        paths["graph_nodes"],
        paths["graph_edges"],
        paths["bus_stops"],
        paths["graph_features"],
        paths["models"] / "selected.joblib",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Candidate atlas inputs are missing: {missing}")

    destinations = pd.read_parquet(paths["destinations"])
    nodes = pd.read_parquet(paths["graph_nodes"])
    edges = pd.read_parquet(paths["graph_edges"])
    access_nodes = pd.read_parquet(paths["bus_stops"])
    selected = load_bundle(paths["models"] / "selected.joblib")
    graph_baseline = load_bundle(paths["models"] / "graph_baseline.joblib")
    graph_params = params["transport_graph"]
    graph = build_transport_graph(
        nodes,
        edges,
        transfer_radius_m=float(graph_params["transfer_radius_m"]),
        transfer_penalty_seconds=float(graph_params["transfer_penalty_seconds"]),
        walking_speed_mps=float(graph_params["walking_speed_mps"]),
    )
    stations = _station_catalog(nodes)
    destinations = _attach_nearest_station(destinations, stations)
    base_anchors = expanded_atlas_origins(params)
    if len(base_anchors) != 765:
        raise ValueError(
            f"Expected 765 base atlas anchors, generated {len(base_anchors)}."
        )
    atlas_params = params["expanded_run"]["atlas"]
    neighbours = int(atlas_params["interpolation_neighbours"])
    discontinuity_threshold = float(atlas_params["discontinuity_threshold_minutes"])
    base_model_values, base_graph_values = _predict_anchor_surfaces(
        base_anchors,
        destinations,
        stations,
        graph,
        access_nodes,
        selected,
        graph_baseline,
        params,
        batch_size=batch_size,
    )
    base_anchors["surface_signature_minutes"] = np.median(base_model_values, axis=1)

    probes = adaptive_atlas_probe_origins(params)
    probe_model_values, probe_graph_values = _predict_anchor_surfaces(
        probes,
        destinations,
        stations,
        graph,
        access_nodes,
        selected,
        graph_baseline,
        params,
        batch_size=batch_size,
    )
    selection_mask = probes["probe_role"].eq("selection").to_numpy()
    validation_mask = probes["probe_role"].eq("validation").to_numpy()
    selection_probes = probes.loc[selection_mask].reset_index(drop=True)
    selection_direct = probe_model_values[selection_mask]
    selection_before = _interpolate(
        base_model_values.astype(np.float32),
        base_anchors,
        selection_probes,
        neighbours=neighbours,
        discontinuity_threshold_minutes=discontinuity_threshold,
    )
    adaptive_anchors, selected_probe_indexes = _select_adaptive_anchors(
        selection_probes,
        selection_direct,
        selection_before,
        count=int(atlas_params["adaptive_anchor_count"]),
        min_separation_m=float(atlas_params["adaptive_min_separation_m"]),
        min_mae_minutes=float(atlas_params["adaptive_min_mae_minutes"]),
    )
    adaptive_model_values = selection_direct[selected_probe_indexes]
    adaptive_graph_values = probe_graph_values[selection_mask][selected_probe_indexes]
    adaptive_scores = adaptive_anchors.pop("adaptive_probe_mae_minutes").to_numpy()
    candidate_anchors = pd.concat([base_anchors, adaptive_anchors], ignore_index=True)
    candidate_anchors["adaptive_probe_mae_minutes"] = None
    candidate_anchors.loc[len(base_anchors) :, "adaptive_probe_mae_minutes"] = (
        adaptive_scores
    )
    candidate_model_array = np.vstack([base_model_values, adaptive_model_values])
    candidate_graph_array = np.vstack([base_graph_values, adaptive_graph_values])
    candidate_anchors["surface_signature_minutes"] = np.median(
        candidate_model_array, axis=1
    )

    validation_probes = probes.loc[validation_mask].reset_index(drop=True)
    validation_direct = probe_model_values[validation_mask]
    validation_before = _interpolate(
        base_model_values.astype(np.float32),
        base_anchors,
        validation_probes,
        neighbours=neighbours,
        discontinuity_threshold_minutes=discontinuity_threshold,
    )
    candidate_validation_after = _interpolate(
        candidate_model_array.astype(np.float32),
        candidate_anchors,
        validation_probes,
        neighbours=neighbours,
        discontinuity_threshold_minutes=discontinuity_threshold,
    )
    validation_before_report = _probe_error_report(validation_direct, validation_before)
    candidate_validation_report = _probe_error_report(
        validation_direct, candidate_validation_after
    )
    adaptive_anchors_accepted = _adaptive_anchors_pass_validation(
        validation_before_report, candidate_validation_report
    )
    if adaptive_anchors_accepted:
        anchors = candidate_anchors
        model_array = candidate_model_array
        graph_array = candidate_graph_array
        validation_after = candidate_validation_after
    else:
        anchors = base_anchors
        model_array = base_model_values
        graph_array = base_graph_values
        validation_after = validation_before
    if "adaptive_probe_mae_minutes" not in anchors:
        anchors["adaptive_probe_mae_minutes"] = None
    selection_after = _interpolate(
        model_array.astype(np.float32),
        anchors,
        selection_probes,
        neighbours=neighbours,
        discontinuity_threshold_minutes=discontinuity_threshold,
    )
    probe_evaluation = {
        "method": "disjoint deterministic model-only outer probes",
        "travel_time_labels_used": False,
        "attempted_anchor_count": int(len(adaptive_anchors)),
        "selected_anchor_count": (
            int(len(adaptive_anchors)) if adaptive_anchors_accepted else 0
        ),
        "adaptive_anchors_accepted": adaptive_anchors_accepted,
        "acceptance_rule": "validation MAE improves without worsening validation p90",
        "candidate_validation": candidate_validation_report,
        "selection": {
            "before": _probe_error_report(selection_direct, selection_before),
            "after": _probe_error_report(selection_direct, selection_after),
        },
        "validation": {
            "before": validation_before_report,
            "after": _probe_error_report(validation_direct, validation_after),
        },
    }

    output_dir = paths["artifacts"] / "atlas"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_temporary = output_dir / "model.tmp.u8"
    graph_temporary = output_dir / "graph.tmp.u8"
    shape = (len(anchors), len(destinations))
    model_values = np.memmap(model_temporary, dtype=np.uint8, mode="w+", shape=shape)
    graph_values = np.memmap(graph_temporary, dtype=np.uint8, mode="w+", shape=shape)
    model_values[:] = model_array
    graph_values[:] = graph_array
    model_values.flush()
    graph_values.flush()
    del model_values
    del graph_values
    model_path = output_dir / "model.u8"
    graph_path = output_dir / "graph.u8"
    model_temporary.replace(model_path)
    graph_temporary.replace(graph_path)
    model_values = np.memmap(model_path, dtype=np.uint8, mode="r", shape=shape)

    test = pd.read_parquet(paths["graph_features"], filters=[("split", "==", "test")])
    evaluation = _evaluate_candidate_atlas(
        model_values,
        anchors,
        test,
        destinations,
        selected,
        neighbours,
        discontinuity_threshold,
    )
    evaluation["adaptive_probe_validation"] = probe_evaluation
    evaluation["production_atlas_vs_traveltime_original_subset"] = (
        _evaluate_production_atlas(test)
    )
    evaluation["atlas_export_duration_seconds"] = time.perf_counter() - started_at
    _atomic_json(output_dir / "evaluation.json", evaluation)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    bounds = params["expanded_run"]["expanded_bounds"]
    metadata = {
        "version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "focus": "wide",
        "detail": "fine",
        "layout": "origin-major",
        "quantisation_step_minutes": ATLAS_STEP_MINUTES,
        "origin_count": len(anchors),
        "cell_count": len(destinations),
        "origin_bounds": bounds,
        "coverage_bounds": bounds,
        "band_ownership": {
            "central": {"grid_bands": ["Zone 1 core"]},
            "inner": {"coverage_regions": ["original"]},
            "wide": {"coverage_regions": ["original", "outer"]},
        },
        "anchor_density": {
            "central": {
                "count": int((anchors["anchor_region"] == "central").sum()),
                "lat_count": int(params["expanded_run"]["atlas"]["inner_lat_count"]),
                "lng_count": int(params["expanded_run"]["atlas"]["inner_lng_count"]),
            },
            "outer": {
                "count": int((anchors["anchor_region"] == "outer").sum()),
                "base_grid_count": int(
                    (
                        (anchors["anchor_region"] == "outer")
                        & (anchors["anchor_source"] == "grid")
                    ).sum()
                ),
                "adaptive_count": int(
                    (anchors["anchor_source"] == "adaptive_probe").sum()
                ),
                "spacing_multiplier": float(
                    params["expanded_run"]["atlas"]["outer_spacing_multiplier"]
                ),
            },
        },
        "interpolation_neighbours": neighbours,
        "discontinuity_threshold_minutes": discontinuity_threshold,
        "interpolation_metrics": evaluation["atlas_vs_direct"]["coverage_bands"],
        "adaptive_probe_validation": probe_evaluation,
        "model_type": selected.model_type,
        "model_lineage": manifest["model_lineage"],
        "dataset_sha256": manifest["validation"]["graph_feature_sha256"],
        "graph_nodes_sha256": _file_sha256(paths["graph_nodes"]),
        "graph_edges_sha256": _file_sha256(paths["graph_edges"]),
        "bus_stops_sha256": _file_sha256(paths["bus_stops"]),
        "source_model_sha256": _file_sha256(paths["models"] / "selected.joblib"),
        "model_file": model_path.name,
        "graph_file": graph_path.name,
        "model_file_sha256": _file_sha256(model_path),
        "graph_file_sha256": _file_sha256(graph_path),
        "origins": (
            anchors[
                [
                    "origin_id",
                    "lat",
                    "lng",
                    "anchor_region",
                    "anchor_source",
                    "lat_index",
                    "lng_index",
                    "surface_signature_minutes",
                    "adaptive_probe_mae_minutes",
                ]
            ]
            .astype(object)
            .where(
                pd.notna(
                    anchors[
                        [
                            "origin_id",
                            "lat",
                            "lng",
                            "anchor_region",
                            "anchor_source",
                            "lat_index",
                            "lng_index",
                            "surface_signature_minutes",
                            "adaptive_probe_mae_minutes",
                        ]
                    ]
                ),
                None,
            )
            .to_dict(orient="records")
        ),
        "cells": _atlas_cell_records(destinations),
    }
    metadata_path = output_dir / "atlas.json"
    metadata_path.write_text(
        json.dumps(_json_value(metadata), separators=(",", ":")), encoding="utf-8"
    )
    manifest["candidate_atlas"] = {
        "atlas_params_sha256": _canonical_sha256(atlas_params),
        "metadata": _artifact_record(metadata_path),
        "model": _artifact_record(model_path),
        "graph": _artifact_record(graph_path),
        "evaluation": _artifact_record(output_dir / "evaluation.json"),
        "promoted": False,
    }
    _atomic_json(paths["manifest"], manifest)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = export_expanded_atlas(args.params, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
