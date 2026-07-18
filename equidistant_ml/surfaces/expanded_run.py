"""Isolated Harrow-to-Sidcup data, training, and evaluation workflow."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess  # nosec B404
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import h3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from equidistant_ml.surfaces.config import load_params, project_path
from equidistant_ml.surfaces.geo import BBox, safe_feature_name
from equidistant_ml.surfaces.grid import build_hierarchical_expanded_grid
from equidistant_ml.surfaces.models import (
    ModelBundle,
    load_bundle,
    predict,
    save_bundle,
    train_baseline,
    train_graph_baseline,
    train_graph_residual_model,
    train_lightgbm,
)

MANIFEST_SCHEMA = 1
CHECKPOINT_SCHEMA = 2
SPLITS = ("train", "tune", "test")
REGIONS = ("original", "outer")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _bbox(values: dict[str, Any]) -> BBox:
    return BBox(
        north=float(values["north"]),
        south=float(values["south"]),
        west=float(values["west"]),
        east=float(values["east"]),
    )


def _point_in_bbox(lat: float, lng: float, bbox: BBox) -> bool:
    return bbox.south <= lat <= bbox.north and bbox.west <= lng <= bbox.east


def _region_for_point(lat: float, lng: float, original: BBox) -> str:
    return "original" if _point_in_bbox(lat, lng, original) else "outer"


def _polygon(bbox: BBox) -> h3.LatLngPoly:
    return h3.LatLngPoly(
        [
            (bbox.south, bbox.west),
            (bbox.south, bbox.east),
            (bbox.north, bbox.east),
            (bbox.north, bbox.west),
        ]
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _training_params_sha256(params: dict[str, Any]) -> str:
    """Hash run inputs while allowing atlas-only experiments to evolve."""
    expanded_run = {
        key: value for key, value in params["expanded_run"].items() if key != "atlas"
    }
    scoped = {**params, "expanded_run": expanded_run}
    return _canonical_sha256(scoped)


def _git_commit() -> str:
    result = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        cwd=project_path("."),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_worktree_state() -> dict[str, Any]:
    status = subprocess.run(  # nosec B603 B607
        ["git", "status", "--porcelain=v1"],
        cwd=project_path("."),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(  # nosec B603 B607
        ["git", "diff", "--binary", "HEAD"],
        cwd=project_path("."),
        check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=project_path("."),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "dirty": bool(status.strip()),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": {
            path: _file_sha256(project_path(path)) for path in sorted(untracked)
        },
    }


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def run_paths(run_id: str) -> dict[str, Path]:
    data = project_path(Path("data/runs") / run_id)
    models = project_path(Path("models/runs") / run_id)
    metrics = project_path(Path("metrics/runs") / run_id)
    artifacts = project_path(Path("artifacts/runs") / run_id)
    return {
        "data": data,
        "destinations": data / "destinations.parquet",
        "origins": data / "origins.parquet",
        "labels": data / "labels.parquet",
        "checkpoints": data / "label_parts",
        "features": data / "features.parquet",
        "graph_features": data / "graph_features.parquet",
        "reference_nodes": data / "graph/reference_nodes.parquet",
        "reference_edges": data / "graph/reference_edges.parquet",
        "graph_nodes": data / "graph/nodes.parquet",
        "graph_edges": data / "graph/edges.parquet",
        "bus_stops": data / "graph/bus_stops.parquet",
        "manifest": data / "manifest.json",
        "validation_record": data / "validation.json",
        "models": models,
        "metrics": metrics / "model.json",
        "artifacts": artifacts,
    }


def expanded_destination_grid(params: dict[str, Any]) -> pd.DataFrame:
    prediction = params["prediction_grid"]
    bands = prediction["bands"]
    fine = [
        dict(id=band_id, **bands[band_id])
        for band_id in ("zone1_core", "zone2_3_inner")
    ]
    expanded = dict(id="expanded_ring", **bands["expanded_ring"])
    return build_hierarchical_expanded_grid(fine, expanded)


def _normalised_station_name(value: str) -> str:
    return safe_feature_name(value.replace("station", ""))


def validate_graph_coverage(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    required_stations: Iterable[str],
) -> dict[str, Any]:
    if nodes.empty or edges.empty:
        raise ValueError("Expanded runs require non-empty versioned graph artifacts.")
    node_names = nodes["name"].fillna("").astype(str).map(_normalised_station_name)
    required_nodes: dict[str, list[str]] = {}
    for required in required_stations:
        target = _normalised_station_name(required)
        matches = nodes.loc[
            node_names.map(lambda value: target in value or value in target), "node_id"
        ].astype(str)
        if matches.empty:
            raise ValueError(f"Required graph station is missing: {required}")
        required_nodes[str(required)] = matches.tolist()

    modes = set(nodes["mode"].fillna("").astype(str)) | set(
        edges["mode"].fillna("").astype(str)
    )
    if "national_rail" not in modes:
        raise ValueError("Expanded graph has no national_rail topology.")

    adjacency: dict[str, set[str]] = {}
    for row in edges.itertuples(index=False):
        adjacency.setdefault(str(row.from_node_id), set()).add(str(row.to_node_id))
    first_name, second_name = list(required_nodes)[:2]
    targets = set(required_nodes[second_name])
    queue = deque(required_nodes[first_name])
    visited = set(queue)
    connected = False
    while queue:
        current = queue.popleft()
        if current in targets:
            connected = True
            break
        for neighbour in adjacency.get(current, set()):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    if not connected:
        raise ValueError(
            f"No graph path connects {first_name} to {second_name}; refusing run."
        )

    line_text = " ".join(nodes["lines"].fillna("").astype(str)).lower()
    return {
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "modes": sorted(modes),
        "required_station_node_ids": required_nodes,
        "required_stations_connected": connected,
        "connection_catalogue": {
            name: name in line_text
            for name in ("southeastern", "chiltern", "metropolitan")
        },
    }


def _assign_split_blocks(
    expanded: BBox,
    original: BBox,
    resolution: int,
    sampling: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, str]:
    blocks = sorted(
        h3.h3shape_to_cells_experimental(
            _polygon(expanded), resolution, contain="overlap"
        )
    )
    by_region: dict[str, list[str]] = {region: [] for region in REGIONS}
    for block in blocks:
        lat, lng = h3.cell_to_latlng(block)
        by_region[_region_for_point(float(lat), float(lng), original)].append(block)

    assignment: dict[str, str] = {}
    for region, region_blocks in by_region.items():
        shuffled = list(region_blocks)
        rng.shuffle(shuffled)
        weights = np.array(
            [float(sampling[split][region]) for split in SPLITS], dtype=float
        )
        raw = weights / weights.sum() * len(shuffled)
        counts = np.floor(raw).astype(int)
        counts = np.maximum(counts, 1)
        while counts.sum() > len(shuffled):
            index = int(np.argmax(counts - 1))
            counts[index] -= 1
        remainder_order = np.argsort(-(raw - np.floor(raw)))
        cursor = 0
        while counts.sum() < len(shuffled):
            counts[int(remainder_order[cursor % len(counts)])] += 1
            cursor += 1
        cursor = 0
        for split, count in zip(SPLITS, counts):
            for block in shuffled[cursor : cursor + int(count)]:
                assignment[block] = split
            cursor += int(count)
    return assignment


def _sample_region_origins(
    *,
    count: int,
    station_fraction: float,
    split: str,
    region: str,
    expanded: BBox,
    original: BBox,
    block_resolution: int,
    block_assignment: dict[str, str],
    stations: pd.DataFrame,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    station_count = int(round(count * station_fraction))
    strategy_counts = {
        "station_jitter": station_count,
        "uniform": count - station_count,
    }
    station_pool = stations.copy()
    station_pool["origin_block"] = [
        h3.latlng_to_cell(float(row.lat), float(row.lng), block_resolution)
        for row in station_pool.itertuples(index=False)
    ]
    station_pool["coverage_region"] = [
        _region_for_point(float(row.lat), float(row.lng), original)
        for row in station_pool.itertuples(index=False)
    ]
    station_pool = station_pool[
        (station_pool["coverage_region"] == region)
        & station_pool["origin_block"].map(block_assignment).eq(split)
        & station_pool.apply(
            lambda row: _point_in_bbox(float(row["lat"]), float(row["lng"]), expanded),
            axis=1,
        )
    ].reset_index(drop=True)
    if station_count and station_pool.empty:
        raise ValueError(f"No {region} station block is assigned to split {split}.")

    rows: list[dict[str, Any]] = []
    for strategy, target_count in strategy_counts.items():
        attempts = 0
        accepted = 0
        while accepted < target_count:
            attempts += 1
            if attempts > max(20_000, target_count * 2_000):
                raise RuntimeError(
                    "Could not sample "
                    f"{target_count} {strategy} {region}/{split} origins."
                )
            if strategy == "uniform":
                lat = float(rng.uniform(expanded.south, expanded.north))
                lng = float(rng.uniform(expanded.west, expanded.east))
            else:
                station = station_pool.iloc[int(rng.integers(0, len(station_pool)))]
                lat = float(station["lat"] + rng.normal(0, 0.006))
                lng = float(station["lng"] + rng.normal(0, 0.009))
            if not _point_in_bbox(lat, lng, expanded):
                continue
            if _region_for_point(lat, lng, original) != region:
                continue
            block = h3.latlng_to_cell(lat, lng, block_resolution)
            if block_assignment.get(block) != split:
                continue
            rows.append(
                {
                    "lat": round(lat, 7),
                    "lng": round(lng, 7),
                    "split": split,
                    "coverage_region": region,
                    "origin_block": block,
                    "sample_strategy": strategy,
                }
            )
            accepted += 1
    return rows


def sample_expanded_origins(
    params: dict[str, Any], graph_nodes: pd.DataFrame
) -> pd.DataFrame:
    run = params["expanded_run"]
    expanded = _bbox(run["expanded_bounds"])
    original = _bbox(run["original_bounds"])
    resolution = int(run["split_h3_resolution"])
    rng = np.random.default_rng(int(run["seed"]))
    assignment = _assign_split_blocks(
        expanded, original, resolution, run["sampling"], rng
    )
    stations = graph_nodes[~graph_nodes["mode"].isin(["bus", "transfer"])].copy()
    stations = stations.drop_duplicates(["name", "lat", "lng"])
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_params = run["sampling"][split]
        for region in REGIONS:
            rows.extend(
                _sample_region_origins(
                    count=int(split_params[region]),
                    station_fraction=float(split_params[f"{region}_station_fraction"]),
                    split=split,
                    region=region,
                    expanded=expanded,
                    original=original,
                    block_resolution=resolution,
                    block_assignment=assignment,
                    stations=stations,
                    rng=rng,
                )
            )
    origins = pd.DataFrame(rows)
    origins.insert(
        0,
        "origin_id",
        [
            f"{run['id']}_{row.split}_{index:04d}"
            for index, row in enumerate(origins.itertuples())
        ],
    )
    return origins


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(project_path("."))),
        "sha256": _file_sha256(path),
    }


def prepare_run(params_path: str = "params.yaml") -> dict[str, Any]:
    params = load_params(params_path)
    run = params["expanded_run"]
    run_id = str(run["id"])
    paths = run_paths(run_id)
    if not paths["graph_nodes"].exists() or not paths["graph_edges"].exists():
        raise FileNotFoundError(
            "Expanded graph artifacts are required before origin sampling: "
            f"{paths['graph_nodes']} and {paths['graph_edges']}"
        )
    if not paths["bus_stops"].exists():
        raise FileNotFoundError(
            "Expanded runs require a versioned non-routable bus-stop artifact: "
            f"{paths['bus_stops']}"
        )
    nodes = pd.read_parquet(paths["graph_nodes"])
    edges = pd.read_parquet(paths["graph_edges"])
    graph_validation = validate_graph_coverage(
        nodes, edges, run["required_graph_stations"]
    )
    destinations = expanded_destination_grid(params)
    origins = sample_expanded_origins(params, nodes)

    expected_origins = sum(
        int(run["sampling"][split][region]) for split in SPLITS for region in REGIONS
    )
    if len(origins) != expected_origins:
        raise ValueError(
            f"Expected {expected_origins} origins, generated {len(origins)}."
        )
    if int((destinations["coverage_region"] == "original").sum()) != 3032:
        raise ValueError("The expanded grid did not preserve the 3,032 original cells.")

    paths["data"].mkdir(parents=True, exist_ok=True)
    destinations.to_parquet(paths["destinations"], index=False)
    origins.to_parquet(paths["origins"], index=False)
    params_file = project_path(params_path)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_id,
        "created_at": _utc_now(),
        "git_commit": _git_commit(),
        "git_worktree": _git_worktree_state(),
        "params": _artifact_record(params_file),
        "training_params_sha256": _training_params_sha256(params),
        "api_definition": params["traveltime"],
        "graph": {
            "nodes": _artifact_record(paths["graph_nodes"]),
            "edges": _artifact_record(paths["graph_edges"]),
            "bus_stops": _artifact_record(paths["bus_stops"]),
            "validation": graph_validation,
        },
        "splits": {
            split: {
                region: int(
                    (
                        (origins["split"] == split)
                        & (origins["coverage_region"] == region)
                    ).sum()
                )
                for region in REGIONS
            }
            for split in SPLITS
        },
        "counts": {
            "origins": int(len(origins)),
            "destinations": int(len(destinations)),
            "original_destinations": int(
                (destinations["coverage_region"] == "original").sum()
            ),
            "outer_destinations": int(
                (destinations["coverage_region"] == "outer").sum()
            ),
            "expected_label_rows": int(len(origins) * len(destinations)),
        },
        "artifacts": {
            "origins": _artifact_record(paths["origins"]),
            "destinations": _artifact_record(paths["destinations"]),
        },
        "model_lineage": {"incumbent_dependency": False, "fresh_labels_only": True},
    }
    _atomic_json(paths["manifest"], manifest)
    return manifest


def _load_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    if not paths["manifest"].exists():
        raise FileNotFoundError("Run manifest is missing; run prepare first.")
    return json.loads(paths["manifest"].read_text(encoding="utf-8"))


def record_operation(
    name: str, seconds: float, params_path: str = "params.yaml"
) -> dict[str, float]:
    params = load_params(params_path)
    paths = run_paths(str(params["expanded_run"]["id"]))
    manifest = _load_manifest(paths)
    manifest["git_commit"] = _git_commit()
    manifest["git_worktree"] = _git_worktree_state()
    operations = manifest.setdefault("operational_metrics_seconds", {})
    operations[str(name)] = float(seconds)
    _atomic_json(paths["manifest"], manifest)
    return operations


def _assert_artifact_hash(
    manifest: dict[str, Any], key: str, path: Path, *, section: str = "artifacts"
) -> None:
    recorded = manifest[section][key]["sha256"]
    actual = _file_sha256(path)
    if actual != recorded:
        raise ValueError(f"{key} hash mismatch: expected {recorded}, got {actual}.")


def _assert_pair_frame(
    frame: pd.DataFrame,
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    name: str,
) -> None:
    expected = len(origins) * len(destinations)
    if len(frame) != expected:
        raise ValueError(f"{name} has {len(frame):,} rows; expected {expected:,}.")
    if frame.duplicated(["origin_id", "destination_id"]).any():
        raise ValueError(f"{name} contains duplicate origin/destination pairs.")
    if set(frame["origin_id"].astype(str)) != set(origins["origin_id"].astype(str)):
        raise ValueError(f"{name} origin IDs do not match the run.")
    if set(frame["destination_id"].astype(str)) != set(
        destinations["destination_id"].astype(str)
    ):
        raise ValueError(f"{name} destination IDs do not match the run.")


def validate_run(
    params_path: str = "params.yaml", *, write_record: bool = False
) -> dict[str, Any]:
    params = load_params(params_path)
    run_id = str(params["expanded_run"]["id"])
    paths = run_paths(run_id)
    manifest = _load_manifest(paths)
    if manifest["run_id"] != run_id:
        raise ValueError("Manifest run ID does not match params.yaml.")
    recorded_training_params = manifest.get("training_params_sha256")
    if recorded_training_params is None:
        params_match = manifest["params"]["sha256"] == _file_sha256(
            project_path(params_path)
        )
    else:
        params_match = recorded_training_params == _training_params_sha256(params)
    if not params_match:
        raise ValueError(
            "Training or data parameters changed after run preparation; "
            "prepare a new run ID."
        )
    _assert_artifact_hash(manifest, "origins", paths["origins"])
    _assert_artifact_hash(manifest, "destinations", paths["destinations"])
    _assert_artifact_hash(manifest, "nodes", paths["graph_nodes"], section="graph")
    _assert_artifact_hash(manifest, "edges", paths["graph_edges"], section="graph")
    _assert_artifact_hash(manifest, "bus_stops", paths["bus_stops"], section="graph")

    origins = pd.read_parquet(paths["origins"])
    destinations = pd.read_parquet(paths["destinations"])
    blocks = {
        split: set(origins.loc[origins["split"] == split, "origin_block"].astype(str))
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = blocks[left] & blocks[right]
            if overlap:
                raise ValueError(
                    f"Origin split blocks overlap for {left}/{right}: "
                    f"{sorted(overlap)[:3]}"
                )

    checkpoint_files = sorted(paths["checkpoints"].glob("*.parquet"))
    if len(checkpoint_files) != len(origins):
        raise ValueError(
            f"Found {len(checkpoint_files)} checkpoint shards; expected {len(origins)}."
        )
    destination_ids = destinations["destination_id"].astype(str).tolist()
    for checkpoint_path in checkpoint_files:
        checkpoint = pd.read_parquet(checkpoint_path)
        if "_checkpoint_fingerprint" not in checkpoint:
            raise ValueError(
                f"Checkpoint has no request fingerprint: {checkpoint_path}"
            )
        if checkpoint["_checkpoint_fingerprint"].nunique() != 1:
            raise ValueError(
                f"Checkpoint has inconsistent fingerprints: {checkpoint_path}"
            )
        if set(checkpoint["_checkpoint_schema"].astype(int)) != {CHECKPOINT_SCHEMA}:
            raise ValueError(f"Checkpoint schema mismatch: {checkpoint_path}")
        if checkpoint["destination_id"].astype(str).tolist() != destination_ids:
            raise ValueError(
                f"Checkpoint is incomplete or reordered: {checkpoint_path}"
            )

    labels = pd.read_parquet(paths["labels"])
    _assert_pair_frame(labels, origins, destinations, "labels")
    reachable = labels["reachable"].astype(bool)
    travel = pd.to_numeric(labels["travel_time_seconds"], errors="coerce")
    target = pd.to_numeric(labels["target_travel_time_seconds"], errors="coerce")
    cap = int(params["traveltime"]["travel_time_seconds"]) + int(
        params["traveltime"]["unreachable_penalty_seconds"]
    )
    if travel[reachable].isna().any() or not np.allclose(
        travel[reachable], target[reachable]
    ):
        raise ValueError("Reachable labels have inconsistent travel-time targets.")
    if travel[~reachable].notna().any() or not np.allclose(target[~reachable], cap):
        raise ValueError("Unreachable labels have inconsistent capped targets.")

    feature_schema: dict[str, list[str]] = {}
    for key in ("features", "graph_features"):
        frame = pd.read_parquet(paths[key])
        _assert_pair_frame(frame, origins, destinations, key)
        required = [
            column
            for column in frame.columns
            if column not in {"travel_time_seconds"}
            and not pd.api.types.is_object_dtype(frame[column])
        ]
        null_columns = [column for column in required if frame[column].isna().any()]
        if null_columns:
            raise ValueError(
                f"{key} has incomplete numeric columns: {null_columns[:8]}"
            )
        feature_schema[key] = list(frame.columns)

    manifest["validated_at"] = _utc_now()
    manifest["validation"] = {
        "complete": True,
        "checkpoint_count": len(checkpoint_files),
        "row_count": int(len(labels)),
        "label_sha256": _file_sha256(paths["labels"]),
        "feature_sha256": _file_sha256(paths["features"]),
        "graph_feature_sha256": _file_sha256(paths["graph_features"]),
        "feature_schema_hash": _canonical_sha256(feature_schema),
    }
    manifest["artifacts"].update(
        {
            "labels": _artifact_record(paths["labels"]),
            "features": _artifact_record(paths["features"]),
            "graph_features": _artifact_record(paths["graph_features"]),
        }
    )
    _atomic_json(paths["manifest"], manifest)
    if write_record:
        _atomic_json(paths["validation_record"], manifest["validation"])
    return manifest["validation"]


def _regression_report(
    frame: pd.DataFrame, prediction: np.ndarray, *, bootstrap_seed: int = 42
) -> dict[str, Any]:
    truth = frame["target_travel_time_seconds"].to_numpy(dtype=float)
    error = prediction - truth
    absolute = np.abs(error)
    origin_mae = (
        pd.DataFrame({"origin_id": frame["origin_id"].astype(str), "error": absolute})
        .groupby("origin_id")["error"]
        .mean()
        .to_numpy()
    )
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.array(
        [
            rng.choice(origin_mae, size=len(origin_mae), replace=True).mean()
            for _ in range(2000)
        ]
    )
    return {
        "rows": int(len(frame)),
        "origins": int(frame["origin_id"].nunique()),
        "mae_seconds": float(absolute.mean()),
        "mae_minutes": float(absolute.mean() / 60),
        "origin_macro_mae_seconds": float(origin_mae.mean()),
        "origin_macro_mae_minutes": float(origin_mae.mean() / 60),
        "origin_bootstrap_mae_95ci_minutes": [
            float(np.quantile(bootstrap, 0.025) / 60),
            float(np.quantile(bootstrap, 0.975) / 60),
        ],
        "median_absolute_error_minutes": float(np.median(absolute) / 60),
        "rmse_minutes": float(np.sqrt(np.mean(np.square(error))) / 60),
        "p90_absolute_error_minutes": float(np.quantile(absolute, 0.90) / 60),
        "p95_absolute_error_minutes": float(np.quantile(absolute, 0.95) / 60),
        "signed_bias_minutes": float(error.mean() / 60),
        "within_5_minutes_pct": float((absolute <= 300).mean() * 100),
        "within_10_minutes_pct": float((absolute <= 600).mean() * 100),
    }


def _slice_reports(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    working = frame.copy()
    working["_prediction"] = prediction
    rail_access_column = (
        "destination_nearest_heavy_rail_distance_m"
        if "destination_nearest_heavy_rail_distance_m" in working
        else "destination_station_1_distance_m"
    )
    masks: dict[str, pd.Series] = {
        "full_rectangle": pd.Series(True, index=working.index),
        "expanded_ring_only": (working["coverage_region_destination"] == "outer"),
        "rail_access": working[rail_access_column] <= 700,
        "thames_crossing": (
            (working["origin_lat"] - 51.505) * (working["destination_lat"] - 51.505) < 0
        ),
        "station_jitter": working["sample_strategy"] == "station_jitter",
        "unreachable": ~working["reachable"].astype(bool),
    }
    for origin_region in REGIONS:
        for destination_region in REGIONS:
            masks[f"{origin_region}_to_{destination_region}"] = (
                working["coverage_region_origin"] == origin_region
            ) & (working["coverage_region_destination"] == destination_region)
    duration = pd.cut(
        working["target_travel_time_seconds"],
        [0, 1800, 3600, 5400, 7200, 12_600],
        labels=["0_30m", "30_60m", "60_90m", "90_120m", "120m_plus"],
        include_lowest=True,
    )
    for label in duration.dropna().unique():
        masks[f"duration_{label}"] = duration == label
    return {
        name: _regression_report(
            working.loc[mask], working.loc[mask, "_prediction"].to_numpy()
        )
        for name, mask in masks.items()
        if int(mask.sum()) >= 2
    }


def _reachability_report(
    bundle: ModelBundle, frame: pd.DataFrame
) -> dict[str, float] | None:
    if bundle.model_type not in {
        "lightgbm_two_stage",
        "hist_gradient_two_stage_fallback",
    }:
        return None
    probability = bundle.model["reachable_classifier"].predict_proba(
        frame[bundle.feature_columns]
    )[:, 1]
    threshold = float(bundle.model.get("reachable_probability_threshold", 0.05))
    predicted = probability >= threshold
    truth = frame["reachable"].astype(bool).to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
    }


def _evaluate_bundle(bundle: ModelBundle, frame: pd.DataFrame) -> dict[str, Any]:
    prediction = predict(bundle, frame)
    report = _regression_report(frame, prediction)
    report["slices"] = _slice_reports(frame, prediction)
    reachability = _reachability_report(bundle, frame)
    if reachability is not None:
        report["reachability"] = reachability
    return report


def train_and_evaluate(params_path: str = "params.yaml") -> dict[str, Any]:
    started_at = time.perf_counter()
    params = load_params(params_path)
    run_id = str(params["expanded_run"]["id"])
    paths = run_paths(run_id)
    validation = validate_run(params_path, write_record=False)

    base = pd.read_parquet(paths["features"])
    base_by_split = {split: base[base["split"] == split].copy() for split in SPLITS}
    del base
    gc.collect()
    if not all(len(base_by_split[split]) for split in SPLITS):
        raise ValueError("Every split must contain base features.")

    baseline = train_baseline(base_by_split["train"], params["baseline"])
    non_graph = train_lightgbm(base_by_split["train"], params["model"])
    model_bundles: dict[str, ModelBundle] = {
        "distance_station_ridge": baseline,
        "non_graph_lightgbm": non_graph,
    }
    tuning: dict[str, Any] = {}
    test: dict[str, Any] = {}
    for name, bundle in model_bundles.items():
        prediction = predict(bundle, base_by_split["tune"])
        tuning[name] = _regression_report(base_by_split["tune"], prediction)
        test[name] = _evaluate_bundle(bundle, base_by_split["test"])
    del base_by_split
    gc.collect()

    graph = pd.read_parquet(paths["graph_features"])
    graph_by_split = {split: graph[graph["split"] == split].copy() for split in SPLITS}
    del graph
    gc.collect()
    if not all(len(graph_by_split[split]) for split in SPLITS):
        raise ValueError("Every split must contain graph features.")
    graph_baseline = train_graph_baseline(
        graph_by_split["train"], params.get("graph_baseline", params["baseline"])
    )
    graph_default = train_graph_residual_model(
        graph_by_split["train"], graph_baseline, params["graph_model"]
    )
    graph_regularized = train_graph_residual_model(
        graph_by_split["train"],
        graph_baseline,
        params["expanded_run"]["regularized_graph_model"],
    )
    graph_bundles = {
        "graph_baseline": graph_baseline,
        "graph_residual_default": graph_default,
        "graph_residual_regularized": graph_regularized,
    }
    model_bundles.update(graph_bundles)
    for name, bundle in graph_bundles.items():
        prediction = predict(bundle, graph_by_split["tune"])
        tuning[name] = _regression_report(graph_by_split["tune"], prediction)
        test[name] = _evaluate_bundle(bundle, graph_by_split["test"])
    selected_name = min(
        tuning,
        key=lambda name: (
            tuning[name]["origin_macro_mae_seconds"],
            tuning[name]["p90_absolute_error_minutes"],
        ),
    )
    selected = model_bundles[selected_name]

    incumbent_path = project_path("models/travel_time_hillclimb_best.joblib")
    incumbent_diagnostic: dict[str, Any] | None = None
    if incumbent_path.exists():
        incumbent = load_bundle(incumbent_path)
        diagnostic = graph_by_split["test"].copy()
        missing = sorted(set(incumbent.feature_columns) - set(diagnostic.columns))
        for column in missing:
            diagnostic[column] = 0.0
        outer = diagnostic[diagnostic["coverage_region_origin"] == "outer"].copy()
        incumbent_diagnostic = {
            "warning": (
                "Diagnostic extrapolation only; missing legacy features are "
                "zero-filled."
            ),
            "zero_filled_features": missing,
            "metrics": _evaluate_bundle(incumbent, outer),
        }

    paths["models"].mkdir(parents=True, exist_ok=True)
    for name, bundle in model_bundles.items():
        save_bundle(bundle, paths["models"] / f"{name}.joblib")
    save_bundle(selected, paths["models"] / "selected.joblib")
    result = {
        "run_id": run_id,
        "generated_at": _utc_now(),
        "selection_rule": "lowest tuning-origin macro MAE; global p90 tie-breaker",
        "test_not_used_for_selection": True,
        "selected_model": selected_name,
        "selected_test": test[selected_name],
        "tuning": tuning,
        "test": test,
        "incumbent_outer_extrapolation": incumbent_diagnostic,
        "validation": validation,
        "training_and_evaluation_duration_seconds": time.perf_counter() - started_at,
    }
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(paths["metrics"], result)

    manifest = _load_manifest(paths)
    manifest["model_lineage"].update(
        {
            "selected_model": selected_name,
            "selection_split": "tune",
            "test_split_used_for_selection": False,
            "models": {
                path.stem: _artifact_record(path)
                for path in sorted(paths["models"].glob("*.joblib"))
            },
            "metrics": _artifact_record(paths["metrics"]),
        }
    )
    _atomic_json(paths["manifest"], manifest)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("validate")
    subparsers.add_parser("train-evaluate")
    record = subparsers.add_parser("record-operation")
    record.add_argument("--name", required=True)
    record.add_argument("--seconds", type=float, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_run(args.params)
    elif args.command == "validate":
        result = validate_run(args.params, write_record=True)
    elif args.command == "train-evaluate":
        result = train_and_evaluate(args.params)
    else:
        result = record_operation(args.name, args.seconds, args.params)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
