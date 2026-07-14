"""Export a compact, browser-readable atlas of offline model surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from equidistant_ml.surfaces.config import project_path
from equidistant_ml.surfaces.geo import haversine_m
from equidistant_ml.surfaces.predict import (
    _load_best_available_model,
    _predict_surfaces_for_origins,
    _prediction_destination_grid,
    _prediction_params,
    _prediction_stations,
)

ATLAS_STEP_MINUTES = 0.5
ATLAS_MAX_VALUE = 255


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _atlas_origins(
    *,
    south: float,
    north: float,
    west: float,
    east: float,
    lat_count: int,
    lng_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "origin_id": f"atlas_{lat_index:02}_{lng_index:02}",
                "lat": float(lat),
                "lng": float(lng),
                "lat_index": lat_index,
                "lng_index": lng_index,
            }
            for lat_index, lat in enumerate(np.linspace(south, north, lat_count))
            for lng_index, lng in enumerate(np.linspace(west, east, lng_count))
        ]
    )


def _quantise(values: np.ndarray) -> np.ndarray:
    return (
        np.rint(values / ATLAS_STEP_MINUTES).clip(0, ATLAS_MAX_VALUE).astype(np.uint8)
    )


def _surface_matrix(
    frame: pd.DataFrame,
    origins: pd.DataFrame,
    destination_ids: list[str],
    column: str,
) -> np.ndarray:
    index = pd.MultiIndex.from_product(
        [origins["origin_id"].astype(str), destination_ids],
        names=["origin_id", "destination_id"],
    )
    values = frame.set_index(["origin_id", "destination_id"])[column].reindex(index)
    if values.isna().any():
        raise ValueError(f"Atlas export produced missing values for {column}.")
    return values.to_numpy(dtype=np.float32).reshape(len(origins), len(destination_ids))


def _interpolate(
    atlas: np.ndarray,
    anchors: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    neighbours: int,
) -> np.ndarray:
    result = np.empty((len(holdout), atlas.shape[1]), dtype=np.float32)
    anchor_lats = anchors["lat"].to_numpy()
    anchor_lngs = anchors["lng"].to_numpy()
    for row_index, row in enumerate(holdout.itertuples()):
        distances = haversine_m(row.lat, row.lng, anchor_lats, anchor_lngs)
        selected = np.argpartition(distances, neighbours - 1)[:neighbours]
        selected_distances = np.maximum(distances[selected], 40.0)
        weights = 1.0 / np.square(selected_distances)
        weights /= weights.sum()
        result[row_index] = np.sum(atlas[selected] * weights[:, None], axis=0)
    return result


def export_atlas(args: argparse.Namespace) -> None:
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = _prediction_params()
    destinations = _prediction_destination_grid("h3", "inner", "fine", None, None)
    stations = _prediction_stations(params)
    nearest_station_indexes = [
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
        for row in destinations.itertuples()
    ]
    destinations = destinations.copy()
    destinations["nearest_station_name"] = [
        stations.iloc[index]["station_name"] for index in nearest_station_indexes
    ]
    destinations["nearest_station_lines"] = [
        stations.iloc[index]["lines"] for index in nearest_station_indexes
    ]
    origins = _atlas_origins(
        south=args.south,
        north=args.north,
        west=args.west,
        east=args.east,
        lat_count=args.lat_count,
        lng_count=args.lng_count,
    )
    destination_ids = destinations["destination_id"].astype(str).tolist()
    model_path = output_dir / "model.u8"
    graph_path = output_dir / "graph.u8"
    reuse_binaries = args.reuse_binaries and model_path.exists() and graph_path.exists()
    model_atlas = np.memmap(
        model_path,
        dtype=np.uint8,
        mode="r+" if reuse_binaries else "w+",
        shape=(len(origins), len(destinations)),
    )
    graph_atlas = np.memmap(
        graph_path,
        dtype=np.uint8,
        mode="r+" if reuse_binaries else "w+",
        shape=(len(origins), len(destinations)),
    )

    if not reuse_binaries:
        for start in range(0, len(origins), args.batch_size):
            batch = origins.iloc[start : start + args.batch_size]
            predicted = _predict_surfaces_for_origins(
                batch[["origin_id", "lat", "lng"]],
                destinations,
                params,
                stations,
                grid_mode="h3",
            )
            model_values = _surface_matrix(
                predicted,
                batch,
                destination_ids,
                "travel_time_minutes",
            )
            graph_values = _surface_matrix(
                predicted,
                batch,
                destination_ids,
                "graph_score_minutes",
            )
            model_atlas[start : start + len(batch)] = _quantise(model_values)
            graph_atlas[start : start + len(batch)] = _quantise(graph_values)
            model_atlas.flush()
            graph_atlas.flush()
            print(
                f"Exported {start + len(batch)}/{len(origins)} atlas origins",
                flush=True,
            )

    validation = None
    if args.validation_origins:
        rng = np.random.default_rng(args.seed)
        holdout = pd.DataFrame(
            [
                {
                    "origin_id": f"validation_{index:02}",
                    "lat": float(rng.uniform(args.south, args.north)),
                    "lng": float(rng.uniform(args.west, args.east)),
                }
                for index in range(args.validation_origins)
            ]
        )
        direct = _predict_surfaces_for_origins(
            holdout,
            destinations,
            params,
            stations,
            grid_mode="h3",
        )
        direct_values = _surface_matrix(
            direct,
            holdout,
            destination_ids,
            "travel_time_minutes",
        )
        decoded_model = np.asarray(model_atlas, dtype=np.float32) * ATLAS_STEP_MINUTES
        interpolated = _interpolate(
            decoded_model,
            origins,
            holdout,
            neighbours=args.neighbours,
        )
        errors = np.abs(interpolated - direct_values)
        validation = {
            "origin_count": len(holdout),
            "neighbours": args.neighbours,
            "mae_minutes_vs_direct_model": float(errors.mean()),
            "median_abs_error_minutes_vs_direct_model": float(np.median(errors)),
            "p90_abs_error_minutes_vs_direct_model": float(np.quantile(errors, 0.9)),
            "within_1_min_pct_vs_direct_model": float((errors <= 1).mean() * 100),
            "within_2_min_pct_vs_direct_model": float((errors <= 2).mean() * 100),
        }

    model_bundle = _load_best_available_model("h3")
    source_model_path = project_path("models/travel_time_hillclimb_best.joblib")
    metadata = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "focus": "inner",
        "detail": "fine",
        "quantisation_step_minutes": ATLAS_STEP_MINUTES,
        "layout": "origin-major",
        "origin_count": len(origins),
        "cell_count": len(destinations),
        "origin_bounds": {
            "south": args.south,
            "north": args.north,
            "west": args.west,
            "east": args.east,
        },
        "origin_grid": {"lat_count": args.lat_count, "lng_count": args.lng_count},
        "interpolation_neighbours": args.neighbours,
        "model_type": model_bundle.model_type if model_bundle else "heuristic",
        "source_model_sha256": (
            _sha256(source_model_path) if source_model_path.exists() else None
        ),
        "model_file": model_path.name,
        "graph_file": graph_path.name,
        "model_file_sha256": _sha256(model_path),
        "graph_file_sha256": _sha256(graph_path),
        "interpolation_validation": validation,
        "origins": [
            {
                key: _json_value(row[key])
                for key in ["origin_id", "lat", "lng", "lat_index", "lng_index"]
            }
            for _, row in origins.iterrows()
        ],
        "cells": [
            {
                key: _json_value(row[key])
                for key in [
                    "destination_id",
                    "lat",
                    "lng",
                    "boundary",
                    "h3_cell",
                    "h3_resolution",
                    "grid_band",
                    "grid_priority",
                    "cell_area_km2",
                    "nearest_station_name",
                    "nearest_station_lines",
                ]
                if key in row
            }
            for _, row in destinations.iterrows()
        ],
    }
    (output_dir / "atlas.json").write_text(
        json.dumps(metadata, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(validation or {}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="frontend/public/model")
    parser.add_argument("--south", type=float, default=51.46)
    parser.add_argument("--north", type=float, default=51.56)
    parser.add_argument("--west", type=float, default=-0.245)
    parser.add_argument("--east", type=float, default=0.03)
    parser.add_argument("--lat-count", type=int, default=20)
    parser.add_argument("--lng-count", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--neighbours", type=int, default=4)
    parser.add_argument("--validation-origins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--reuse-binaries", action="store_true")
    return parser


if __name__ == "__main__":
    export_atlas(build_parser().parse_args())
