"""Measure production and expanded-candidate non-functional characteristics."""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from equidistant_ml.surfaces.config import load_params, project_path
from equidistant_ml.surfaces.expanded_atlas import _feature_batch, _station_catalog
from equidistant_ml.surfaces.expanded_run import _atomic_json, run_paths
from equidistant_ml.surfaces.models import load_bundle, predict
from equidistant_ml.surfaces.predict import (
    predict_group_surface,
    surface_to_grid_response,
)
from equidistant_ml.surfaces.transport_graph import build_transport_graph

PARTICIPANTS = [
    {"lat": 51.551808, "lng": -0.195603, "name": "Northwest"},
    {"lat": 51.500729, "lng": -0.124625, "name": "Central"},
    {"lat": 51.462600, "lng": 0.028600, "name": "Southeast"},
    {"lat": 51.520300, "lng": -0.255000, "name": "West"},
    {"lat": 51.535000, "lng": 0.025000, "name": "East"},
    {"lat": 51.445000, "lng": -0.030000, "name": "South"},
]


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class CandidateRuntime:
    def __init__(self, params: dict):
        self.params = params
        self.run_id = str(params["expanded_run"]["id"])
        self.paths = run_paths(self.run_id)
        self.destinations = pd.read_parquet(self.paths["destinations"])
        self.nodes = pd.read_parquet(self.paths["graph_nodes"])
        self.edges = pd.read_parquet(self.paths["graph_edges"])
        self.access_nodes = pd.read_parquet(self.paths["bus_stops"])
        graph_params = params["transport_graph"]
        self.graph = build_transport_graph(
            self.nodes,
            self.edges,
            transfer_radius_m=float(graph_params["transfer_radius_m"]),
            transfer_penalty_seconds=float(graph_params["transfer_penalty_seconds"]),
            walking_speed_mps=float(graph_params["walking_speed_mps"]),
        )
        self.stations = _station_catalog(self.nodes)
        self.model = load_bundle(self.paths["models"] / "selected.joblib")
        self.graph_model = load_bundle(self.paths["models"] / "graph_baseline.joblib")

    def response(self, participant_count: int) -> dict:
        friends = PARTICIPANTS[:participant_count]
        origins = pd.DataFrame(
            [
                {
                    "origin_id": f"friend_{index}",
                    "lat": friend["lat"],
                    "lng": friend["lng"],
                }
                for index, friend in enumerate(friends)
            ]
        )
        features = _feature_batch(
            origins,
            self.destinations,
            self.stations,
            self.graph,
            self.access_nodes,
            self.params,
        )
        model = predict(self.model, features).reshape(participant_count, -1) / 60
        graph = predict(self.graph_model, features).reshape(participant_count, -1) / 60
        graph_access = (
            features["graph_access_seconds"].to_numpy().reshape(participant_count, -1)
            / 60
        )
        graph_egress = (
            features["graph_egress_seconds"].to_numpy().reshape(participant_count, -1)
            / 60
        )
        graph_path = (
            features["graph_path_seconds"].to_numpy().reshape(participant_count, -1)
            / 60
        )
        graph_total = (
            features["graph_total_seconds"].to_numpy().reshape(participant_count, -1)
            / 60
        )
        rail_advantage = (
            features["graph_rail_advantage_seconds"]
            .to_numpy()
            .reshape(participant_count, -1)
            / 60
        )

        def balanced(values: np.ndarray) -> np.ndarray:
            spread = (
                values.std(axis=0, ddof=1)
                if participant_count > 1
                else np.zeros(values.shape[1])
            )
            return values.mean(axis=0) + 0.5 * spread

        mean = model.mean(axis=0)
        fairness = (
            model.std(axis=0, ddof=1) if participant_count > 1 else np.zeros(len(mean))
        )
        score = mean + 0.5 * fairness
        graph_score = balanced(graph)
        graph_path_score = balanced(graph_path)
        graph_total_score = balanced(graph_total)
        rail_advantage_score = balanced(rail_advantage)
        cells = []
        for index, destination in enumerate(self.destinations.itertuples(index=False)):
            boundary = destination.boundary
            if hasattr(boundary, "tolist"):
                boundary = boundary.tolist()
            cell = {
                "destination_id": str(destination.destination_id),
                "lat": float(destination.lat),
                "lng": float(destination.lng),
                "boundary": boundary,
                "h3_cell": str(destination.h3_cell),
                "h3_resolution": int(destination.h3_resolution),
                "grid_band": str(destination.grid_band),
                "grid_priority": int(destination.grid_priority),
                "cell_area_km2": float(destination.cell_area_km2),
                "coverage_region": str(destination.coverage_region),
                "x_index": index,
                "y_index": 0,
                "score_minutes": float(score[index]),
                "model_score_minutes": float(score[index]),
                "graph_score_minutes": float(graph_score[index]),
                "model_residual_minutes": float(score[index] - graph_score[index]),
                "graph_path_minutes": float(graph_path_score[index]),
                "graph_total_minutes": float(graph_total_score[index]),
                "graph_rail_advantage_minutes": float(rail_advantage_score[index]),
                "max_minutes": float(model[:, index].max()),
                "mean_minutes": float(mean[index]),
                "fairness_minutes": float(fairness[index]),
                "included_friend_indexes": ",".join(map(str, range(participant_count))),
            }
            for friend_index in range(participant_count):
                model_value = float(model[friend_index, index])
                graph_value = float(graph[friend_index, index])
                cell[f"friend_{friend_index}_minutes"] = model_value
                cell[f"friend_{friend_index}_model_minutes"] = float(
                    model[friend_index, index]
                )
                cell[f"friend_{friend_index}_graph_minutes"] = graph_value
                cell[f"friend_{friend_index}_model_residual_minutes"] = (
                    model_value - graph_value
                )
                cell[f"friend_{friend_index}_graph_access_minutes"] = float(
                    graph_access[friend_index, index]
                )
                cell[f"friend_{friend_index}_graph_egress_minutes"] = float(
                    graph_egress[friend_index, index]
                )
                cell[f"friend_{friend_index}_name"] = friends[friend_index]["name"]
            cells.append(cell)
        return {
            "lats": [],
            "lngs": [],
            "Z": [],
            "cells": cells,
            "metadata": {
                "source": "candidate_direct_benchmark",
                "cell_count": len(cells),
                "participant_count": participant_count,
            },
        }


def _production_response(participant_count: int) -> dict:
    frame = predict_group_surface(
        PARTICIPANTS[:participant_count],
        combine="balanced",
        grid_mode="h3",
        focus="inner",
        detail="fine",
    )
    return surface_to_grid_response(frame, "model_score_minutes")


def _measure_call(callable_) -> tuple[dict, dict]:
    start = time.perf_counter()
    response = callable_()
    inference = time.perf_counter() - start
    serialize_start = time.perf_counter()
    payload = json.dumps(response, separators=(",", ":"), default=str).encode("utf-8")
    serialization = time.perf_counter() - serialize_start
    process = psutil.Process()
    measurements = {
        "inference_seconds": inference,
        "serialization_seconds": serialization,
        "rss_bytes": int(process.memory_info().rss),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "raw_response_bytes": len(payload),
        "gzip_response_bytes": len(gzip.compress(payload)),
    }
    return response, measurements


def _single(variant: str, participants: int) -> dict:
    if variant == "candidate":
        runtime = CandidateRuntime(load_params())
        _, result = _measure_call(lambda: runtime.response(participants))
    else:
        _, result = _measure_call(lambda: _production_response(participants))
    return result


def _cold_subprocess(variant: str, participants: int) -> dict:
    command = [
        sys.executable,
        "-m",
        "equidistant_ml.surfaces.nfr",
        "single",
        "--variant",
        variant,
        "--participants",
        str(participants),
    ]
    start = time.perf_counter()
    result = subprocess.run(  # nosec B603
        command,
        cwd=project_path("."),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["cold_process_seconds"] = time.perf_counter() - start
    return payload


def _browser_benchmark(atlas_dir: Path) -> dict:
    result = subprocess.run(  # nosec B603 B607
        [
            "node",
            "build/benchmark-atlas.mjs",
            str(atlas_dir),
        ],
        cwd=project_path("frontend"),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run_nfr(params_path: str = "params.yaml") -> dict:
    params = load_params(params_path)
    run_id = str(params["expanded_run"]["id"])
    paths = run_paths(run_id)
    candidate_runtime = CandidateRuntime(params)
    production_warmup = _production_response(2)
    del production_warmup
    direct: dict[str, dict] = {"production": {}, "candidate": {}}
    for participant_count in (2, 3, 6):
        for variant in ("production", "candidate"):
            cold = _cold_subprocess(variant, participant_count)
            if variant == "candidate":
                _, first_warm = _measure_call(
                    lambda count=participant_count: candidate_runtime.response(count)
                )
                _, second_warm = _measure_call(
                    lambda count=participant_count: candidate_runtime.response(count)
                )
            else:
                _, first_warm = _measure_call(
                    lambda count=participant_count: _production_response(count)
                )
                _, second_warm = _measure_call(
                    lambda count=participant_count: _production_response(count)
                )
            direct[variant][str(participant_count)] = {
                "cold": cold,
                "warm_first": first_warm,
                "warm_second": second_warm,
            }

    production_atlas = project_path("frontend/public/model")
    candidate_atlas = paths["artifacts"] / "atlas"
    browser = {
        "production": _browser_benchmark(production_atlas),
        "candidate": _browser_benchmark(candidate_atlas),
    }
    current_build = _directory_size(project_path("frontend/dist"))
    current_model_assets = _directory_size(production_atlas)
    candidate_model_assets = _directory_size(candidate_atlas)
    sizes = {
        "frontend_dist_bytes": current_build,
        "current_model_asset_bytes": current_model_assets,
        "candidate_model_asset_bytes": candidate_model_assets,
        "estimated_candidate_sites_build_bytes": current_build
        - current_model_assets
        + candidate_model_assets,
        "candidate_training_model_set_bytes": _directory_size(paths["models"]),
        "candidate_selected_model_bytes": (paths["models"] / "selected.joblib")
        .stat()
        .st_size,
        "candidate_selected_direct_artifacts_bytes": (
            paths["models"] / "selected.joblib"
        )
        .stat()
        .st_size
        + (paths["models"] / "graph_baseline.joblib").stat().st_size,
        "candidate_graph_bytes": paths["graph_nodes"].stat().st_size
        + paths["graph_edges"].stat().st_size
        + paths["bus_stops"].stat().st_size,
        "candidate_data_bytes": _directory_size(paths["data"]),
    }
    direct_deltas = {}
    for participant_key in ("2", "3", "6"):
        production = direct["production"][participant_key]["warm_second"]
        candidate = direct["candidate"][participant_key]["warm_second"]
        direct_deltas[participant_key] = {
            metric: {
                "absolute": candidate[metric] - production[metric],
                "percent": (
                    (candidate[metric] / production[metric] - 1) * 100
                    if production[metric]
                    else None
                ),
            }
            for metric in (
                "inference_seconds",
                "serialization_seconds",
                "rss_bytes",
                "raw_response_bytes",
                "gzip_response_bytes",
            )
        }
    browser_deltas = {
        metric: {
            "absolute_ms": browser["candidate"][metric] - browser["production"][metric],
            "percent": (
                browser["candidate"][metric] / browser["production"][metric] - 1
            )
            * 100,
        }
        for metric in (
            "cold_core_load_ms",
            "first_surface_render_ms",
            "warm_surface_render_ms",
            "lazy_graph_load_ms",
        )
    }
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    model_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    atlas_evaluation = json.loads(
        (paths["artifacts"] / "atlas/evaluation.json").read_text(encoding="utf-8")
    )
    operational = dict(manifest.get("operational_metrics_seconds", {}))
    operational["training_and_evaluation"] = model_metrics[
        "training_and_evaluation_duration_seconds"
    ]
    operational["atlas_export"] = atlas_evaluation["atlas_export_duration_seconds"]
    baseline = {
        "recorded_before_run": {
            "sites_build_bytes": 7_100_000,
            "model_asset_set_bytes": 4_500_000,
            "initial_atlas_gzip_bytes": 1_530_000,
            "three_person_3032_cell_cold_seconds": 1.40,
            "three_person_3032_cell_warm_seconds": 1.20,
            "three_person_raw_json_bytes": 4_850_000,
            "three_person_gzip_bytes": 852_000,
            "peak_rss_bytes": 384_000_000,
        }
    }
    result = {
        "run_id": run_id,
        "measured_at": time.time(),
        "release_threshold_applied": False,
        "baseline": baseline,
        "direct_inference": direct,
        "browser_atlas": browser,
        "browser_methodology": (
            "local Node file load, decode, interpolation, and scoring; transfer "
            "cost is represented by raw/gzip/Brotli asset sizes"
        ),
        "sizes": sizes,
        "deltas": {
            "candidate_minus_production_direct": direct_deltas,
            "candidate_minus_production_browser": browser_deltas,
            "candidate_model_assets_bytes": candidate_model_assets
            - current_model_assets,
            "estimated_sites_build_bytes": sizes[
                "estimated_candidate_sites_build_bytes"
            ]
            - current_build,
        },
        "operational_metrics_seconds": operational,
        "worker_asset_protection": "covered by frontend/tests/worker.test.mjs",
    }
    _atomic_json(paths["artifacts"] / "nfr.json", result)
    report = {
        "run_id": run_id,
        "accuracy": model_metrics,
        "atlas": atlas_evaluation,
        "nfr": result,
        "promotion": {
            "approved": False,
            "candidate_only": True,
            "production_assets_unchanged": True,
        },
    }
    _atomic_json(paths["artifacts"] / "report.json", report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single")
    single.add_argument("--variant", choices=["production", "candidate"], required=True)
    single.add_argument("--participants", type=int, choices=[2, 3, 6], required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    if args.command == "single":
        print(json.dumps(_single(args.variant, args.participants)))
    else:
        print(json.dumps(run_nfr(args.params), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
