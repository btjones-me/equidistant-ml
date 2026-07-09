"""CLI entrypoints for the DVC surface modelling pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from equidistant_ml.surfaces.config import ensure_parent, load_params, project_path
from equidistant_ml.surfaces.features import build_feature_frame
from equidistant_ml.surfaces.geo import read_station_catalog
from equidistant_ml.surfaces.grid import (
    bbox_from_params,
    build_destination_grid,
    build_smoke_labels,
    sample_origin_anchors,
)
from equidistant_ml.surfaces.models import (
    evaluate_graph_models,
    evaluate_models,
    load_bundle,
    save_bundle,
    split_by_origin,
    train_baseline,
    train_graph_baseline,
    train_graph_residual_model,
    train_lightgbm,
)
from equidistant_ml.surfaces.transport_graph import (
    add_graph_features,
    build_reference_from_station_catalog,
    build_transport_graph,
    fetch_tfl_transport_reference,
    merge_references,
    read_naptan_nodes,
    write_graph_artifacts,
)
from equidistant_ml.surfaces.traveltime import (
    TravelTimeClient,
    TravelTimeCredentials,
    fetch_origin_surfaces,
)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    out = ensure_parent(path)
    df.to_parquet(out, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(project_path(path))


def cmd_build_grid(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    grid_params = params["grid"]
    grid = build_destination_grid(
        bbox_from_params(params),
        int(grid_params["x_size"]),
        int(grid_params["y_size"]),
    )
    write_parquet(grid, args.output)


def cmd_sample_origins(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    stations = read_station_catalog(params["features"]["stations_path"])
    origins = sample_origin_anchors(
        bbox_from_params(params),
        stations,
        int(params["sampling"]["origin_count"]),
        int(params["sampling"]["seed"]),
        float(params["sampling"]["station_bias_fraction"]),
    )
    write_parquet(origins, args.output)


def cmd_fetch_traveltime(args: argparse.Namespace) -> None:
    load_dotenv(project_path(".env"))
    params = load_params(args.params)
    origins = read_parquet(args.origins)
    destinations = read_parquet(args.destinations)
    if args.max_origins:
        origins = origins.head(args.max_origins)
    if args.max_destinations:
        destinations = destinations.head(args.max_destinations)

    travel_params = params["traveltime"]
    if args.mock:
        labels = build_smoke_labels(origins, destinations)
    else:
        client = TravelTimeClient(
            TravelTimeCredentials.from_env(),
            timeout_seconds=int(travel_params["timeout_seconds"]),
            sleep_seconds=float(travel_params["sleep_seconds"]),
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
            checkpoint_dir=args.checkpoint_dir,
        )
    write_parquet(labels, args.output)


def cmd_build_features(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    features_params = params["features"]
    labels = read_parquet(args.labels)
    origins = read_parquet(args.origins)
    destinations = read_parquet(args.destinations)
    stations = read_station_catalog(features_params["stations_path"])
    features = build_feature_frame(
        labels,
        origins,
        destinations,
        stations,
        nearest_station_count=int(features_params["nearest_station_count"]),
        density_radius_m=float(features_params["density_radius_m"]),
    )
    write_parquet(features, args.output)


def cmd_fetch_transport_reference(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    source_params = params.get("transport_sources", {})
    stations = read_station_catalog(params["features"]["stations_path"])
    references = [build_reference_from_station_catalog(stations)]

    if not args.mock and source_params.get("tfl_enabled", True):
        try:
            references.append(
                fetch_tfl_transport_reference(
                    source_params.get(
                        "tfl_modes",
                        ["tube", "overground", "elizabeth-line", "dlr", "tram"],
                    ),
                    bus_route_limit=int(source_params.get("bus_route_limit", 25)),
                    app_key=source_params.get("tfl_app_key"),
                    timeout_seconds=int(source_params.get("timeout_seconds", 30)),
                )
            )
        except Exception as exc:
            print(f"TfL transport fetch failed; using local reference only: {exc}")

    naptan_path = source_params.get("naptan_csv_path")
    if naptan_path:
        naptan_nodes = read_naptan_nodes(naptan_path)
        if not naptan_nodes.empty:
            references.append((naptan_nodes, naptan_nodes.iloc[0:0].copy()))

    nodes, edges = merge_references(references)
    write_parquet(nodes, args.nodes_output)
    write_parquet(edges, args.edges_output)


def cmd_build_transport_graph(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    graph_params = params.get("transport_graph", {})
    nodes = read_parquet(args.nodes)
    edges = read_parquet(args.edges)
    graph = build_transport_graph(
        nodes,
        edges,
        transfer_radius_m=float(graph_params.get("transfer_radius_m", 180)),
        transfer_penalty_seconds=float(
            graph_params.get("transfer_penalty_seconds", 120)
        ),
        walking_speed_mps=float(graph_params.get("walking_speed_mps", 1.35)),
    )
    write_graph_artifacts(
        graph,
        nodes_path=args.nodes_output,
        edges_path=args.edges_output,
    )


def cmd_build_graph_features(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    graph_params = params.get("transport_graph", {})
    feature_params = params.get("graph_features", {})
    features = read_parquet(args.features)
    nodes = read_parquet(args.graph_nodes)
    edges = read_parquet(args.graph_edges)
    graph = build_transport_graph(
        nodes,
        edges,
        transfer_radius_m=float(graph_params.get("transfer_radius_m", 180)),
        transfer_penalty_seconds=float(
            graph_params.get("transfer_penalty_seconds", 120)
        ),
        walking_speed_mps=float(graph_params.get("walking_speed_mps", 1.35)),
    )
    graph_features = add_graph_features(
        features,
        graph,
        walking_speed_mps=float(graph_params.get("walking_speed_mps", 1.35)),
        access_node_limit=int(feature_params.get("access_node_limit", 4)),
        max_access_distance_m=float(feature_params.get("max_access_distance_m", 1600)),
        bus_density_radius_m=float(feature_params.get("bus_density_radius_m", 500)),
    )
    write_parquet(graph_features, args.output)


def cmd_train_baseline(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    features = read_parquet(args.features)
    train, _ = split_by_origin(
        features,
        float(params["validation"]["validation_fraction"]),
        int(params["validation"]["seed"]),
    )
    bundle = train_baseline(train, params["baseline"])
    save_bundle(bundle, project_path(args.output))


def cmd_train_model(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    features = read_parquet(args.features)
    train, _ = split_by_origin(
        features,
        float(params["validation"]["validation_fraction"]),
        int(params["validation"]["seed"]),
    )
    bundle = train_lightgbm(train, params["model"])
    save_bundle(bundle, project_path(args.output))


def cmd_train_graph_baseline(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    features = read_parquet(args.features)
    train, _ = split_by_origin(
        features,
        float(params["validation"]["validation_fraction"]),
        int(params["validation"]["seed"]),
    )
    bundle = train_graph_baseline(
        train, params.get("graph_baseline", params["baseline"])
    )
    save_bundle(bundle, project_path(args.output))


def cmd_train_graph_model(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    features = read_parquet(args.features)
    graph_baseline = load_bundle(project_path(args.graph_baseline_model))
    train, _ = split_by_origin(
        features,
        float(params["validation"]["validation_fraction"]),
        int(params["validation"]["seed"]),
    )
    bundle = train_graph_residual_model(
        train,
        graph_baseline,
        params.get("graph_model", params["model"]),
    )
    save_bundle(bundle, project_path(args.output))


def cmd_evaluate(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    features = read_parquet(args.features)
    baseline = load_bundle(project_path(args.baseline_model))
    model = load_bundle(project_path(args.model))
    metrics = evaluate_models(
        features,
        baseline,
        model,
        float(params["validation"]["validation_fraction"]),
        int(params["validation"]["seed"]),
    )
    out = ensure_parent(args.output)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def cmd_evaluate_graph(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    features = read_parquet(args.features)
    baseline = load_bundle(project_path(args.baseline_model))
    graph_baseline = load_bundle(project_path(args.graph_baseline_model))
    model = load_bundle(project_path(args.model))
    metrics = evaluate_graph_models(
        features,
        baseline,
        graph_baseline,
        model,
        float(params["validation"]["validation_fraction"]),
        int(params["validation"]["seed"]),
    )
    out = ensure_parent(args.output)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def cmd_smoke(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    cmd_build_grid(
        argparse.Namespace(params=args.params, output=str(out_dir / "grid.parquet"))
    )
    cmd_sample_origins(
        argparse.Namespace(params=args.params, output=str(out_dir / "origins.parquet"))
    )
    cmd_fetch_traveltime(
        argparse.Namespace(
            params=args.params,
            origins=str(out_dir / "origins.parquet"),
            destinations=str(out_dir / "grid.parquet"),
            output=str(out_dir / "labels.parquet"),
            mock=True,
            max_origins=args.max_origins,
            max_destinations=args.max_destinations,
            checkpoint_dir=None,
        )
    )
    cmd_build_features(
        argparse.Namespace(
            params=args.params,
            labels=str(out_dir / "labels.parquet"),
            origins=str(out_dir / "origins.parquet"),
            destinations=str(out_dir / "grid.parquet"),
            output=str(out_dir / "features.parquet"),
        )
    )


def cmd_graph_smoke(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    cmd_smoke(args)
    cmd_fetch_transport_reference(
        argparse.Namespace(
            params=args.params,
            mock=True,
            nodes_output=str(out_dir / "transport_nodes.parquet"),
            edges_output=str(out_dir / "transport_edges.parquet"),
        )
    )
    cmd_build_transport_graph(
        argparse.Namespace(
            params=args.params,
            nodes=str(out_dir / "transport_nodes.parquet"),
            edges=str(out_dir / "transport_edges.parquet"),
            nodes_output=str(out_dir / "transport_graph_nodes.parquet"),
            edges_output=str(out_dir / "transport_graph_edges.parquet"),
        )
    )
    cmd_build_graph_features(
        argparse.Namespace(
            params=args.params,
            features=str(out_dir / "features.parquet"),
            graph_nodes=str(out_dir / "transport_graph_nodes.parquet"),
            graph_edges=str(out_dir / "transport_graph_edges.parquet"),
            output=str(out_dir / "graph_features.parquet"),
        )
    )
    cmd_train_baseline(
        argparse.Namespace(
            params=args.params,
            features=str(out_dir / "graph_features.parquet"),
            output=str(out_dir / "baseline_model.joblib"),
        )
    )
    cmd_train_graph_baseline(
        argparse.Namespace(
            params=args.params,
            features=str(out_dir / "graph_features.parquet"),
            output=str(out_dir / "graph_baseline_model.joblib"),
        )
    )
    cmd_train_graph_model(
        argparse.Namespace(
            params=args.params,
            features=str(out_dir / "graph_features.parquet"),
            graph_baseline_model=str(out_dir / "graph_baseline_model.joblib"),
            output=str(out_dir / "travel_time_graph_residual.joblib"),
        )
    )
    cmd_evaluate_graph(
        argparse.Namespace(
            params=args.params,
            features=str(out_dir / "graph_features.parquet"),
            baseline_model=str(out_dir / "baseline_model.joblib"),
            graph_baseline_model=str(out_dir / "graph_baseline_model.joblib"),
            model=str(out_dir / "travel_time_graph_residual.joblib"),
            output=str(out_dir / "graph_model.json"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_params(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--params", default="params.yaml")

    build_grid = subparsers.add_parser("build-grid")
    add_params(build_grid)
    build_grid.add_argument("--output", default="data/interim/destination_grid.parquet")
    build_grid.set_defaults(func=cmd_build_grid)

    sample_origins = subparsers.add_parser("sample-origins")
    add_params(sample_origins)
    sample_origins.add_argument(
        "--output", default="data/interim/origin_anchors.parquet"
    )
    sample_origins.set_defaults(func=cmd_sample_origins)

    fetch = subparsers.add_parser("fetch-traveltime")
    add_params(fetch)
    fetch.add_argument("--origins", default="data/interim/origin_anchors.parquet")
    fetch.add_argument(
        "--destinations", default="data/interim/destination_grid.parquet"
    )
    fetch.add_argument("--output", default="data/interim/traveltime_labels.parquet")
    fetch.add_argument("--mock", action="store_true")
    fetch.add_argument("--max-origins", type=int)
    fetch.add_argument("--max-destinations", type=int)
    fetch.add_argument(
        "--checkpoint-dir",
        default="data/interim/traveltime_labels.parts",
        help="Directory for per-origin fetch checkpoints. Existing shards are reused.",
    )
    fetch.set_defaults(func=cmd_fetch_traveltime)

    features = subparsers.add_parser("build-features")
    add_params(features)
    features.add_argument("--labels", default="data/interim/traveltime_labels.parquet")
    features.add_argument("--origins", default="data/interim/origin_anchors.parquet")
    features.add_argument(
        "--destinations", default="data/interim/destination_grid.parquet"
    )
    features.add_argument(
        "--output", default="data/features/traveltime_features.parquet"
    )
    features.set_defaults(func=cmd_build_features)

    transport_ref = subparsers.add_parser("fetch-transport-reference")
    add_params(transport_ref)
    transport_ref.add_argument("--mock", action="store_true")
    transport_ref.add_argument(
        "--nodes-output", default="data/reference/transport_nodes.parquet"
    )
    transport_ref.add_argument(
        "--edges-output", default="data/reference/transport_edges.parquet"
    )
    transport_ref.set_defaults(func=cmd_fetch_transport_reference)

    transport_graph = subparsers.add_parser("build-transport-graph")
    add_params(transport_graph)
    transport_graph.add_argument(
        "--nodes", default="data/reference/transport_nodes.parquet"
    )
    transport_graph.add_argument(
        "--edges", default="data/reference/transport_edges.parquet"
    )
    transport_graph.add_argument(
        "--nodes-output", default="data/reference/transport_graph_nodes.parquet"
    )
    transport_graph.add_argument(
        "--edges-output", default="data/reference/transport_graph_edges.parquet"
    )
    transport_graph.set_defaults(func=cmd_build_transport_graph)

    graph_features = subparsers.add_parser("build-graph-features")
    add_params(graph_features)
    graph_features.add_argument(
        "--features", default="data/features/traveltime_features.parquet"
    )
    graph_features.add_argument(
        "--graph-nodes", default="data/reference/transport_graph_nodes.parquet"
    )
    graph_features.add_argument(
        "--graph-edges", default="data/reference/transport_graph_edges.parquet"
    )
    graph_features.add_argument(
        "--output", default="data/features/traveltime_graph_features.parquet"
    )
    graph_features.set_defaults(func=cmd_build_graph_features)

    baseline = subparsers.add_parser("train-baseline")
    add_params(baseline)
    baseline.add_argument(
        "--features", default="data/features/traveltime_features.parquet"
    )
    baseline.add_argument("--output", default="models/baseline_model.joblib")
    baseline.set_defaults(func=cmd_train_baseline)

    model = subparsers.add_parser("train-model")
    add_params(model)
    model.add_argument(
        "--features", default="data/features/traveltime_features.parquet"
    )
    model.add_argument("--output", default="models/travel_time_model.joblib")
    model.set_defaults(func=cmd_train_model)

    graph_baseline = subparsers.add_parser("train-graph-baseline")
    add_params(graph_baseline)
    graph_baseline.add_argument(
        "--features", default="data/features/traveltime_graph_features.parquet"
    )
    graph_baseline.add_argument(
        "--output", default="models/graph_baseline_model.joblib"
    )
    graph_baseline.set_defaults(func=cmd_train_graph_baseline)

    graph_model = subparsers.add_parser("train-graph-model")
    add_params(graph_model)
    graph_model.add_argument(
        "--features", default="data/features/traveltime_graph_features.parquet"
    )
    graph_model.add_argument(
        "--graph-baseline-model", default="models/graph_baseline_model.joblib"
    )
    graph_model.add_argument(
        "--output", default="models/travel_time_graph_residual.joblib"
    )
    graph_model.set_defaults(func=cmd_train_graph_model)

    evaluate = subparsers.add_parser("evaluate")
    add_params(evaluate)
    evaluate.add_argument(
        "--features", default="data/features/traveltime_features.parquet"
    )
    evaluate.add_argument("--baseline-model", default="models/baseline_model.joblib")
    evaluate.add_argument("--model", default="models/travel_time_model.joblib")
    evaluate.add_argument("--output", default="metrics/model.json")
    evaluate.set_defaults(func=cmd_evaluate)

    evaluate_graph = subparsers.add_parser("evaluate-graph")
    add_params(evaluate_graph)
    evaluate_graph.add_argument(
        "--features", default="data/features/traveltime_graph_features.parquet"
    )
    evaluate_graph.add_argument(
        "--baseline-model", default="models/baseline_model.joblib"
    )
    evaluate_graph.add_argument(
        "--graph-baseline-model", default="models/graph_baseline_model.joblib"
    )
    evaluate_graph.add_argument(
        "--model", default="models/travel_time_graph_residual.joblib"
    )
    evaluate_graph.add_argument("--output", default="metrics/model.json")
    evaluate_graph.set_defaults(func=cmd_evaluate_graph)

    smoke = subparsers.add_parser("smoke")
    add_params(smoke)
    smoke.add_argument("--output-dir", default="data/smoke")
    smoke.add_argument("--max-origins", type=int, default=6)
    smoke.add_argument("--max-destinations", type=int, default=36)
    smoke.set_defaults(func=cmd_smoke)

    graph_smoke = subparsers.add_parser("graph-smoke")
    add_params(graph_smoke)
    graph_smoke.add_argument("--output-dir", default="data/smoke_graph")
    graph_smoke.add_argument("--max-origins", type=int, default=6)
    graph_smoke.add_argument("--max-destinations", type=int, default=36)
    graph_smoke.set_defaults(func=cmd_graph_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
