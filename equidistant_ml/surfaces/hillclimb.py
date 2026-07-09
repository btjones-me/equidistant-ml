"""Hill-climb graph/model combinations against TravelTime-labelled holdouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from equidistant_ml.surfaces.config import ensure_parent, load_params, project_path
from equidistant_ml.surfaces.features import feature_columns
from equidistant_ml.surfaces.geo import read_station_catalog
from equidistant_ml.surfaces.models import (
    ModelBundle,
    column_bundle,
    load_bundle,
    predict,
    save_bundle,
    train_graph_baseline,
    train_graph_residual_model,
    weighted_blend_bundle,
)
from equidistant_ml.surfaces.transport_graph import (
    add_graph_features,
    build_reference_from_station_catalog,
    build_transport_graph,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    bundle: ModelBundle
    family: str


def read_frame(path: str) -> pd.DataFrame:
    return pd.read_parquet(project_path(path))


def write_frame(frame: pd.DataFrame, path: str) -> None:
    out = ensure_parent(path)
    frame.to_parquet(out, index=False)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_error = np.abs(y_true - y_pred)
    r2 = 0.0
    if len(y_true) > 1 and not np.allclose(y_true, y_true[0]):
        r2 = float(r2_score(y_true, y_pred))
    return {
        "mae_min": float(mean_absolute_error(y_true, y_pred) / 60),
        "rmse_min": float((mean_squared_error(y_true, y_pred) ** 0.5) / 60),
        "median_abs_error_min": float(np.median(abs_error) / 60),
        "p80_abs_error_min": float(np.quantile(abs_error, 0.80) / 60),
        "p90_abs_error_min": float(np.quantile(abs_error, 0.90) / 60),
        "p95_abs_error_min": float(np.quantile(abs_error, 0.95) / 60),
        "mean_signed_error_min": float(np.mean(y_pred - y_true) / 60),
        "within_5_min_pct": float(np.mean(abs_error <= 300) * 100),
        "within_10_min_pct": float(np.mean(abs_error <= 600) * 100),
        "r2": r2,
    }


def split_origins(
    features: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    origin_ids = features["origin_id"].drop_duplicates().to_numpy()
    rng.shuffle(origin_ids)
    validation_count = max(1, int(round(len(origin_ids) * validation_fraction)))
    validation_ids = set(origin_ids[:validation_count])
    fit = features[~features["origin_id"].isin(validation_ids)].copy()
    tune = features[features["origin_id"].isin(validation_ids)].copy()
    return fit, tune


def ensure_graph_features(
    source_path: str,
    output_path: str,
    *,
    params: dict[str, Any],
    refresh: bool,
) -> pd.DataFrame:
    out = project_path(output_path)
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    frame = read_frame(source_path)
    stations = read_station_catalog(params["features"]["stations_path"])
    nodes, edges = build_reference_from_station_catalog(stations)
    graph_params = params.get("transport_graph", {})
    feature_params = params.get("graph_features", {})
    graph = build_transport_graph(
        nodes,
        edges,
        transfer_radius_m=float(graph_params.get("transfer_radius_m", 180)),
        transfer_penalty_seconds=float(
            graph_params.get("transfer_penalty_seconds", 120)
        ),
        walking_speed_mps=float(graph_params.get("walking_speed_mps", 1.35)),
    )
    graph_frame = add_graph_features(
        frame,
        graph,
        walking_speed_mps=float(graph_params.get("walking_speed_mps", 1.35)),
        access_node_limit=int(feature_params.get("access_node_limit", 4)),
        max_access_distance_m=float(feature_params.get("max_access_distance_m", 1600)),
        bus_density_radius_m=float(feature_params.get("bus_density_radius_m", 500)),
    )
    write_frame(graph_frame, output_path)
    return graph_frame


def train_direct_regressor(
    features: pd.DataFrame, params: dict[str, Any]
) -> ModelBundle:
    cols = feature_columns(features)
    try:
        from lightgbm import LGBMRegressor

        model: Any = LGBMRegressor(
            objective="regression",
            n_estimators=int(params.get("n_estimators", 350)),
            learning_rate=float(params.get("learning_rate", 0.04)),
            num_leaves=int(params.get("num_leaves", 39)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            random_state=int(params.get("seed", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
            verbosity=-1,
        )
        model_type = "direct_lightgbm_regressor"
    except (ImportError, OSError):
        model = HistGradientBoostingRegressor(
            max_iter=int(params.get("n_estimators", 350)),
            learning_rate=float(params.get("learning_rate", 0.04)),
            random_state=int(params.get("seed", 42)),
            min_samples_leaf=int(params.get("min_child_samples", 20)),
        )
        model_type = "direct_hist_gradient_regressor"
    wrapped = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    wrapped.fit(features[cols], features["target_travel_time_seconds"])
    return ModelBundle(model=wrapped, feature_columns=cols, model_type=model_type)


def train_residual_over_base(
    features: pd.DataFrame,
    base_bundle: ModelBundle,
    params: dict[str, Any],
    *,
    base_prediction: np.ndarray | None = None,
) -> ModelBundle:
    cols = feature_columns(features)
    if base_prediction is None:
        base_prediction = predict(base_bundle, features)
    residual_target = (
        features["target_travel_time_seconds"].to_numpy() - base_prediction
    )
    try:
        from lightgbm import LGBMRegressor

        regressor: Any = LGBMRegressor(
            objective="regression",
            n_estimators=int(params.get("n_estimators", 350)),
            learning_rate=float(params.get("learning_rate", 0.04)),
            num_leaves=int(params.get("num_leaves", 39)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            random_state=int(params.get("seed", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
            verbosity=-1,
        )
        model_type = "residual_over_base_lightgbm"
    except (ImportError, OSError):
        regressor = HistGradientBoostingRegressor(
            max_iter=int(params.get("n_estimators", 350)),
            learning_rate=float(params.get("learning_rate", 0.04)),
            random_state=int(params.get("seed", 42)),
            min_samples_leaf=int(params.get("min_child_samples", 20)),
        )
        model_type = "residual_over_base_hist_gradient_fallback"
    regressor.fit(features[cols], residual_target)
    return ModelBundle(
        model={
            "base_bundle": base_bundle,
            "residual_regressor": regressor,
            "residual_feature_columns": cols,
            "cap_seconds": float(features["target_travel_time_seconds"].max()),
        },
        feature_columns=sorted(set(base_bundle.feature_columns).union(cols)),
        model_type=model_type,
    )


def residual_param_variants(
    base_params: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    default = dict(base_params)
    seed = int(default.get("seed", 42))
    n_jobs = int(default.get("n_jobs", -1))
    variants = [
        ("default", {}),
        (
            "smoother",
            {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 35,
            },
        ),
        (
            "higher_capacity",
            {
                "n_estimators": 550,
                "learning_rate": 0.025,
                "num_leaves": 63,
                "min_child_samples": 10,
            },
        ),
        (
            "higher_capacity_more_trees",
            {
                "n_estimators": 850,
                "learning_rate": 0.015,
                "num_leaves": 63,
                "min_child_samples": 10,
            },
        ),
        (
            "higher_capacity_more_leaves",
            {
                "n_estimators": 700,
                "learning_rate": 0.02,
                "num_leaves": 95,
                "min_child_samples": 8,
            },
        ),
        (
            "more_leaves_127",
            {
                "n_estimators": 800,
                "learning_rate": 0.018,
                "num_leaves": 127,
                "min_child_samples": 8,
            },
        ),
        (
            "more_leaves_127_low_child",
            {
                "n_estimators": 800,
                "learning_rate": 0.018,
                "num_leaves": 127,
                "min_child_samples": 5,
            },
        ),
        (
            "more_leaves_159",
            {
                "n_estimators": 850,
                "learning_rate": 0.016,
                "num_leaves": 159,
                "min_child_samples": 8,
            },
        ),
        (
            "more_leaves_159_more_trees",
            {
                "n_estimators": 1100,
                "learning_rate": 0.012,
                "num_leaves": 159,
                "min_child_samples": 8,
            },
        ),
        (
            "more_leaves_191",
            {
                "n_estimators": 950,
                "learning_rate": 0.014,
                "num_leaves": 191,
                "min_child_samples": 8,
            },
        ),
        (
            "more_leaves_191_child12",
            {
                "n_estimators": 900,
                "learning_rate": 0.014,
                "num_leaves": 191,
                "min_child_samples": 12,
            },
        ),
        (
            "more_leaves_255_regularized",
            {
                "n_estimators": 950,
                "learning_rate": 0.012,
                "num_leaves": 255,
                "min_child_samples": 10,
                "reg_lambda": 1.0,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ),
        (
            "more_leaves_95_more_trees",
            {
                "n_estimators": 950,
                "learning_rate": 0.014,
                "num_leaves": 95,
                "min_child_samples": 8,
            },
        ),
        (
            "more_leaves_95_child12",
            {
                "n_estimators": 750,
                "learning_rate": 0.018,
                "num_leaves": 95,
                "min_child_samples": 12,
            },
        ),
        (
            "higher_capacity_low_child",
            {
                "n_estimators": 650,
                "learning_rate": 0.02,
                "num_leaves": 63,
                "min_child_samples": 5,
            },
        ),
        (
            "higher_capacity_regularized",
            {
                "n_estimators": 650,
                "learning_rate": 0.02,
                "num_leaves": 63,
                "min_child_samples": 10,
                "reg_lambda": 1.0,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ),
        (
            "higher_capacity_lr03",
            {
                "n_estimators": 450,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "min_child_samples": 10,
            },
        ),
        (
            "small_leaves",
            {
                "n_estimators": 450,
                "learning_rate": 0.035,
                "num_leaves": 23,
                "min_child_samples": 25,
            },
        ),
        (
            "fast_regular",
            {
                "n_estimators": 220,
                "learning_rate": 0.06,
                "num_leaves": 31,
                "min_child_samples": 30,
            },
        ),
    ]
    resolved = []
    for name, override in variants:
        params = {**default, **override, "seed": seed, "n_jobs": n_jobs}
        resolved.append((name, params))
    return resolved


def train_ridge_stack(
    y_true: np.ndarray,
    prediction_frame: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[LinearRegression, list[str]]:
    _ = alpha
    columns = list(prediction_frame.columns)
    stacker = LinearRegression(positive=True)
    stacker.fit(prediction_frame[columns], y_true)
    return stacker, columns


def optimise_weight(
    y_true: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    steps: int = 40,
) -> tuple[float, dict[str, float]]:
    best_weight = 0.0
    best_metrics: dict[str, float] | None = None
    for weight in np.linspace(0, 1, steps + 1):
        prediction = weight * left + (1 - weight) * right
        result = metrics(y_true, prediction)
        if best_metrics is None or result["mae_min"] < best_metrics["mae_min"]:
            best_weight = float(weight)
            best_metrics = result
    if best_metrics is None:
        raise RuntimeError("No blend weights were evaluated.")
    return best_weight, best_metrics


def optimise_simplex(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    step: float = 0.05,
) -> tuple[dict[str, float], dict[str, float]]:
    names = list(predictions)
    if len(names) != 3:
        raise ValueError("Simplex optimisation currently expects three predictions.")
    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None
    values = np.arange(0, 1 + step / 2, step)
    for first in values:
        for second in values:
            third = 1 - first - second
            if third < -1e-9:
                continue
            weights = {
                names[0]: float(first),
                names[1]: float(second),
                names[2]: float(max(third, 0.0)),
            }
            prediction = np.zeros_like(y_true, dtype=float)
            for name in names:
                prediction = prediction + weights[name] * predictions[name]
            result = metrics(y_true, prediction)
            if best_metrics is None or result["mae_min"] < best_metrics["mae_min"]:
                best_metrics = result
                best_weights = weights
    if best_weights is None or best_metrics is None:
        raise RuntimeError("No simplex weights were evaluated.")
    return best_weights, best_metrics


def sliced_metrics(
    features: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float] | None]:
    y_true = features["target_travel_time_seconds"].to_numpy()
    slices: dict[str, pd.Series] = {
        "all": pd.Series(True, index=features.index),
        "zone1_core_destinations": features.get("grid_band", "").eq("Zone 1 core"),
        "zones2_3_destinations": features.get("grid_band", "").eq("Zones 2-3"),
        "truth_under_45min": features["target_travel_time_seconds"] < 2700,
        "truth_45_to_90min": features["target_travel_time_seconds"].between(
            2700, 5400, inclusive="left"
        ),
        "rail_corridor_access": features.get(
            "destination_nearest_heavy_rail_distance_m",
            pd.Series(np.inf, index=features.index),
        )
        <= 700,
        "graph_path_available": features.get(
            "graph_has_path", pd.Series(0, index=features.index)
        )
        >= 1,
    }
    result: dict[str, dict[str, float] | None] = {}
    for name, mask in slices.items():
        mask_array = mask.to_numpy(dtype=bool)
        if mask_array.sum() < 2:
            result[name] = None
            continue
        metric = metrics(y_true[mask_array], y_pred[mask_array])
        metric["rows"] = int(mask_array.sum())
        result[name] = metric
    return result


def evaluate_candidate(
    candidate: Candidate,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    y_true = frame["target_travel_time_seconds"].to_numpy()
    prediction = predict(candidate.bundle, frame)
    return prediction, metrics(y_true, prediction)


def run_hillclimb(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    train_features = ensure_graph_features(
        args.train_features,
        args.train_graph_features,
        params=params,
        refresh=args.refresh_graph_features,
    )
    holdout_features = ensure_graph_features(
        args.holdout_features,
        args.holdout_graph_features,
        params=params,
        refresh=args.refresh_graph_features,
    )
    fit_features, tune_features = split_origins(
        train_features,
        validation_fraction=args.tune_fraction,
        seed=args.seed,
    )

    incumbent = Candidate(
        "current_central_aug_model",
        load_bundle(project_path(args.current_model)),
        "incumbent",
    )
    fit_incumbent_prediction = predict(incumbent.bundle, fit_features)
    full_incumbent_prediction = predict(incumbent.bundle, train_features)
    graph_prior = Candidate(
        "graph_prior_column",
        column_bundle("graph_total_seconds"),
        "graph_prior",
    )
    graph_baseline = Candidate(
        "graph_ridge_baseline",
        train_graph_baseline(fit_features, params.get("graph_baseline", {})),
        "graph_baseline",
    )
    graph_residual = Candidate(
        "graph_residual_model",
        train_graph_residual_model(
            fit_features,
            graph_baseline.bundle,
            params.get("graph_model", params.get("model", {})),
        ),
        "graph_residual",
    )
    graph_direct = Candidate(
        "graph_direct_regressor",
        train_direct_regressor(
            fit_features,
            params.get("graph_model", params.get("model", {})),
        ),
        "graph_direct",
    )
    current_residual = Candidate(
        "current_model_graph_residual",
        train_residual_over_base(
            fit_features,
            incumbent.bundle,
            params.get("graph_model", params.get("model", {})),
            base_prediction=fit_incumbent_prediction,
        ),
        "current_model_residual",
    )
    full_graph_baseline = Candidate(
        "graph_ridge_baseline_full_train",
        train_graph_baseline(train_features, params.get("graph_baseline", {})),
        "graph_baseline_full_train",
    )
    full_graph_residual = Candidate(
        "graph_residual_model_full_train",
        train_graph_residual_model(
            train_features,
            full_graph_baseline.bundle,
            params.get("graph_model", params.get("model", {})),
        ),
        "graph_residual_full_train",
    )
    full_graph_direct = Candidate(
        "graph_direct_regressor_full_train",
        train_direct_regressor(
            train_features,
            params.get("graph_model", params.get("model", {})),
        ),
        "graph_direct_full_train",
    )
    full_current_residual = Candidate(
        "current_model_graph_residual_full_train",
        train_residual_over_base(
            train_features,
            incumbent.bundle,
            params.get("graph_model", params.get("model", {})),
            base_prediction=full_incumbent_prediction,
        ),
        "current_model_residual_full_train",
    )
    current_residual_variants = [
        Candidate(
            f"current_model_graph_residual_full_train_{variant_name}",
            train_residual_over_base(
                train_features,
                incumbent.bundle,
                variant_params,
                base_prediction=full_incumbent_prediction,
            ),
            f"current_model_residual_full_train_{variant_name}",
        )
        for variant_name, variant_params in residual_param_variants(
            params.get("graph_model", params.get("model", {}))
        )
        if variant_name != "default"
    ]

    base_candidates = [
        incumbent,
        graph_prior,
        graph_baseline,
        graph_residual,
        graph_direct,
        current_residual,
    ]
    y_tune = tune_features["target_travel_time_seconds"].to_numpy()
    tune_predictions: dict[str, np.ndarray] = {}
    candidates = list(base_candidates)
    trace: list[dict[str, Any]] = []

    for candidate in base_candidates:
        prediction, result = evaluate_candidate(candidate, tune_features)
        tune_predictions[candidate.name] = prediction
        trace.append(
            {
                "iteration": len(trace),
                "candidate": candidate.name,
                "family": candidate.family,
                "tune": result,
            }
        )

    for candidate in [
        graph_prior,
        graph_baseline,
        graph_residual,
        graph_direct,
        current_residual,
    ]:
        weight, result = optimise_weight(
            y_tune,
            tune_predictions[incumbent.name],
            tune_predictions[candidate.name],
        )
        blend = Candidate(
            f"blend_current_{candidate.name}_{weight:.3f}",
            weighted_blend_bundle(
                [
                    (incumbent.name, weight, incumbent.bundle),
                    (candidate.name, 1 - weight, candidate.bundle),
                ]
            ),
            "two_model_blend",
        )
        candidates.append(blend)
        tune_predictions[blend.name] = predict(blend.bundle, tune_features)
        trace.append(
            {
                "iteration": len(trace),
                "candidate": blend.name,
                "family": blend.family,
                "weights": {
                    incumbent.name: weight,
                    candidate.name: 1 - weight,
                },
                "tune": result,
            }
        )

    simplex_inputs = {
        incumbent.name: tune_predictions[incumbent.name],
        graph_residual.name: tune_predictions[graph_residual.name],
        graph_direct.name: tune_predictions[graph_direct.name],
    }
    simplex_weights, simplex_result = optimise_simplex(y_tune, simplex_inputs)
    simplex_blend = Candidate(
        "simplex_current_residual_direct",
        weighted_blend_bundle(
            [
                (incumbent.name, simplex_weights[incumbent.name], incumbent.bundle),
                (
                    graph_residual.name,
                    simplex_weights[graph_residual.name],
                    graph_residual.bundle,
                ),
                (
                    graph_direct.name,
                    simplex_weights[graph_direct.name],
                    graph_direct.bundle,
                ),
            ]
        ),
        "three_model_simplex",
    )
    candidates.append(simplex_blend)
    tune_predictions[simplex_blend.name] = predict(simplex_blend.bundle, tune_features)
    trace.append(
        {
            "iteration": len(trace),
            "candidate": simplex_blend.name,
            "family": simplex_blend.family,
            "weights": simplex_weights,
            "tune": simplex_result,
        }
    )

    stack_prediction_frame = pd.DataFrame(
        {
            incumbent.name: tune_predictions[incumbent.name],
            graph_prior.name: tune_predictions[graph_prior.name],
            graph_baseline.name: tune_predictions[graph_baseline.name],
            graph_residual.name: tune_predictions[graph_residual.name],
            graph_direct.name: tune_predictions[graph_direct.name],
            current_residual.name: tune_predictions[current_residual.name],
        }
    )
    stacker, stack_columns = train_ridge_stack(
        y_tune,
        stack_prediction_frame,
        alpha=float(args.stack_alpha),
    )
    stack_weights = dict(
        zip(
            stack_columns,
            stacker.coef_.astype(float).tolist(),
            strict=True,
        )
    )
    stack_blend = Candidate(
        "ridge_stack_predictions",
        weighted_blend_bundle(
            [
                (incumbent.name, stack_weights[incumbent.name], incumbent.bundle),
                (graph_prior.name, stack_weights[graph_prior.name], graph_prior.bundle),
                (
                    graph_baseline.name,
                    stack_weights[graph_baseline.name],
                    graph_baseline.bundle,
                ),
                (
                    graph_residual.name,
                    stack_weights[graph_residual.name],
                    graph_residual.bundle,
                ),
                (
                    graph_direct.name,
                    stack_weights[graph_direct.name],
                    graph_direct.bundle,
                ),
                (
                    current_residual.name,
                    stack_weights[current_residual.name],
                    current_residual.bundle,
                ),
            ],
            intercept=float(stacker.intercept_),
        ),
        "ridge_stack",
    )
    candidates.append(stack_blend)
    tune_prediction = predict(stack_blend.bundle, tune_features)
    trace.append(
        {
            "iteration": len(trace),
            "candidate": stack_blend.name,
            "family": stack_blend.family,
            "weights": stack_weights,
            "intercept": float(stacker.intercept_),
            "tune": metrics(y_tune, tune_prediction),
        }
    )

    full_train_candidates = [
        full_graph_baseline,
        full_graph_residual,
        full_graph_direct,
        full_current_residual,
        *current_residual_variants,
    ]
    candidates.extend(full_train_candidates)
    for candidate in full_train_candidates:
        prediction = predict(candidate.bundle, tune_features)
        trace.append(
            {
                "iteration": len(trace),
                "candidate": candidate.name,
                "family": candidate.family,
                "trained_on_full_train": True,
                "tune": metrics(y_tune, prediction),
            }
        )

    holdout_results: dict[str, dict[str, Any]] = {}
    incumbent_holdout_prediction, incumbent_holdout_metrics = evaluate_candidate(
        incumbent, holdout_features
    )
    incumbent_mae = incumbent_holdout_metrics["mae_min"]
    best_candidate: Candidate | None = None
    best_holdout_metrics: dict[str, float] | None = None
    best_prediction: np.ndarray | None = None
    for candidate in candidates:
        prediction, result = evaluate_candidate(candidate, holdout_features)
        result["mae_delta_vs_incumbent_min"] = result["mae_min"] - incumbent_mae
        result["mae_improvement_vs_incumbent_pct"] = float(
            (1 - result["mae_min"] / incumbent_mae) * 100
        )
        holdout_results[candidate.name] = {
            "family": candidate.family,
            "metrics": result,
            "slices": sliced_metrics(holdout_features, prediction),
        }
        if (
            best_holdout_metrics is None
            or result["mae_min"] < best_holdout_metrics["mae_min"]
        ):
            best_candidate = candidate
            best_holdout_metrics = result
            best_prediction = prediction

    if (
        best_candidate is None
        or best_holdout_metrics is None
        or best_prediction is None
    ):
        raise RuntimeError("No candidate was evaluated.")

    should_promote = best_candidate.name != incumbent.name and (
        best_holdout_metrics["mae_min"]
        < incumbent_mae - float(args.min_promote_delta_min)
    )
    if should_promote:
        save_bundle(best_candidate.bundle, project_path(args.output_model))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_features": args.train_features,
        "holdout_features": args.holdout_features,
        "fit_origins": int(fit_features["origin_id"].nunique()),
        "tune_origins": int(tune_features["origin_id"].nunique()),
        "holdout_origins": int(holdout_features["origin_id"].nunique()),
        "holdout_rows": int(len(holdout_features)),
        "incumbent": incumbent.name,
        "best_candidate": best_candidate.name,
        "promoted": should_promote,
        "promoted_model_path": args.output_model if should_promote else None,
        "promotion_min_delta_min": float(args.min_promote_delta_min),
        "trace": trace,
        "holdout_results": holdout_results,
    }
    out = ensure_parent(args.output_metrics)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    summary_path = ensure_parent(args.output_summary)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# Graph Combination Hill Climb\n\n")
        handle.write(f"Generated: `{report['generated_at']}`\n\n")
        handle.write(
            f"Best candidate: `{best_candidate.name}` "
            f"({best_holdout_metrics['mae_min']:.3f} min MAE, "
            f"{best_holdout_metrics['mae_improvement_vs_incumbent_pct']:.2f}% "
            "vs incumbent).\n\n"
        )
        handle.write(
            "| Candidate | Family | Holdout MAE | P90 | Within 5 min | "
            "Delta vs incumbent |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for name, result in sorted(
            holdout_results.items(),
            key=lambda item: item[1]["metrics"]["mae_min"],
        ):
            metric = cast(dict[str, float], result["metrics"])
            handle.write(
                f"| `{name}` | {result['family']} | {metric['mae_min']:.3f} | "
                f"{metric['p90_abs_error_min']:.3f} | "
                f"{metric['within_5_min_pct']:.1f}% | "
                f"{metric['mae_delta_vs_incumbent_min']:+.3f} |\n"
            )

    print(
        json.dumps(
            {
                "best_candidate": best_candidate.name,
                "best_holdout_mae_min": best_holdout_metrics["mae_min"],
                "incumbent_mae_min": incumbent_mae,
                "promoted": should_promote,
                "metrics": args.output_metrics,
                "summary": args.output_summary,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--train-features",
        default="data/holdout/central_aug_train_20260709/features.parquet",
    )
    parser.add_argument(
        "--holdout-features",
        default="data/holdout/central_20260709/features.parquet",
    )
    parser.add_argument(
        "--train-graph-features",
        default="data/experiments/graph_hillclimb/train_graph_features.parquet",
    )
    parser.add_argument(
        "--holdout-graph-features",
        default="data/experiments/graph_hillclimb/holdout_graph_features.parquet",
    )
    parser.add_argument(
        "--current-model",
        default="models/travel_time_model_central_aug.joblib",
    )
    parser.add_argument(
        "--output-model",
        default="models/travel_time_hillclimb_best.joblib",
    )
    parser.add_argument(
        "--output-metrics",
        default="metrics/graph_hillclimb_20260709.json",
    )
    parser.add_argument(
        "--output-summary",
        default="docs/graph_hillclimb_results.md",
    )
    parser.add_argument("--refresh-graph-features", action="store_true")
    parser.add_argument("--tune-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--stack-alpha", type=float, default=0.01)
    parser.add_argument("--min-promote-delta-min", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_hillclimb(args)


if __name__ == "__main__":
    main()
