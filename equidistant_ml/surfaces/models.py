"""Training, evaluation, and offline surface prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from equidistant_ml.surfaces.features import feature_columns

GroupCombine = Literal["max", "mean", "fairness", "balanced"]


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    feature_columns: list[str]
    model_type: str


def split_by_origin(
    features: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    origin_ids = features["origin_id"].drop_duplicates().to_numpy()
    if len(origin_ids) < 2:
        return features.copy(), features.copy()
    train_ids, validation_ids = train_test_split(
        origin_ids,
        test_size=validation_fraction,
        random_state=seed,
    )
    train = features[features["origin_id"].isin(train_ids)].copy()
    validation = features[features["origin_id"].isin(validation_ids)].copy()
    return train, validation


def train_baseline(features: pd.DataFrame, params: Dict) -> ModelBundle:
    cols = feature_columns(features)
    selected = [
        column
        for column in cols
        if column
        in {
            "haversine_distance_m",
            "abs_delta_lat",
            "abs_delta_lng",
            "origin_station_1_distance_m",
            "destination_station_1_distance_m",
            "origin_station_density",
            "destination_station_density",
        }
    ]
    alpha = float(params.get("alpha", 1.0))
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(features[selected], features["target_travel_time_seconds"])
    return ModelBundle(
        model=model, feature_columns=selected, model_type="baseline_ridge"
    )


def train_lightgbm(features: pd.DataFrame, params: Dict) -> ModelBundle:
    cols = feature_columns(features)
    model_type = "lightgbm"
    try:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            objective="regression",
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            random_state=int(params.get("seed", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
            verbosity=-1,
        )
    except OSError:
        model_type = "hist_gradient_boosting_fallback"
        model = HistGradientBoostingRegressor(
            max_iter=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            random_state=int(params.get("seed", 42)),
            min_samples_leaf=int(params.get("min_child_samples", 20)),
        )
    wrapped = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    wrapped.fit(features[cols], features["target_travel_time_seconds"])
    return ModelBundle(model=wrapped, feature_columns=cols, model_type=model_type)


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "feature_columns": bundle.feature_columns,
            "model_type": bundle.model_type,
        },
        path,
    )


def load_bundle(path: str | Path) -> ModelBundle:
    artifact = joblib.load(path)
    return ModelBundle(
        model=artifact["model"],
        feature_columns=list(artifact["feature_columns"]),
        model_type=str(artifact["model_type"]),
    )


def predict(bundle: ModelBundle, features: pd.DataFrame) -> np.ndarray:
    missing = set(bundle.feature_columns) - set(features.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    values = bundle.model.predict(features[bundle.feature_columns])
    return np.maximum(values, 0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae_seconds": float(mean_absolute_error(y_true, y_pred)),
        "mae_minutes": float(mean_absolute_error(y_true, y_pred) / 60),
        "rmse_seconds": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "rmse_minutes": float((mean_squared_error(y_true, y_pred) ** 0.5) / 60),
        "median_absolute_error_seconds": float(np.median(np.abs(y_true - y_pred))),
        "median_absolute_error_minutes": float(np.median(np.abs(y_true - y_pred)) / 60),
        "r2": float(r2_score(y_true, y_pred)),
    }


def bucket_metrics(
    frame: pd.DataFrame, y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    working = frame.copy()
    working["prediction"] = y_pred
    working["abs_error"] = np.abs(
        working["target_travel_time_seconds"] - working["prediction"]
    )
    buckets: Dict[str, Dict[str, float]] = {}
    working["travel_time_bucket"] = pd.cut(
        working["target_travel_time_seconds"],
        bins=[0, 900, 1800, 2700, 3600, 5400, 7200, 10_800, np.inf],
        include_lowest=True,
    ).astype(str)
    working["station_access_bucket"] = pd.cut(
        working["origin_station_1_distance_m"],
        bins=[0, 300, 600, 1000, 1600, np.inf],
        include_lowest=True,
    ).astype(str)
    for column in ["travel_time_bucket", "station_access_bucket"]:
        grouped = working.groupby(column, observed=True)["abs_error"]
        buckets[column] = {
            key: float(value / 60) for key, value in grouped.mean().to_dict().items()
        }
    return buckets


def evaluate_models(
    features: pd.DataFrame,
    baseline: ModelBundle,
    model: ModelBundle,
    validation_fraction: float,
    seed: int,
) -> Dict:
    _, validation = split_by_origin(features, validation_fraction, seed)
    y_true = validation["target_travel_time_seconds"].to_numpy()
    baseline_pred = predict(baseline, validation)
    model_pred = predict(model, validation)
    baseline_metrics = regression_metrics(y_true, baseline_pred)
    model_metrics = regression_metrics(y_true, model_pred)
    improvement = 1 - model_metrics["mae_seconds"] / baseline_metrics["mae_seconds"]
    return {
        "baseline": baseline_metrics,
        "model": model_metrics,
        "baseline_improvement_pct": float(improvement * 100),
        "promising": bool(improvement >= 0.10),
        "validation_rows": int(len(validation)),
        "validation_origins": int(validation["origin_id"].nunique()),
        "bucketed_mae_minutes": bucket_metrics(validation, model_pred),
    }


def combine_surfaces(surface_columns: pd.DataFrame, mode: GroupCombine) -> pd.Series:
    if mode == "max":
        return surface_columns.max(axis=1)
    if mode == "mean":
        return surface_columns.mean(axis=1)
    if mode == "fairness":
        return surface_columns.std(axis=1).fillna(0)
    if mode == "balanced":
        return surface_columns.mean(axis=1) + 0.5 * surface_columns.std(axis=1).fillna(
            0
        )
    raise ValueError(f"Unsupported combine mode: {mode}")
