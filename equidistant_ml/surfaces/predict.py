"""Offline travel-time surface prediction."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from equidistant_ml.surfaces.config import load_params, project_path
from equidistant_ml.surfaces.features import build_feature_frame
from equidistant_ml.surfaces.geo import read_station_catalog
from equidistant_ml.surfaces.grid import bbox_from_params, build_destination_grid
from equidistant_ml.surfaces.models import (
    GroupCombine,
    ModelBundle,
    combine_surfaces,
    load_bundle,
    predict,
)


def default_destination_grid(
    params: dict, x_size: int | None, y_size: int | None
) -> pd.DataFrame:
    grid_params = params["grid"]
    return build_destination_grid(
        bbox_from_params(params),
        int(x_size or grid_params["x_size"]),
        int(y_size or grid_params["y_size"]),
    )


def _labels_for_prediction(origin_id: str, destinations: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_id": origin_id,
            "destination_id": destinations["destination_id"].astype(str),
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


def _load_best_available_model() -> ModelBundle | None:
    for path in ["models/travel_time_model.joblib", "models/baseline_model.joblib"]:
        resolved = project_path(path)
        if resolved.exists():
            return load_bundle(resolved)
    return None


def predict_origin_surface(
    origin_lat: float,
    origin_lng: float,
    *,
    origin_id: str = "origin_0",
    x_size: int | None = None,
    y_size: int | None = None,
) -> pd.DataFrame:
    params = load_params()
    destinations = default_destination_grid(params, x_size, y_size)
    origins = pd.DataFrame(
        [{"origin_id": origin_id, "lat": float(origin_lat), "lng": float(origin_lng)}]
    )
    labels = _labels_for_prediction(origin_id, destinations)
    stations = read_station_catalog(params["features"]["stations_path"])
    features = build_feature_frame(
        labels,
        origins,
        destinations,
        stations,
        nearest_station_count=int(params["features"]["nearest_station_count"]),
        density_radius_m=float(params["features"]["density_radius_m"]),
        include_target=False,
    )
    bundle = _load_best_available_model()
    predictions = predict(bundle, features) if bundle else _heuristic_seconds(features)
    result = destinations.copy()
    result["origin_id"] = origin_id
    result["travel_time_seconds"] = predictions
    result["travel_time_minutes"] = result["travel_time_seconds"] / 60
    return result


def predict_group_surface(
    friends: Iterable[dict],
    *,
    combine: GroupCombine = "balanced",
    x_size: int | None = None,
    y_size: int | None = None,
) -> pd.DataFrame:
    surfaces = []
    for index, friend in enumerate(friends):
        surface = predict_origin_surface(
            float(friend["lat"]),
            float(friend["lng"]),
            origin_id=f"friend_{index}",
            x_size=x_size,
            y_size=y_size,
        )
        surfaces.append(
            surface[
                [
                    "destination_id",
                    "lat",
                    "lng",
                    "x_index",
                    "y_index",
                    "travel_time_seconds",
                ]
            ].rename(columns={"travel_time_seconds": f"friend_{index}_seconds"})
        )
    combined = surfaces[0]
    for surface in surfaces[1:]:
        combined = combined.merge(
            surface.drop(columns=["lat", "lng", "x_index", "y_index"]),
            on="destination_id",
        )
    friend_columns = [
        column for column in combined.columns if column.endswith("_seconds")
    ]
    combined["score_seconds"] = combine_surfaces(
        combined[friend_columns],
        combine,
    )
    combined["score_minutes"] = combined["score_seconds"] / 60
    combined["max_seconds"] = combined[friend_columns].max(axis=1)
    combined["mean_seconds"] = combined[friend_columns].mean(axis=1)
    combined["fairness_seconds"] = combined[friend_columns].std(axis=1).fillna(0)
    return combined


def surface_to_grid_response(df: pd.DataFrame, value_column: str) -> dict:
    pivot = df.pivot(
        index="y_index", columns="x_index", values=value_column
    ).sort_index()
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
    return {
        "lats": lats,
        "lngs": lngs,
        "Z": pivot.to_numpy().round(3).tolist(),
        "cells": df.to_dict(orient="records"),
    }
