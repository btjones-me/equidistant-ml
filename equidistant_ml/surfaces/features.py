"""Feature engineering for public-transport travel-time modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equidistant_ml.surfaces.geo import (
    bearing_degrees,
    haversine_m,
    line_vocab,
    nearest_station_features,
    numeric_feature_columns,
    safe_feature_name,
    station_density,
)


def build_feature_frame(
    labels: pd.DataFrame,
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    nearest_station_count: int,
    density_radius_m: float,
    include_target: bool = True,
) -> pd.DataFrame:
    merged = labels.merge(origins, on="origin_id", suffixes=("", "_origin"))
    merged = merged.merge(
        destinations,
        on="destination_id",
        suffixes=("_origin", "_destination"),
    )
    merged = merged.rename(
        columns={
            "lat_origin": "origin_lat",
            "lng_origin": "origin_lng",
            "lat_destination": "destination_lat",
            "lng_destination": "destination_lng",
        }
    )

    merged["haversine_distance_m"] = haversine_m(
        merged["origin_lat"].to_numpy(),
        merged["origin_lng"].to_numpy(),
        merged["destination_lat"].to_numpy(),
        merged["destination_lng"].to_numpy(),
    )
    bearing = bearing_degrees(
        merged["origin_lat"].to_numpy(),
        merged["origin_lng"].to_numpy(),
        merged["destination_lat"].to_numpy(),
        merged["destination_lng"].to_numpy(),
    )
    merged["bearing_sin"] = np.sin(np.radians(bearing))
    merged["bearing_cos"] = np.cos(np.radians(bearing))
    merged["abs_delta_lat"] = np.abs(merged["destination_lat"] - merged["origin_lat"])
    merged["abs_delta_lng"] = np.abs(merged["destination_lng"] - merged["origin_lng"])

    lines = line_vocab(stations)
    origin_points = (
        merged[["origin_id", "origin_lat", "origin_lng"]]
        .drop_duplicates("origin_id")
        .rename(columns={"origin_lat": "lat", "origin_lng": "lng"})
        .reset_index(drop=True)
    )
    destination_points = (
        merged[["destination_id", "destination_lat", "destination_lng"]]
        .drop_duplicates("destination_id")
        .rename(columns={"destination_lat": "lat", "destination_lng": "lng"})
        .reset_index(drop=True)
    )
    origin_station_features = nearest_station_features(
        origin_points[["lat", "lng"]], stations, "origin", nearest_station_count, lines
    )
    origin_station_features.insert(0, "origin_id", origin_points["origin_id"])
    origin_station_features["origin_station_density"] = station_density(
        origin_points["lat"].to_numpy(),
        origin_points["lng"].to_numpy(),
        stations,
        density_radius_m,
    )
    destination_station_features = nearest_station_features(
        destination_points[["lat", "lng"]],
        stations,
        "destination",
        nearest_station_count,
        lines,
    )
    destination_station_features.insert(
        0, "destination_id", destination_points["destination_id"]
    )
    destination_station_features["destination_station_density"] = station_density(
        destination_points["lat"].to_numpy(),
        destination_points["lng"].to_numpy(),
        stations,
        density_radius_m,
    )

    feature_frame = merged.merge(origin_station_features, on="origin_id").merge(
        destination_station_features,
        on="destination_id",
    )
    feature_frame["origin_bus_density"] = 0.0
    feature_frame["destination_bus_density"] = 0.0
    for line in lines:
        safe_line = safe_feature_name(line)
        feature_frame[f"same_nearest_line_{safe_line}"] = (
            (feature_frame[f"origin_line_{safe_line}"] > 0)
            & (feature_frame[f"destination_line_{safe_line}"] > 0)
        ).astype(float)

    feature_frame["reachable"] = feature_frame.get("reachable", True).astype(bool)
    if not include_target and "target_travel_time_seconds" in feature_frame:
        feature_frame = feature_frame.drop(columns=["target_travel_time_seconds"])
    return feature_frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = numeric_feature_columns(frame.columns)
    return [column for column in cols if pd.api.types.is_numeric_dtype(frame[column])]
