"""Geographic helpers used by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pyprojroot import here

EARTH_RADIUS_M = 6_371_000


@dataclass(frozen=True)
class BBox:
    north: float
    south: float
    west: float
    east: float


def haversine_m(
    lat1: np.ndarray | float,
    lng1: np.ndarray | float,
    lat2: np.ndarray | float,
    lng2: np.ndarray | float,
) -> np.ndarray:
    """Vectorized haversine distance in metres."""
    lat1_rad = np.radians(lat1)
    lng1_rad = np.radians(lng1)
    lat2_rad = np.radians(lat2)
    lng2_rad = np.radians(lng2)
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_degrees(
    lat1: np.ndarray | float,
    lng1: np.ndarray | float,
    lat2: np.ndarray | float,
    lng2: np.ndarray | float,
) -> np.ndarray:
    """Vectorized initial bearing in degrees from point 1 to point 2."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlng = np.radians(lng2) - np.radians(lng1)
    x = np.sin(dlng) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(dlng)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def read_station_catalog(path: str = "data/london_tubes.csv") -> pd.DataFrame:
    stations = pd.read_csv(here() / path)
    stations = stations.rename(
        columns={
            "NAME": "station_name",
            "LINES": "lines",
            "x": "lng",
            "y": "lat",
        }
    )
    stations = stations[["station_name", "lines", "lat", "lng"]].copy()
    stations["lines"] = stations["lines"].fillna("").astype(str)
    return stations


def line_vocab(stations: pd.DataFrame) -> list[str]:
    values: set[str] = set()
    for lines in stations["lines"].fillna(""):
        for line in str(lines).split(","):
            line = line.strip()
            if line and line.lower() != "n/a":
                values.add(line)
    return sorted(values)


def safe_feature_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in safe.split("_") if part)


def station_density(
    lat: np.ndarray,
    lng: np.ndarray,
    stations: pd.DataFrame,
    radius_m: float,
) -> np.ndarray:
    station_lats = stations["lat"].to_numpy()
    station_lngs = stations["lng"].to_numpy()
    densities = np.zeros(len(lat), dtype=float)
    for index, (point_lat, point_lng) in enumerate(zip(lat, lng)):
        distances = haversine_m(point_lat, point_lng, station_lats, station_lngs)
        densities[index] = float(np.sum(distances <= radius_m))
    return densities


def nearest_station_features(
    points: pd.DataFrame,
    stations: pd.DataFrame,
    prefix: str,
    n: int,
    lines: Sequence[str],
) -> pd.DataFrame:
    """Return nearest-station distances and line membership features."""
    station_lats = stations["lat"].to_numpy()
    station_lngs = stations["lng"].to_numpy()
    rows: list[dict[str, float | str]] = []
    for _, point in points.iterrows():
        distances = haversine_m(point["lat"], point["lng"], station_lats, station_lngs)
        nearest_indices = np.argsort(distances)[:n]
        row: dict[str, float | str] = {}
        nearest_lines: set[str] = set()
        for rank, station_index in enumerate(nearest_indices, start=1):
            station = stations.iloc[int(station_index)]
            row[f"{prefix}_station_{rank}_distance_m"] = float(
                distances[int(station_index)]
            )
            row[f"{prefix}_station_{rank}_name"] = str(station["station_name"])
            for line in str(station["lines"]).split(","):
                line = line.strip()
                if line and line.lower() != "n/a":
                    nearest_lines.add(line)
        for line in lines:
            row[f"{prefix}_line_{safe_feature_name(line)}"] = (
                1.0 if line in nearest_lines else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows, index=points.index)


def numeric_feature_columns(columns: Iterable[str]) -> list[str]:
    excluded_suffixes = ("_id", "_name", "_status")
    excluded_columns = {
        "target_travel_time_seconds",
        "travel_time_seconds",
        "reachable",
    }
    return [
        column
        for column in columns
        if not any(column.endswith(suffix) for suffix in excluded_suffixes)
        and column not in excluded_columns
    ]
