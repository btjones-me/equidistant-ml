"""Destination grid and origin-anchor sampling."""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from equidistant_ml.surfaces.geo import BBox, haversine_m


def bbox_from_params(params: dict) -> BBox:
    bbox = params["bbox"]
    return BBox(
        north=float(bbox["north"]),
        south=float(bbox["south"]),
        west=float(bbox["west"]),
        east=float(bbox["east"]),
    )


def build_destination_grid(bbox: BBox, x_size: int, y_size: int) -> pd.DataFrame:
    lats = np.linspace(bbox.north, bbox.south, y_size)
    lngs = np.linspace(bbox.west, bbox.east, x_size)
    rows = []
    for y_index, lat in enumerate(lats):
        for x_index, lng in enumerate(lngs):
            rows.append(
                {
                    "destination_id": f"d_{y_index:03d}_{x_index:03d}",
                    "lat": round(float(lat), 7),
                    "lng": round(float(lng), 7),
                    "x_index": x_index,
                    "y_index": y_index,
                }
            )
    return pd.DataFrame(rows)


def _point_in_bbox(lat: float, lng: float, bbox: BBox) -> bool:
    return bbox.south <= lat <= bbox.north and bbox.west <= lng <= bbox.east


def build_h3_destination_grid(band_params: list[dict]) -> pd.DataFrame:
    """Build a mixed-resolution H3 grid from priority-ordered bbox bands."""
    rows: list[dict] = []
    occupied: list[BBox] = []
    seen_cells: set[str] = set()
    occupied_parent_cells: set[tuple[int, str]] = set()
    for priority, band in enumerate(band_params):
        bbox = BBox(
            north=float(band["north"]),
            south=float(band["south"]),
            west=float(band["west"]),
            east=float(band["east"]),
        )
        resolution = int(band["resolution"])
        band_id = str(band["id"])
        label = str(band.get("label", band_id))
        buffer_degrees = float(band.get("buffer_degrees", 0.0))
        fill_bbox = BBox(
            north=bbox.north + buffer_degrees,
            south=bbox.south - buffer_degrees,
            west=bbox.west - buffer_degrees,
            east=bbox.east + buffer_degrees,
        )
        polygon = h3.LatLngPoly(
            [
                (fill_bbox.south, fill_bbox.west),
                (fill_bbox.south, fill_bbox.east),
                (fill_bbox.north, fill_bbox.east),
                (fill_bbox.north, fill_bbox.west),
            ]
        )
        for cell in sorted(h3.h3shape_to_cells(polygon, resolution)):
            if cell in seen_cells:
                continue
            if (resolution, cell) in occupied_parent_cells:
                continue
            lat, lng = h3.cell_to_latlng(cell)
            if any(_point_in_bbox(lat, lng, previous) for previous in occupied):
                continue
            boundary = [
                [round(float(point_lat), 7), round(float(point_lng), 7)]
                for point_lat, point_lng in h3.cell_to_boundary(cell)
            ]
            rows.append(
                {
                    "destination_id": cell,
                    "lat": round(float(lat), 7),
                    "lng": round(float(lng), 7),
                    "x_index": len(rows),
                    "y_index": 0,
                    "h3_cell": cell,
                    "h3_resolution": resolution,
                    "boundary": boundary,
                    "grid_band": label,
                    "grid_priority": priority,
                    "cell_area_km2": round(float(h3.cell_area(cell, unit="km^2")), 6),
                }
            )
            seen_cells.add(cell)
            for parent_resolution in range(resolution):
                occupied_parent_cells.add(
                    (parent_resolution, h3.cell_to_parent(cell, parent_resolution))
                )
        occupied.append(bbox)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["grid_priority", "h3_resolution", "destination_id"]
    ).reset_index(drop=True)
    result["cell_index"] = result.index
    return result


def sample_origin_anchors(
    bbox: BBox,
    stations: pd.DataFrame,
    count: int,
    seed: int,
    station_bias_fraction: float = 0.35,
) -> pd.DataFrame:
    """Sample origins from uniform London coverage plus station-biased jitter."""
    rng = np.random.default_rng(seed)
    station_count = min(int(round(count * station_bias_fraction)), count)
    uniform_count = count - station_count

    uniform = pd.DataFrame(
        {
            "lat": rng.uniform(bbox.south, bbox.north, uniform_count),
            "lng": rng.uniform(bbox.west, bbox.east, uniform_count),
            "sample_strategy": "uniform",
        }
    )

    station_indices = rng.choice(len(stations), size=station_count, replace=True)
    station_rows = stations.iloc[station_indices].reset_index(drop=True)
    # 0.006 degrees is roughly 650m latitude in London; enough to cover access areas.
    station = pd.DataFrame(
        {
            "lat": np.clip(
                station_rows["lat"].to_numpy() + rng.normal(0, 0.006, station_count),
                bbox.south,
                bbox.north,
            ),
            "lng": np.clip(
                station_rows["lng"].to_numpy() + rng.normal(0, 0.009, station_count),
                bbox.west,
                bbox.east,
            ),
            "sample_strategy": "station_jitter",
        }
    )
    origins = pd.concat([uniform, station], ignore_index=True)
    origins = origins.sample(frac=1, random_state=seed).reset_index(drop=True)
    origins.insert(0, "origin_id", [f"o_{index:04d}" for index in range(len(origins))])
    origins["lat"] = origins["lat"].round(7)
    origins["lng"] = origins["lng"].round(7)
    return origins


def nearest_grid_origin_split(
    origins: pd.DataFrame, validation_fraction: float
) -> pd.Series:
    """Deterministic spatial-ish holdout by sorting origins north-west to south-east."""
    ordered = origins.sort_values(
        ["lat", "lng"], ascending=[False, True]
    ).index.to_numpy()
    holdout_count = max(1, int(round(len(origins) * validation_fraction)))
    holdout_indices = set(
        ordered[:: max(1, len(origins) // holdout_count)][:holdout_count]
    )
    return pd.Series(
        np.where(origins.index.isin(holdout_indices), "spatial_validation", "train"),
        index=origins.index,
    )


def build_smoke_labels(
    origins: pd.DataFrame, destinations: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for _, origin in origins.iterrows():
        distances = haversine_m(
            origin["lat"],
            origin["lng"],
            destinations["lat"].to_numpy(),
            destinations["lng"].to_numpy(),
        )
        for destination, distance_m in zip(
            destinations.itertuples(index=False), distances
        ):
            travel_time = 600 + distance_m / 7.2
            rows.append(
                {
                    "origin_id": origin["origin_id"],
                    "destination_id": destination.destination_id,
                    "travel_time_seconds": round(float(travel_time), 2),
                    "target_travel_time_seconds": round(float(travel_time), 2),
                    "reachable": True,
                    "api_status": "MOCK",
                }
            )
    return pd.DataFrame(rows)
