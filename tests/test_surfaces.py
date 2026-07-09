import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import equidistant_ml.app as app
from equidistant_ml.surfaces.features import build_feature_frame, feature_columns
from equidistant_ml.surfaces.geo import BBox, read_station_catalog
from equidistant_ml.surfaces.grid import (
    build_destination_grid,
    build_smoke_labels,
    sample_origin_anchors,
)
from equidistant_ml.surfaces.models import (
    evaluate_models,
    train_baseline,
    train_lightgbm,
)
from equidistant_ml.surfaces.traveltime import (
    TravelTimeClient,
    fetch_origin_surfaces,
    parse_fast_matrix_response,
)


def test_grid_and_origin_sampling_shapes():
    bbox = BBox(north=51.55, south=51.50, west=-0.20, east=-0.10)
    grid = build_destination_grid(bbox, x_size=4, y_size=3)
    stations = read_station_catalog()
    origins = sample_origin_anchors(bbox, stations, count=10, seed=1)

    assert len(grid) == 12
    assert len(origins) == 10
    assert grid["destination_id"].is_unique
    assert origins["origin_id"].is_unique
    assert origins["lat"].between(bbox.south, bbox.north).all()
    assert origins["lng"].between(bbox.west, bbox.east).all()


def test_traveltime_payload_and_response_parsing():
    origin = pd.Series({"origin_id": "o_0001", "lat": 51.51, "lng": -0.13})
    destinations = pd.DataFrame(
        [
            {"destination_id": "d_1", "lat": 51.52, "lng": -0.14},
            {"destination_id": "d_2", "lat": 51.53, "lng": -0.15},
        ]
    )
    payload = TravelTimeClient.build_one_to_many_payload(
        origin,
        destinations,
        transportation_type="public_transport",
        arrival_time_period="weekday_morning",
        travel_time_seconds=3600,
        properties=["travel_time"],
    )

    search = payload["arrival_searches"]["one_to_many"][0]
    assert search["departure_location_id"] == "o_0001"
    assert search["arrival_location_ids"] == ["d_1", "d_2"]
    assert search["transportation"] == {"type": "public_transport"}

    parsed = parse_fast_matrix_response(
        "o_0001",
        destinations,
        {
            "results": [
                {
                    "search_id": "o_0001",
                    "locations": [
                        {"id": "d_1", "properties": {"travel_time": 1234}},
                    ],
                    "unreachable": ["d_2"],
                }
            ]
        },
        travel_time_limit_seconds=3600,
        unreachable_penalty_seconds=600,
    )

    assert (
        bool(parsed.loc[parsed["destination_id"] == "d_1", "reachable"].item()) is True
    )
    assert (
        bool(parsed.loc[parsed["destination_id"] == "d_2", "reachable"].item()) is False
    )
    assert (
        parsed.loc[
            parsed["destination_id"] == "d_2", "target_travel_time_seconds"
        ].item()
        == 4200
    )


def test_traveltime_fetch_reuses_origin_checkpoints(tmp_path):
    origin = pd.DataFrame([{"origin_id": "o_0001", "lat": 51.51, "lng": -0.13}])
    destinations = pd.DataFrame(
        [
            {"destination_id": "d_1", "lat": 51.52, "lng": -0.14},
            {"destination_id": "d_2", "lat": 51.53, "lng": -0.15},
        ]
    )

    class FakeClient:
        calls = 0

        def build_one_to_many_payload(self, *args, **kwargs):
            return TravelTimeClient.build_one_to_many_payload(*args, **kwargs)

        def post_fast_matrix(self, payload):
            self.calls += 1
            return {
                "results": [
                    {
                        "search_id": "o_0001",
                        "locations": [
                            {"id": "d_1", "properties": {"travel_time": 1000}},
                            {"id": "d_2", "properties": {"travel_time": 1100}},
                        ],
                        "unreachable": [],
                    }
                ]
            }

    client = FakeClient()
    checkpoint_dir = tmp_path / "parts"
    first = fetch_origin_surfaces(
        origin,
        destinations,
        client,
        transportation_type="public_transport",
        arrival_time_period="weekday_morning",
        travel_time_seconds=3600,
        unreachable_penalty_seconds=600,
        properties=["travel_time"],
        checkpoint_dir=checkpoint_dir,
    )
    second = fetch_origin_surfaces(
        origin,
        destinations,
        client,
        transportation_type="public_transport",
        arrival_time_period="weekday_morning",
        travel_time_seconds=3600,
        unreachable_penalty_seconds=600,
        properties=["travel_time"],
        checkpoint_dir=checkpoint_dir,
    )

    assert client.calls == 1
    assert len(first) == 2
    assert second["target_travel_time_seconds"].tolist() == [1000.0, 1100.0]


def _small_feature_frame():
    bbox = BBox(north=51.55, south=51.50, west=-0.20, east=-0.10)
    grid = build_destination_grid(bbox, x_size=4, y_size=3)
    stations = read_station_catalog()
    origins = sample_origin_anchors(bbox, stations, count=8, seed=2)
    labels = build_smoke_labels(origins, grid)
    return build_feature_frame(
        labels,
        origins,
        grid,
        stations,
        nearest_station_count=3,
        density_radius_m=1200,
    )


def test_feature_builder_excludes_label_leakage():
    features = _small_feature_frame()
    cols = feature_columns(features)

    assert "haversine_distance_m" in cols
    assert "travel_time_seconds" not in cols
    assert "target_travel_time_seconds" not in cols
    assert features["origin_station_1_distance_m"].notna().all()


def test_baseline_model_training_and_metrics():
    features = _small_feature_frame()
    baseline = train_baseline(features, {"alpha": 1.0})
    model = train_lightgbm(
        features,
        {
            "n_estimators": 5,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 1,
            "seed": 1,
            "n_jobs": 1,
        },
    )
    metrics = evaluate_models(
        features,
        baseline,
        model,
        validation_fraction=0.25,
        seed=1,
    )

    assert metrics["validation_rows"] > 0
    assert "mae_minutes" in metrics["model"]
    assert np.isfinite(metrics["model"]["mae_minutes"])


def test_group_surface_api_returns_grid():
    client = TestClient(app.app)
    response = client.post(
        "/api/group-surface",
        json={
            "friends": [
                {"lat": 51.51, "lng": -0.13, "name": "A"},
                {"lat": 51.53, "lng": -0.18, "name": "B"},
            ],
            "combine": "balanced",
            "x_size": 5,
            "y_size": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["lats"]) == 5
    assert len(body["lngs"]) == 5
    assert len(body["Z"]) == 5
    assert len(body["Z"][0]) == 5
