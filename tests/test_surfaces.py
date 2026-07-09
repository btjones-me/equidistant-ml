import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import equidistant_ml.app as app
from equidistant_ml.surfaces.features import build_feature_frame, feature_columns
from equidistant_ml.surfaces.geo import BBox, read_station_catalog
from equidistant_ml.surfaces.grid import (
    build_destination_grid,
    build_h3_destination_grid,
    build_smoke_labels,
    sample_origin_anchors,
)
from equidistant_ml.surfaces.models import (
    ModelBundle,
    column_bundle,
    evaluate_models,
    predict,
    train_baseline,
    train_graph_baseline,
    train_graph_residual_model,
    train_lightgbm,
    weighted_blend_bundle,
)
from equidistant_ml.surfaces.transport_graph import (
    add_graph_features,
    build_reference_from_station_catalog,
    build_transport_graph,
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


def test_h3_grid_uses_mixed_resolutions_and_boundaries():
    grid = build_h3_destination_grid(
        [
            {
                "id": "core",
                "label": "Core",
                "north": 51.52,
                "south": 51.50,
                "west": -0.14,
                "east": -0.10,
                "resolution": 9,
            },
            {
                "id": "context",
                "label": "Context",
                "north": 51.54,
                "south": 51.48,
                "west": -0.18,
                "east": -0.08,
                "resolution": 8,
            },
        ]
    )

    assert len(grid) > 0
    assert grid["destination_id"].is_unique
    assert set(grid["h3_resolution"]) == {8, 9}
    assert grid["boundary"].map(len).min() >= 6


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


def _small_graph_feature_frame():
    features = _small_feature_frame()
    stations = read_station_catalog()
    nodes, edges = build_reference_from_station_catalog(stations)
    graph = build_transport_graph(
        nodes,
        edges,
        transfer_radius_m=180,
        transfer_penalty_seconds=120,
        walking_speed_mps=1.35,
    )
    return add_graph_features(
        features,
        graph,
        walking_speed_mps=1.35,
        access_node_limit=4,
        max_access_distance_m=1600,
        bus_density_radius_m=500,
    )


def test_feature_builder_excludes_label_leakage():
    features = _small_feature_frame()
    cols = feature_columns(features)

    assert "haversine_distance_m" in cols
    assert "travel_time_seconds" not in cols
    assert "target_travel_time_seconds" not in cols
    assert features["origin_station_1_distance_m"].notna().all()


def test_transport_graph_adds_shortest_path_features():
    features = _small_graph_feature_frame()

    assert "graph_total_seconds" in features
    assert "graph_score_seconds" not in features
    assert features["graph_total_seconds"].notna().all()
    assert features["origin_nearest_heavy_rail_distance_m"].notna().all()
    assert features["graph_modes"].map(lambda value: isinstance(value, str)).all()


def test_graph_baseline_and_residual_model_predict_shape():
    features = _small_graph_feature_frame()
    baseline = train_graph_baseline(features, {"alpha": 1.0})
    model = train_graph_residual_model(
        features,
        baseline,
        {
            "n_estimators": 5,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 1,
            "seed": 1,
            "n_jobs": 1,
        },
    )
    predictions = predict(model, features.head(5))

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_weighted_blend_and_residual_bundle_prediction():
    features = pd.DataFrame(
        {
            "base_seconds": [600.0, 900.0],
            "graph_seconds": [500.0, 1000.0],
            "residual_feature": [1.0, 2.0],
        }
    )

    class ConstantResidual:
        def predict(self, frame):
            return np.full(len(frame), -60.0)

    blend = weighted_blend_bundle(
        [
            ("base", 0.75, column_bundle("base_seconds")),
            ("graph", 0.25, column_bundle("graph_seconds")),
        ]
    )
    residual = ModelBundle(
        model={
            "base_bundle": column_bundle("base_seconds"),
            "residual_regressor": ConstantResidual(),
            "residual_feature_columns": ["residual_feature"],
            "cap_seconds": 10_000.0,
        },
        feature_columns=["base_seconds", "residual_feature"],
        model_type="residual_over_base_lightgbm",
    )

    assert np.allclose(predict(blend, features), [575.0, 925.0])
    assert np.allclose(predict(residual, features), [540.0, 840.0])


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


def test_group_surface_api_returns_h3_layers():
    client = TestClient(app.app)
    response = client.post(
        "/api/group-surface",
        json={
            "friends": [
                {"lat": 51.51, "lng": -0.13, "name": "A"},
                {"lat": 51.53, "lng": -0.18, "name": "B"},
            ],
            "combine": "balanced",
            "grid_mode": "h3",
            "focus": "central",
            "detail": "fast",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["grid_type"] == "h3"
    assert len(body["cells"]) > 0
    assert body["Z"] == []
    first = body["cells"][0]
    assert "boundary" in first
    assert "friend_0_minutes" in first
    assert "friend_1_minutes" in first
    assert "friend_0_model_minutes" in first
    assert "friend_1_model_minutes" in first
    assert "model_score_minutes" in first
    assert "graph_score_minutes" in first
    assert "model_residual_minutes" in first
    assert "graph_total_seconds" not in first
    assert body["metadata"]["source"] == "api"
    assert response.headers["content-encoding"] == "gzip"


def test_group_surface_api_respects_included_friend_indexes():
    client = TestClient(app.app)
    response = client.post(
        "/api/group-surface",
        json={
            "friends": [
                {"lat": 51.551808, "lng": -0.195603, "name": "Alex"},
                {"lat": 51.53, "lng": -0.18, "name": "B"},
            ],
            "combine": "balanced",
            "included_friend_indexes": [0],
            "grid_mode": "h3",
            "focus": "central",
            "detail": "fast",
        },
    )

    assert response.status_code == 200
    body = response.json()
    first = body["cells"][0]
    assert np.isclose(first["model_score_minutes"], first["friend_0_model_minutes"])
    assert "friend_1_model_minutes" in first


def test_comparison_surface_endpoint_returns_reference_metrics(monkeypatch):
    calls = []

    def fake_reference_group_surface(*args, **kwargs):
        calls.append(kwargs)
        frame = pd.DataFrame(
            [
                {
                    "destination_id": "d_1",
                    "lat": 51.51,
                    "lng": -0.13,
                    "x_index": 0,
                    "y_index": 0,
                    "h3_cell": "d_1",
                    "h3_resolution": 9,
                    "boundary": [
                        [51.51, -0.13],
                        [51.511, -0.13],
                        [51.511, -0.129],
                    ],
                    "grid_band": "Zone 1 core",
                    "grid_priority": 0,
                    "friend_0_model_minutes": 12.0,
                    "friend_1_model_minutes": 18.0,
                    "model_score_minutes": 12.0,
                    "friend_0_reference_minutes": 14.0,
                    "friend_0_reference_seconds": 840.0,
                    "friend_0_reference_reachable": True,
                    "reference_score_minutes": 14.0,
                    "signed_error_minutes": -2.0,
                    "abs_error_minutes": 2.0,
                },
                {
                    "destination_id": "d_2",
                    "lat": 51.512,
                    "lng": -0.132,
                    "x_index": 1,
                    "y_index": 0,
                    "h3_cell": "d_2",
                    "h3_resolution": 9,
                    "boundary": [
                        [51.512, -0.132],
                        [51.513, -0.132],
                        [51.513, -0.131],
                    ],
                    "grid_band": "Zone 1 core",
                    "grid_priority": 0,
                    "friend_0_model_minutes": 20.0,
                    "friend_1_model_minutes": 25.0,
                    "model_score_minutes": 20.0,
                    "friend_0_reference_minutes": 21.0,
                    "friend_0_reference_seconds": 1260.0,
                    "friend_0_reference_reachable": True,
                    "reference_score_minutes": 21.0,
                    "signed_error_minutes": -1.0,
                    "abs_error_minutes": 1.0,
                },
            ]
        )
        return frame, {
            "reference_origin_count": 1,
            "included_friend_indexes": [0],
            "mae_minutes": 1.5,
            "median_abs_error_minutes": 1.5,
            "p90_abs_error_minutes": 1.9,
            "within_5_min_pct": 100.0,
            "within_10_min_pct": 100.0,
            "mean_signed_error_minutes": -1.5,
        }

    monkeypatch.setattr(app, "reference_group_surface", fake_reference_group_surface)
    client = TestClient(app.app)
    response = client.post(
        "/api/comparison-surface",
        json={
            "friends": [
                {"lat": 51.551808, "lng": -0.195603, "name": "Alex"},
                {"lat": 51.53, "lng": -0.18, "name": "B"},
            ],
            "combine": "balanced",
            "included_friend_indexes": [0],
            "focus": "central",
            "detail": "fast",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["grid_type"] == "h3"
    assert body["metadata"]["comparison"]["mae_minutes"] == 1.5
    assert body["metadata"]["value_columns"]["reference"] == "reference_score_minutes"
    assert body["metadata"]["value_columns"]["graph"] == "graph_score_minutes"
    assert body["metadata"]["value_columns"]["error"] == "signed_error_minutes"
    assert "reference_score_minutes" in body["cells"][0]
    assert "signed_error_minutes" in body["cells"][0]
    assert "abs_error_minutes" in body["cells"][0]
    assert calls[0]["grid_mode"] == "h3"
