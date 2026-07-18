import json

import h3
import numpy as np
import pandas as pd
import pytest
import requests

from equidistant_ml.surfaces.config import load_params
from equidistant_ml.surfaces.expanded_atlas import (
    _adaptive_anchors_pass_validation,
    _atlas_cell_records,
    _select_adaptive_anchors,
    _select_interpolation_neighbours,
    adaptive_atlas_probe_origins,
    expanded_atlas_origins,
)
from equidistant_ml.surfaces.expanded_run import (
    _training_params_sha256,
    expanded_destination_grid,
    sample_expanded_origins,
    validate_graph_coverage,
)
from equidistant_ml.surfaces.predict import (
    _validate_supported_origins,
    h3_destination_grid,
)
from equidistant_ml.surfaces.traveltime import (
    TravelTimeClient,
    TravelTimeCredentials,
    fetch_origin_surfaces,
)


class FakeTravelTimeClient:
    def __init__(self):
        self.calls = 0

    def build_one_to_many_payload(self, *args, **kwargs):
        return TravelTimeClient.build_one_to_many_payload(*args, **kwargs)

    def post_fast_matrix(self, payload):
        self.calls += 1
        search = payload["arrival_searches"]["one_to_many"][0]
        return {
            "results": [
                {
                    "search_id": search["id"],
                    "locations": [
                        {"id": destination_id, "properties": {"travel_time": 900}}
                        for destination_id in search["arrival_location_ids"]
                    ],
                    "unreachable": [],
                }
            ]
        }


def test_traveltime_client_retries_transient_network_errors(monkeypatch):
    attempts = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    def fake_post(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise requests.ConnectionError("temporary DNS failure")
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("equidistant_ml.surfaces.traveltime.time.sleep", lambda _: None)
    client = TravelTimeClient(
        TravelTimeCredentials("app", "key"), sleep_seconds=0, max_retries=1
    )

    assert client.post_fast_matrix({}) == {"results": []}
    assert len(attempts) == 2


def _fetch_with_run(origin, destinations, client, checkpoint_dir):
    return fetch_origin_surfaces(
        origin,
        destinations,
        client,
        transportation_type="public_transport",
        arrival_time_period="weekday_morning",
        travel_time_seconds=10_800,
        unreachable_penalty_seconds=1_800,
        properties=["travel_time"],
        checkpoint_dir=checkpoint_dir,
        run_id="test_run",
    )


def test_fingerprinted_checkpoint_exact_resume_and_mismatch_rejection(tmp_path):
    origins = pd.DataFrame([{"origin_id": "origin_1", "lat": 51.51, "lng": -0.13}])
    destinations = pd.DataFrame(
        [
            {"destination_id": "d1", "lat": 51.52, "lng": -0.14},
            {"destination_id": "d2", "lat": 51.53, "lng": -0.15},
        ]
    )
    client = FakeTravelTimeClient()
    first = _fetch_with_run(origins, destinations, client, tmp_path)
    resumed = _fetch_with_run(origins, destinations, client, tmp_path)

    assert client.calls == 1
    assert first.equals(resumed)
    assert not list(tmp_path.glob("*.tmp.parquet"))

    changed = destinations.copy()
    changed.loc[0, "lat"] += 0.001
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _fetch_with_run(origins, changed, client, tmp_path)


def test_fingerprinted_checkpoint_refuses_incomplete_shard(tmp_path):
    origins = pd.DataFrame([{"origin_id": "origin_1", "lat": 51.51, "lng": -0.13}])
    destinations = pd.DataFrame(
        [
            {"destination_id": "d1", "lat": 51.52, "lng": -0.14},
            {"destination_id": "d2", "lat": 51.53, "lng": -0.15},
        ]
    )
    client = FakeTravelTimeClient()
    _fetch_with_run(origins, destinations, client, tmp_path)
    shard_path = next(tmp_path.glob("*.parquet"))
    pd.read_parquet(shard_path).head(1).to_parquet(shard_path, index=False)

    with pytest.raises(ValueError, match="destinations mismatch"):
        _fetch_with_run(origins, destinations, client, tmp_path)


def test_expanded_grid_preserves_original_cells_without_parent_overlap():
    grid = expanded_destination_grid(load_params())
    cells = set(grid["destination_id"])

    assert len(grid) == 3538
    assert (grid["coverage_region"] == "original").sum() == 3032
    assert set(grid["h3_resolution"]) == {8, 9}
    for cell in cells:
        for parent_resolution in range(h3.get_resolution(cell)):
            assert h3.cell_to_parent(cell, parent_resolution) not in cells


def test_wide_direct_inference_uses_the_same_hierarchical_grid():
    grid = h3_destination_grid(load_params(), focus="wide", detail="fine")

    assert len(grid) == 3538
    assert (grid["coverage_region"] == "original").sum() == 3032


def test_wide_direct_inference_rejects_out_of_bounds_participants():
    params = load_params()
    outside = pd.DataFrame([{"lat": 51.6, "lng": -0.1}])

    with pytest.raises(ValueError, match="outside the coverage area"):
        _validate_supported_origins(outside, params, "h3", "wide")


def test_origin_splits_are_exact_and_h3_block_disjoint():
    params = load_params()
    run = params["expanded_run"]
    bounds = run["expanded_bounds"]
    polygon = h3.LatLngPoly(
        [
            (bounds["south"], bounds["west"]),
            (bounds["south"], bounds["east"]),
            (bounds["north"], bounds["east"]),
            (bounds["north"], bounds["west"]),
        ]
    )
    blocks = h3.h3shape_to_cells_experimental(
        polygon, run["split_h3_resolution"], contain="overlap"
    )
    nodes = pd.DataFrame(
        [
            {
                "node_id": cell,
                "name": f"Station {cell}",
                "mode": "tube",
                "lat": h3.cell_to_latlng(cell)[0],
                "lng": h3.cell_to_latlng(cell)[1],
                "lines": "Test",
            }
            for cell in blocks
        ]
    )
    origins = sample_expanded_origins(params, nodes)

    assert len(origins) == 560
    assert origins.groupby(["split", "coverage_region"]).size().to_dict() == {
        ("test", "original"): 48,
        ("test", "outer"): 48,
        ("train", "original"): 320,
        ("train", "outer"): 96,
        ("tune", "original"): 32,
        ("tune", "outer"): 16,
    }
    split_blocks = {
        split: set(origins.loc[origins["split"] == split, "origin_block"])
        for split in ("train", "tune", "test")
    }
    assert split_blocks["train"].isdisjoint(split_blocks["tune"])
    assert split_blocks["train"].isdisjoint(split_blocks["test"])
    assert split_blocks["tune"].isdisjoint(split_blocks["test"])


def test_training_lineage_ignores_atlas_only_tuning():
    params = load_params()
    original_hash = _training_params_sha256(params)
    params["expanded_run"]["atlas"]["adaptive_anchor_count"] += 50

    assert _training_params_sha256(params) == original_hash

    params["expanded_run"]["sampling"]["train"]["outer"] += 1
    assert _training_params_sha256(params) != original_hash


def test_graph_coverage_requires_connected_national_rail_endpoints():
    nodes = pd.DataFrame(
        [
            {
                "node_id": "harrow",
                "name": "Harrow-on-the-Hill",
                "mode": "tube",
                "lines": "Metropolitan, Chiltern",
            },
            {
                "node_id": "sidcup",
                "name": "Sidcup Rail Station",
                "mode": "national_rail",
                "lines": "Southeastern",
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "from_node_id": "harrow",
                "to_node_id": "sidcup",
                "mode": "national_rail",
                "line": "test",
            }
        ]
    )

    report = validate_graph_coverage(nodes, edges, ["Harrow-on-the-Hill", "Sidcup"])
    assert report["required_stations_connected"] is True
    assert all(report["connection_catalogue"].values())

    with pytest.raises(ValueError, match="non-empty"):
        validate_graph_coverage(
            nodes, edges.iloc[0:0], ["Harrow-on-the-Hill", "Sidcup"]
        )


def test_candidate_atlas_anchor_density_matches_v2_plan():
    origins = expanded_atlas_origins(load_params())

    assert len(origins) == 765
    assert origins["origin_id"].is_unique
    assert (origins["anchor_region"] == "central").sum() == 560
    assert (origins["anchor_region"] == "outer").sum() == 205


def test_candidate_atlas_serialises_h3_boundaries_as_coordinate_arrays():
    boundary = np.array(
        [
            np.array([51.5, -0.1]),
            np.array([51.51, -0.09]),
            np.array([51.49, -0.08]),
        ],
        dtype=object,
    )
    destinations = pd.DataFrame(
        [
            {
                "destination_id": "cell",
                "lat": 51.5,
                "lng": -0.09,
                "boundary": boundary,
                "h3_cell": "cell",
                "h3_resolution": 9,
                "grid_band": "Zone 1 core",
                "grid_priority": 0,
                "coverage_region": "original",
                "cell_area_km2": 0.1,
                "nearest_station_name": "Test",
                "nearest_station_lines": "Test line",
            }
        ]
    )

    records = _atlas_cell_records(destinations)

    assert records[0]["boundary"] == [
        [51.5, -0.1],
        [51.51, -0.09],
        [51.49, -0.08],
    ]
    assert "array(" not in json.dumps(records)


def test_adaptive_atlas_probes_are_disjoint_and_outer_only():
    params = load_params()
    probes = adaptive_atlas_probe_origins(params)
    atlas = params["expanded_run"]["atlas"]

    assert probes.groupby("probe_role").size().to_dict() == {
        "selection": 350,
        "validation": 350,
    }
    assert probes["origin_id"].is_unique
    assert not (
        probes["lat"].between(atlas["inner_south"], atlas["inner_north"])
        & probes["lng"].between(atlas["inner_west"], atlas["inner_east"])
    ).any()


def test_discontinuity_aware_interpolation_drops_a_lone_regime():
    selected = _select_interpolation_neighbours(
        np.array([350.0, 950.0, 1145.0, 1415.0]),
        4,
        signatures=np.array([181.0, 67.0, 58.0, 66.0]),
        discontinuity_threshold_minutes=20.0,
    )

    assert selected.tolist() == [2, 3, 1]


def test_adaptive_anchor_selection_respects_error_and_spacing():
    probes = pd.DataFrame(
        [
            {"origin_id": "one", "lat": 51.44, "lng": -0.25},
            {"origin_id": "two", "lat": 51.4401, "lng": -0.2501},
            {"origin_id": "three", "lat": 51.45, "lng": -0.20},
        ]
    )
    direct = np.array([[100, 100], [95, 95], [20, 20]], dtype=np.uint8)
    interpolated = np.zeros((3, 2), dtype=np.float32)

    anchors, indexes = _select_adaptive_anchors(
        probes,
        direct,
        interpolated,
        count=2,
        min_separation_m=250,
        min_mae_minutes=10,
    )

    assert indexes.tolist() == [0, 2]
    assert anchors["anchor_source"].eq("adaptive_probe").all()


def test_adaptive_anchors_must_generalise_on_disjoint_probes():
    before = {"mae_minutes": 4.6, "p90_absolute_error_minutes": 5.0}

    assert _adaptive_anchors_pass_validation(
        before, {"mae_minutes": 4.2, "p90_absolute_error_minutes": 4.9}
    )
    assert not _adaptive_anchors_pass_validation(
        before, {"mae_minutes": 5.0, "p90_absolute_error_minutes": 4.8}
    )
    assert not _adaptive_anchors_pass_validation(
        before, {"mae_minutes": 4.2, "p90_absolute_error_minutes": 5.1}
    )
