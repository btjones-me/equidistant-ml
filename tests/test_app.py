"""Tests for `equidistant_ml` app."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyprojroot import here

import equidistant_ml.app as app

client = TestClient(app.app)


@pytest.fixture
def payload():
    with open(
        Path(here() / "tests" / "test_payloads" / "1.json"), mode="r"
    ) as f:  # dummy in place
        payload = json.load(f)
    return payload


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "equidistant-api"


def test_predict(payload):
    response = client.post(
        "/predict",
        json=payload,
    )
    assert response.status_code == 200

    result = response.json()

    assert len(result["lats"]) == payload["y_size"]
    assert len(result["lngs"]) == payload["x_size"]
    assert len(result["Z"]) == payload["y_size"]
    assert len(result["Z"][0]) == payload["x_size"]
    assert np.mean(result["Z"][0]) != 0


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "equidistant-api",
        "model": "offline-public-transport",
    }


def test_geocode_endpoint_returns_safe_shape(monkeypatch):
    monkeypatch.setattr(
        app,
        "_geocode_london",
        lambda query: (
            {
                "name": "King's Cross",
                "lat": 51.5308,
                "lng": -0.1238,
                "detail": f"{query}, London",
            },
        ),
    )

    response = client.get("/api/geocode", params={"q": "Kings Cross"})

    assert response.status_code == 200
    assert response.json()["results"][0]["name"] == "King's Cross"
    assert response.json()["results"][0]["detail"] == "Kings Cross, London"


def test_predict_gaussian(payload, caplog):
    payload["mode"] = "gaussian"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert len(result["lats"]) == payload["y_size"]
    assert len(result["lngs"]) == payload["x_size"]
    assert len(result["Z"]) == payload["y_size"]
    assert len(result["Z"][0]) == payload["x_size"]
    assert np.mean(result["Z"][0]) != 0
    assert "engine: 'gaussian'" in caplog.text


def test_predict_linear(payload, caplog):
    payload["mode"] = "walking"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert len(result["lats"]) == payload["y_size"]
    assert len(result["lngs"]) == payload["x_size"]
    assert len(result["Z"]) == payload["y_size"]
    assert len(result["Z"][0]) == payload["x_size"]
    assert np.mean(result["Z"][0]) != 0
    assert "engine: 'linear'" in caplog.text


def test_get_plots(payload):
    client.post(
        "/plot/gaussian/3d",
        json=payload,
    )
    client.post(
        "/plot/linear/3d",
        json=payload,
    )
    client.post(
        "/plot/gaussian/contour",
        json=payload,
    )
    client.post(
        "/plot/linear/contour",
        json=payload,
    )
