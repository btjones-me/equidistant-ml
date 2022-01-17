"""Tests for `equidistant_ml` app."""
import json
from pathlib import Path

import numpy as np
import pytest
from assertpy import assert_that

from fastapi.testclient import TestClient
from pyprojroot import here

import equidistant_ml.app as app


client = TestClient(app.app)


@pytest.fixture
def payload():
    with open(Path(here() / "tests" / "test_payloads" / "1.json"), mode='r') as f:  # dummy in place
        payload = json.load(f)
    return payload

@pytest.mark.skip()
def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict(payload):
    response = client.post(
        "/predict",
        json=payload,
    )
    assert response.status_code == 200

    result = response.json()

    assert len(result['lats']) == payload['y_size']
    assert len(result['lngs']) == payload['x_size']
    assert len(result['Z']) == payload['y_size']
    assert len(result['Z'][0]) == payload['x_size']
    assert np.mean(result['Z'][0]) != 0


def test_predict_gaussian(payload, caplog):
    payload["mode"] = 'gaussian'

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert len(result['lats']) == payload['y_size']
    assert len(result['lngs']) == payload['x_size']
    assert len(result['Z']) == payload['y_size']
    assert len(result['Z'][0]) == payload['x_size']
    assert np.mean(result['Z'][0]) != 0
    print(caplog.text)
    assert "engine: 'gaussian'" in caplog.text


def test_predict_linear(payload, caplog):
    payload["mode"] = 'walking'

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert len(result['lats']) == payload['y_size']
    assert len(result['lngs']) == payload['x_size']
    assert len(result['Z']) == payload['y_size']
    assert len(result['Z'][0]) == payload['x_size']
    assert np.mean(result['Z'][0]) != 0
    assert "engine: 'linear'" in caplog.text


def test_get_plots(payload):
    response = client.post(
        "/plot/gaussian/3d",
        json=payload,
    )
    response = client.post(
        "/plot/linear/3d",
        json=payload,
    )
    response = client.post(
        "/plot/gaussian/contour",
        json=payload,
    )
    response = client.post(
        "/plot/linear/contour",
        json=payload,
    )
