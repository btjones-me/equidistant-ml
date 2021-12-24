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
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """


@pytest.mark.skip()
def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():

    with open(Path(here() / "tests" / "test_payloads" / "1.json"), mode='r') as f:  # dummy in place
        payload = json.load(f)

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
