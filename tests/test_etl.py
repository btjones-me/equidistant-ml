import datetime as datetime

from equidistant_ml.etl.get_lattice_data import GetDirections
from equidistant_ml.utils import DatetimeUtils


class DummyResponse:
    url = "https://maps.googleapis.com/maps/api/directions/json?key=test-secret"
    from_cache = False

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "status": "OK",
            "routes": [
                {
                    "legs": [
                        {
                            "distance": {"value": 1234},
                            "duration": {"value": 567},
                            "start_location": {"lat": 51.5, "lng": -0.1},
                            "end_location": {"lat": 51.6, "lng": -0.2},
                        }
                    ]
                }
            ],
        }


def test_make_dir_request_redacts_api_key(monkeypatch):
    def fake_get(url, params):
        assert params["key"] == "test-secret"
        return DummyResponse()

    monkeypatch.setattr("equidistant_ml.etl.get_lattice_data.session.get", fake_get)
    generator = GetDirections(nrows=1, api_key="test-secret", dry_run=True)

    row = generator.make_dir_request("1893456000", (51.5, -0.1), (51.6, -0.2))

    assert "test-secret" not in row["request_url"]
    assert row["distance"] == 1234
    assert row["duration"] == 567


def test_dry_run_does_not_require_api_key():
    generator = GetDirections(nrows=2, dry_run=True)

    assert len(generator.df) == 2
    assert generator.df["api_status"].tolist() == ["DRY_RUN", "DRY_RUN"]
    assert generator.df["request_url"].isna().all()


def test_random_epoch_time_defaults_to_future():
    epoch = int(DatetimeUtils.get_random_epoch_time())
    now = int(datetime.datetime.now().timestamp())

    assert epoch > now
