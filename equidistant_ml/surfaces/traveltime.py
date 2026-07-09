"""TravelTime API client and response normalization."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import pandas as pd
import requests

TRAVELTIME_FAST_URL = "https://api.traveltimeapp.com/v4/time-filter/fast"


@dataclass(frozen=True)
class TravelTimeCredentials:
    app_id: str
    api_key: str

    @classmethod
    def from_env(cls) -> "TravelTimeCredentials":
        app_id = os.getenv("TRAVELTIME_APP_ID")
        api_key = os.getenv("TRAVELTIME_API_KEY")
        if not app_id or not api_key:
            raise ValueError(
                "TravelTime credentials missing. Set TRAVELTIME_APP_ID and "
                "TRAVELTIME_API_KEY in .env."
            )
        return cls(app_id=app_id, api_key=api_key)


class TravelTimeClient:
    def __init__(
        self,
        credentials: TravelTimeCredentials,
        timeout_seconds: int = 90,
        sleep_seconds: float = 1.0,
    ):
        self.credentials = credentials
        self.timeout_seconds = timeout_seconds
        self.sleep_seconds = sleep_seconds

    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Application-Id": self.credentials.app_id,
            "X-Api-Key": self.credentials.api_key,
        }

    @staticmethod
    def build_one_to_many_payload(
        origin: pd.Series,
        destinations: pd.DataFrame,
        *,
        transportation_type: str,
        arrival_time_period: str,
        travel_time_seconds: int,
        properties: Iterable[str],
    ) -> Dict[str, Any]:
        origin_id = str(origin["origin_id"])
        destination_ids = destinations["destination_id"].astype(str).tolist()
        locations = [
            {
                "id": origin_id,
                "coords": {"lat": float(origin["lat"]), "lng": float(origin["lng"])},
            }
        ]
        locations.extend(
            {
                "id": str(row.destination_id),
                "coords": {"lat": float(row.lat), "lng": float(row.lng)},
            }
            for row in destinations.itertuples(index=False)
        )
        return {
            "locations": locations,
            "arrival_searches": {
                "one_to_many": [
                    {
                        "id": origin_id,
                        "departure_location_id": origin_id,
                        "arrival_location_ids": destination_ids,
                        "travel_time": int(travel_time_seconds),
                        "arrival_time_period": arrival_time_period,
                        "properties": list(properties),
                        "transportation": {"type": transportation_type},
                    }
                ]
            },
        }

    def post_fast_matrix(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            TRAVELTIME_FAST_URL,
            headers=self.headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return response.json()


def parse_fast_matrix_response(
    origin_id: str,
    destinations: pd.DataFrame,
    response_json: Dict[str, Any],
    travel_time_limit_seconds: int,
    unreachable_penalty_seconds: int,
) -> pd.DataFrame:
    result = next(
        (
            item
            for item in response_json.get("results", [])
            if item.get("search_id") == origin_id
        ),
        None,
    )
    if result is None:
        raise ValueError(f"TravelTime response missing search_id={origin_id}")

    reachable = {
        item["id"]: item.get("properties", {})
        for item in result.get("locations", [])
        if "id" in item
    }
    unreachable_ids = set(result.get("unreachable", []))
    rows = []
    for destination_id in destinations["destination_id"].astype(str):
        props = reachable.get(destination_id)
        is_reachable = props is not None and destination_id not in unreachable_ids
        travel_time = props.get("travel_time") if props else None
        rows.append(
            {
                "origin_id": origin_id,
                "destination_id": destination_id,
                "travel_time_seconds": travel_time,
                "target_travel_time_seconds": (
                    float(travel_time)
                    if travel_time is not None
                    else float(travel_time_limit_seconds + unreachable_penalty_seconds)
                ),
                "reachable": bool(is_reachable),
                "api_status": "OK" if is_reachable else "UNREACHABLE",
            }
        )
    return pd.DataFrame(rows)


def fetch_origin_surfaces(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    client: TravelTimeClient,
    *,
    transportation_type: str,
    arrival_time_period: str,
    travel_time_seconds: int,
    unreachable_penalty_seconds: int,
    properties: Iterable[str],
    max_origins: int | None = None,
) -> pd.DataFrame:
    frames = []
    selected_origins = origins.head(max_origins) if max_origins else origins
    for _, origin in selected_origins.iterrows():
        payload = client.build_one_to_many_payload(
            origin,
            destinations,
            transportation_type=transportation_type,
            arrival_time_period=arrival_time_period,
            travel_time_seconds=travel_time_seconds,
            properties=properties,
        )
        response_json = client.post_fast_matrix(payload)
        frames.append(
            parse_fast_matrix_response(
                str(origin["origin_id"]),
                destinations,
                response_json,
                travel_time_seconds,
                unreachable_penalty_seconds,
            )
        )
    return pd.concat(frames, ignore_index=True)
