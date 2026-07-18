"""TravelTime API client and response normalization."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import requests
from tqdm import tqdm

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
        max_hits_per_minute: float | None = None,
        max_retries: int = 6,
    ):
        self.credentials = credentials
        self.timeout_seconds = timeout_seconds
        self.sleep_seconds = sleep_seconds
        self.max_hits_per_minute = max_hits_per_minute
        self.max_retries = max_retries
        self._minimum_interval = max(
            sleep_seconds,
            60.0 / max_hits_per_minute if max_hits_per_minute else 0.0,
        )
        self._last_request_started = 0.0

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
        for attempt in range(self.max_retries + 1):
            wait_seconds = self._minimum_interval - (
                time.monotonic() - self._last_request_started
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_started = time.monotonic()
            try:
                response = requests.post(
                    TRAVELTIME_FAST_URL,
                    headers=self.headers(),
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2**attempt, 60.0))
                continue
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response.json()
            if response.status_code != 429:
                if attempt >= self.max_retries:
                    response.raise_for_status()
                time.sleep(min(2**attempt, 60.0))
                continue
            retry_after = response.headers.get("Retry-After")
            try:
                retry_seconds = float(retry_after) if retry_after else 0.0
            except ValueError:
                retry_seconds = 0.0
            # The non-paid tier is five hits/minute. Once throttled, back down
            # to that rate for the remainder of the run.
            self._minimum_interval = max(self._minimum_interval, 12.0)
            if attempt >= self.max_retries:
                response.raise_for_status()
            time.sleep(max(retry_seconds, min(2**attempt, 60.0)))
        raise RuntimeError("TravelTime retry loop exited unexpectedly.")


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
    return normalize_label_frame(pd.DataFrame(rows))


def normalize_label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame["travel_time_seconds"] = pd.to_numeric(
        frame["travel_time_seconds"],
        errors="coerce",
    )
    frame["target_travel_time_seconds"] = pd.to_numeric(
        frame["target_travel_time_seconds"],
        errors="coerce",
    )
    frame["reachable"] = frame["reachable"].astype(bool)
    return frame


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
    checkpoint_dir: str | Path | None = None,
    run_id: str | None = None,
    checkpoint_schema: int = 2,
) -> pd.DataFrame:
    frames = []
    property_list = list(properties)
    selected_origins = origins.head(max_origins) if max_origins else origins
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoint_path:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    destination_payload = [
        {
            "destination_id": str(row.destination_id),
            "lat": round(float(row.lat), 7),
            "lng": round(float(row.lng), 7),
        }
        for row in destinations.itertuples(index=False)
    ]
    request_context = {
        "run_id": run_id,
        "checkpoint_schema": checkpoint_schema,
        "transportation_type": transportation_type,
        "arrival_time_period": arrival_time_period,
        "travel_time_seconds": int(travel_time_seconds),
        "unreachable_penalty_seconds": int(unreachable_penalty_seconds),
        "properties": property_list,
        "destinations": destination_payload,
    }

    for _, origin in tqdm(
        selected_origins.iterrows(),
        total=len(selected_origins),
        desc="traveltime origins",
    ):
        origin_id = str(origin["origin_id"])
        origin_checkpoint = (
            checkpoint_path / f"{origin_id.replace('/', '_')}.parquet"
            if checkpoint_path
            else None
        )
        fingerprint_payload = {
            **request_context,
            "origin": {
                "origin_id": origin_id,
                "lat": round(float(origin["lat"]), 7),
                "lng": round(float(origin["lng"]), 7),
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if origin_checkpoint and origin_checkpoint.exists():
            checkpoint = pd.read_parquet(origin_checkpoint)
            if run_id is not None:
                if "_checkpoint_fingerprint" not in checkpoint:
                    raise ValueError(
                        f"Legacy checkpoint cannot be reused for run {run_id}: "
                        f"{origin_checkpoint}"
                    )
                fingerprints = set(checkpoint["_checkpoint_fingerprint"].astype(str))
                if fingerprints != {fingerprint}:
                    raise ValueError(
                        f"Checkpoint fingerprint mismatch for {origin_id}; refusing "
                        "to mix data from different runs."
                    )
                expected_ids = destinations["destination_id"].astype(str).tolist()
                if checkpoint["destination_id"].astype(str).tolist() != expected_ids:
                    raise ValueError(
                        f"Checkpoint destinations mismatch for {origin_id}."
                    )
                checkpoint = checkpoint[
                    [column for column in checkpoint if not column.startswith("_")]
                ]
            frames.append(normalize_label_frame(checkpoint))
            continue

        payload = client.build_one_to_many_payload(
            origin,
            destinations,
            transportation_type=transportation_type,
            arrival_time_period=arrival_time_period,
            travel_time_seconds=travel_time_seconds,
            properties=property_list,
        )
        response_json = client.post_fast_matrix(payload)
        labels = parse_fast_matrix_response(
            origin_id,
            destinations,
            response_json,
            travel_time_seconds,
            unreachable_penalty_seconds,
        )
        if origin_checkpoint:
            checkpoint = labels.copy()
            if run_id is not None:
                checkpoint["_checkpoint_fingerprint"] = fingerprint
                checkpoint["_checkpoint_schema"] = int(checkpoint_schema)
            temporary = origin_checkpoint.with_suffix(".tmp.parquet")
            checkpoint.to_parquet(temporary, index=False)
            temporary.replace(origin_checkpoint)
        frames.append(labels)
    return pd.concat(frames, ignore_index=True)
