"""Generate Google Directions samples for journey-time modelling."""

import argparse
import datetime as datetime
import json
import os
import pathlib
from typing import Dict, Tuple

import loguru
import numpy as np
import pandas as pd
import requests_cache
from dotenv import load_dotenv
from pyprojroot import here
from tqdm import tqdm

from equidistant_ml.utils import DatetimeUtils, sget

DEFAULT_UPPER_LEFT = (51.5350, -0.1939)
DEFAULT_LOWER_RIGHT = (51.5232, -0.1569)
DEFAULT_API_KEY_ENV = "GMAPS_API_KEY_MLIGENT"
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

session = requests_cache.CachedSession(
    str(here() / "gmaps_cache"), backend="sqlite", ignored_parameters=["key"]
)


class GetDirections:
    def __init__(
        self,
        nrows: int = 10,
        api_key: str | None = None,
        mode: str = "transit",
        upper_left: Tuple[float, float] = DEFAULT_UPPER_LEFT,
        lower_right: Tuple[float, float] = DEFAULT_LOWER_RIGHT,
        min_days_ahead: int = 1,
        max_days_ahead: int = 49,
        use_cache: bool = True,
        dry_run: bool = False,
        seed: int = 42,
    ):
        if nrows < 1:
            raise ValueError("nrows must be at least 1")
        if not dry_run and not api_key:
            raise ValueError(
                f"Google Maps API key missing. Set {DEFAULT_API_KEY_ENV} "
                "or pass --api-key-env."
            )
        if min_days_ahead < 0 or max_days_ahead < min_days_ahead:
            raise ValueError("Expected 0 <= min_days_ahead <= max_days_ahead")

        if seed is not None:
            np.random.seed(seed)

        self.api_key = api_key
        self.mode = mode
        self.upper_left = upper_left
        self.lower_right = lower_right
        self.min_days_ahead = min_days_ahead
        self.max_days_ahead = max_days_ahead
        self.use_cache = use_cache
        self.dry_run = dry_run
        self.df = self.main(n=nrows)

    @staticmethod
    def get_grid_coords(
        upper_left: Tuple[float, float] = DEFAULT_UPPER_LEFT,
        lower_right: Tuple[float, float] = DEFAULT_LOWER_RIGHT,
        size: int = 10,
    ):
        lats = np.linspace(upper_left[0], lower_right[0], size)
        lngs = np.linspace(upper_left[1], lower_right[1], size)
        coords = [[i, j] for i in lats for j in lngs]
        rounded_coords = np.round(coords, 5)
        return pd.DataFrame(rounded_coords, columns=["Latitude", "Longitude"])

    @staticmethod
    def get_random_coords(
        upper_left: Tuple[float, float] = DEFAULT_UPPER_LEFT,
        lower_right: Tuple[float, float] = DEFAULT_LOWER_RIGHT,
        n: int = 10,
    ):
        lats = np.random.uniform(upper_left[0], lower_right[0], n)
        lngs = np.random.uniform(upper_left[1], lower_right[1], n)
        coords = np.round(list(zip(lats, lngs)), 8)
        return pd.DataFrame(coords, columns=["Latitude", "Longitude"])

    @classmethod
    def get_random_coord_pairs(
        cls,
        upper_left: Tuple[float, float] = DEFAULT_UPPER_LEFT,
        lower_right: Tuple[float, float] = DEFAULT_LOWER_RIGHT,
        n: int = 10,
    ):
        df1 = cls.get_random_coords(upper_left, lower_right, n=n)
        df2 = cls.get_random_coords(upper_left, lower_right, n=n)
        df = pd.concat([df1, df2], axis=1)
        df.columns = ["start_lat", "start_lng", "end_lat", "end_lng"]
        return df

    @staticmethod
    def process_req(response_json: Dict):
        """Extract stable columns from a Google Directions response."""
        return {
            "api_status": response_json.get("status"),
            "api_error_message": response_json.get("error_message"),
            "distance": sget(
                response_json, "routes", 0, "legs", 0, "distance", "value"
            ),
            "duration": sget(
                response_json, "routes", 0, "legs", 0, "duration", "value"
            ),
            "start_location_lat": sget(
                response_json, "routes", 0, "legs", 0, "start_location", "lat"
            ),
            "start_location_lng": sget(
                response_json, "routes", 0, "legs", 0, "start_location", "lng"
            ),
            "end_location_lat": sget(
                response_json, "routes", 0, "legs", 0, "end_location", "lat"
            ),
            "end_location_lng": sget(
                response_json, "routes", 0, "legs", 0, "end_location", "lng"
            ),
        }

    def make_dir_request(
        self,
        epoch_time: str,
        origin_coords: Tuple[float, float],
        dest_coords: Tuple[float, float],
    ):
        """Request one route from the Google Directions API."""
        human_time = datetime.datetime.fromtimestamp(int(epoch_time)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        origin = f"{origin_coords[0]},{origin_coords[1]}"
        dest = f"{dest_coords[0]},{dest_coords[1]}"
        params = {
            "origin": origin,
            "destination": dest,
            "mode": self.mode,
            "key": self.api_key,
            "alternatives": False,
            "units": "metric",
            "departure_time": epoch_time,
        }

        loguru.logger.debug(f"Requesting {self.mode} route at {human_time}")
        if self.use_cache:
            resp = session.get(GOOGLE_DIRECTIONS_URL, params=params)
        else:
            with session.cache_disabled():
                resp = session.get(GOOGLE_DIRECTIONS_URL, params=params)
        resp.raise_for_status()

        response_json = resp.json()
        api_status = response_json.get("status")
        if api_status != "OK":
            loguru.logger.warning(f"Google Directions status was {api_status}")

        sanitized_url = (
            resp.url.replace(self.api_key, "REDACTED") if self.api_key else resp.url
        )
        return pd.Series(
            {
                "datetime": human_time,
                "epoch_time": epoch_time,
                "mode": self.mode,
                "origin": origin,
                "destination": dest,
                "request_url": sanitized_url,
                "from_cache": bool(getattr(resp, "from_cache", False)),
                "response_json": json.dumps(response_json),
                **self.process_req(response_json),
            }
        )

    def make_dry_run_row(
        self,
        epoch_time: str,
        origin_coords: Tuple[float, float],
        dest_coords: Tuple[float, float],
    ):
        human_time = datetime.datetime.fromtimestamp(int(epoch_time)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        origin = f"{origin_coords[0]},{origin_coords[1]}"
        dest = f"{dest_coords[0]},{dest_coords[1]}"
        return pd.Series(
            {
                "datetime": human_time,
                "epoch_time": epoch_time,
                "mode": self.mode,
                "origin": origin,
                "destination": dest,
                "request_url": None,
                "from_cache": False,
                "response_json": None,
                "api_status": "DRY_RUN",
                "api_error_message": None,
                "distance": None,
                "duration": None,
                "start_location_lat": origin_coords[0],
                "start_location_lng": origin_coords[1],
                "end_location_lat": dest_coords[0],
                "end_location_lng": dest_coords[1],
            }
        )

    def main(self, n: int = 10):
        pairs = self.get_random_coord_pairs(self.upper_left, self.lower_right, n=n)
        samples = []
        for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="directions"):
            epoch_time = DatetimeUtils.get_random_epoch_time(
                min_days_ahead=self.min_days_ahead,
                max_days_ahead=self.max_days_ahead,
            )
            origin_coords = (row["start_lat"], row["start_lng"])
            dest_coords = (row["end_lat"], row["end_lng"])
            if self.dry_run:
                sample = self.make_dry_run_row(epoch_time, origin_coords, dest_coords)
            else:
                sample = self.make_dir_request(epoch_time, origin_coords, dest_coords)
            samples.append(sample)

        return pd.concat(samples, axis=1).T


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrows", type=int, default=100)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--mode",
        default="transit",
        choices=["transit", "walking", "driving", "bicycling"],
    )
    parser.add_argument("--upper-left-lat", type=float, default=DEFAULT_UPPER_LEFT[0])
    parser.add_argument("--upper-left-lng", type=float, default=DEFAULT_UPPER_LEFT[1])
    parser.add_argument("--lower-right-lat", type=float, default=DEFAULT_LOWER_RIGHT[0])
    parser.add_argument("--lower-right-lng", type=float, default=DEFAULT_LOWER_RIGHT[1])
    parser.add_argument("--min-days-ahead", type=int, default=1)
    parser.add_argument("--max-days-ahead", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="data/training")
    parser.add_argument(
        "--output-format", choices=["parquet", "csv"], default="parquet"
    )
    return parser


def write_output(df: pd.DataFrame, output_dir: str, output_format: str):
    timestring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = pathlib.Path(here() / output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"directions.{timestring}.{output_format}"
    if output_format == "csv":
        df.to_csv(out_file, index=False)
    else:
        df.to_parquet(out_file)
    return out_file


def main():
    load_dotenv()
    args = build_parser().parse_args()
    api_key = os.getenv(args.api_key_env)
    generator = GetDirections(
        nrows=args.nrows,
        api_key=api_key,
        mode=args.mode,
        upper_left=(args.upper_left_lat, args.upper_left_lng),
        lower_right=(args.lower_right_lat, args.lower_right_lng),
        min_days_ahead=args.min_days_ahead,
        max_days_ahead=args.max_days_ahead,
        use_cache=not args.no_cache,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    out_file = write_output(generator.df, args.output_dir, args.output_format)
    loguru.logger.info(f"Wrote {len(generator.df)} rows to {out_file}")


if __name__ == "__main__":
    main()
