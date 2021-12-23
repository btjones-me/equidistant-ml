"""Get lattice data takes random coordinates and gets the distance between them."""
import datetime as datetime
import pathlib

import loguru
import numpy as np
import pandas as pd
from pyprojroot import here
from dotenv import load_dotenv
import os
from loguru import logger
import requests_cache
from pyprojroot import here
from tqdm import tqdm
from typing import List, Dict, Tuple

from equidistant_ml.utils import sget, DatetimeUtils

session = requests_cache.CachedSession(
    str(here() / "gmaps_cache"), backend="sqlite", ignored_parameters=["key"]
)

load_dotenv()  # take environment variables from .env.

GMAPS_API_KEY = os.getenv("GMAPS_API_KEY_MLIGENT")

np.random.seed(42)


class GetDirections:
    def __init__(self, nrows=10):
        self.api_key = GMAPS_API_KEY
        self.df = self.main(n=nrows)

    @staticmethod
    def _convert_timestamp_to_seconds(dtime: datetime.datetime):
        return dtime.timestamp()

    @staticmethod
    def get_grid_coords(upper_left=(51.5350, -0.1939), lower_right=(51.5232, -0.1569)):
        lats = np.linspace(upper_left[0], lower_right[0], 10)
        lngs = np.linspace(upper_left[1], lower_right[1], 10)
        coords = [[i, j] for i in lats for j in lngs]
        coords = np.round(coords, 5)
        df = pd.DataFrame(coords, columns=["Latitude", "Longitude"])
        return df

    @staticmethod
    def get_random_coords(
        upper_left=(51.5350, -0.1939), lower_right=(51.5232, -0.1569), n=10
    ):

        lats = np.random.uniform(upper_left[0], lower_right[0], n)
        lngs = np.random.uniform(upper_left[1], lower_right[1], n)
        coords = list(zip(lats, lngs))
        coords = np.round(coords, 8)
        df = pd.DataFrame(coords, columns=["Latitude", "Longitude"])
        return df

    @classmethod
    def get_random_coord_pairs(cls, n=10):
        df1 = cls.get_random_coords(n=n)
        df2 = cls.get_random_coords(n=n)
        df = pd.concat([df1, df2], axis=1)
        df.columns = ["start_lat", "start_lng", "end_lat", "end_lng"]
        return df

    @staticmethod
    def process_req(r):
        """Get the request, return the key info."""
        d = {}

        d["distance"] = sget(
            r, "routes", 0, "legs", 0, "distance", "value"
        )  # in metres
        d["duration"] = sget(
            r, "routes", 0, "legs", 0, "duration", "value"
        )  # in seconds
        d["start_location_lat"] = sget(
            r, "routes", 0, "legs", 0, "start_location", "lat"
        )
        d["start_location_lng"] = sget(
            r, "routes", 0, "legs", 0, "start_location", "lng"
        )
        d["end_location_lat"] = sget(r, "routes", 0, "legs", 0, "end_location", "lat")
        d["end_location_lng"] = sget(r, "routes", 0, "legs", 0, "end_location", "lng")

        return d

    def make_dir_request(self, epoch_time: str, origin_coords: Tuple[float, float], dest_coords: Tuple[float, float]):
        """Make request to directions api.

        https://maps.googleapis.com/maps/api/directions/json?
        origin=Disneyland&destination=Universal+Studios+Hollywood&key=YOUR_API_KEY
        """
        url = f"https://maps.googleapis.com/maps/api/directions/json"
        # directions by road don't need a departure or arrival time, it will give the average
        # directions by transit do need a departure or arrival time, or it defaults to now
        human_time = datetime.datetime.fromtimestamp(int(epoch_time)) \
                     .strftime("%Y-%m-%dT%H:%M:%S")  # see time

        origin = f"{str(origin_coords[0])},{str(origin_coords[1])}"
        dest = f"{str(dest_coords[0])},{str(dest_coords[1])}"
        loguru.logger.debug(f"Making request at: {human_time},\n{origin=}\n{dest=}")

        resp = session.get(
            url,
            params={
                "origin": origin,
                "destination": dest,
                "mode": "transit",
                "key": self.api_key,
                "alternatives": False,
                "units": "metric",
                "departure_time": epoch_time,
            },  # in seconds, omit for road
        )

        s = pd.Series({**{'datetime': human_time, 'epoch_time': epoch_time},
                       **self.process_req(resp.json()),
                       **{'url': resp.url, 'json': resp.json(), 'origin': origin, "destination": dest}})
        return s

    def main(self, n=10):
        # generate a random set of coordinate pairs
        pairs = self.get_random_coord_pairs(n=n)
        # for each coordinate pair, get the public transport time between both points
        ls = []
        for _, row in pairs.iterrows():
            epoch_time = DatetimeUtils.get_random_epoch_time()
            s = self.make_dir_request(epoch_time,
                                      (row[0], row[1]),
                                      (row[2], row[3]))
            ls.append(s)
        df = pd.concat(ls, axis=1).T

        return df


if __name__ == "__main__":

    gd = GetDirections(nrows=10)
    df = gd.df
    timestring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = pathlib.Path(here() / f'data/training/directions.{timestring}.parquet')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file)
