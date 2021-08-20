"""Get lattice data takes random coordinates and gets the distance between them."""
import datetime as datetime

import numpy as np
import pandas as pd
from pyprojroot import here
from dotenv import load_dotenv
import os
from loguru import logger
import requests_cache
from pyprojroot import here
from tqdm import tqdm
from typing import List, Dict

from equidistant_ml.utils import sget


session = requests_cache.CachedSession(
    str(here() / "gmaps_cache"), backend="sqlite", ignored_parameters=["key"]
)

load_dotenv()  # take environment variables from .env.

GMAPS_API_KEY = os.getenv("GMAPS_API_KEY_MLIGENT")


class GetDirections:
    def __init__(self):
        self.api_key = GMAPS_API_KEY
        pass

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
        coords = np.round(coords, 5)
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

    def make_dir_request(self):
        """Make request to directions api.

        https://maps.googleapis.com/maps/api/directions/json?
        origin=Disneyland&destination=Universal+Studios+Hollywood&key=YOUR_API_KEY
        """
        url = f"https://maps.googleapis.com/maps/api/directions/json"
        # directions by road don't need a departure or arrival time, it will give the average
        # directions by transit do need a departure or arrival time, or it defaults to now
        resp = session.get(
            url,
            params={
                "origin": "51.530500,-0.177560",
                "destination": "51.529465,-0.124663",
                "mode": "transit",
                "key": self.api_key,
                "alternatives": False,
                "units": "metric",
                "departure_time": 1629417550,
            },  # in seconds, omit for road
        )

        s = pd.Series(self.process_req(resp.json))

    def main(self):
        # generate a random set of coordinates
        # generate a random set of coordinate pairs
        # for each coordinate pair, get the public transport time between both points
        # append the results to a dataframe
        pass
