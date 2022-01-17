
from dataclasses import dataclass

import numpy as np
import pandas as pd

from equidistant_ml.base_grid import BaseGrid


@dataclass
class LinearInference(BaseGrid):
    lat: float = None
    lng: float = None
    x0: float = None
    x1: float = None
    y0: float = None
    y1: float = None
    x_size: int = 50
    y_size: int = 50

    def __post_init__(self):
        x = np.linspace(self.x0, self.x1, self.x_size)
        y = np.linspace(self.y0, self.y1, self.y_size)

        # produces 2x2d arrays where one axis is invariant
        # eg [[0, 1, 2]     [[0, 0, 0]
        #     [0, 1, 2]      [2, 2, 2]
        #     [0, 1, 2]]     [4, 4, 4]]
        self.xs, self.ys = np.meshgrid(x, y)
        self.zs = self.find_distance()

        self.df = pd.DataFrame(self.zs)
        self.df.columns = np.round(x, 5)
        self.df.set_index(np.round(y, 5), inplace=True)

    def find_distance(self):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).
        """
        # Convert decimal degrees to Radians:
        lon1 = np.radians(self.xs)
        lat1 = np.radians(self.ys)
        lon2 = np.radians(self.lng)
        lat2 = np.radians(self.lat)

        # Implementing Haversine Formula:
        dlon = np.subtract(lon2, lon1)
        dlat = np.subtract(lat2, lat1)

        a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
                   np.multiply(np.cos(lat1),
                               np.multiply(np.cos(lat2),
                                           np.power(np.sin(np.divide(dlon, 2)), 2))))
        c = np.multiply(2, np.arcsin(np.sqrt(a)))
        r = 6371

        return c * r


if __name__ == "__main__":
    import json
    from pathlib import Path
    from pyprojroot import here

    with open(Path(here() / "tests" / "test_payloads" / "1.json"), mode='r') as f:  # dummy in place
        payload = json.load(f)
    inference_args = {
        "x0": payload["bbox"]["top_left"]["lng"],
        "x1": payload["bbox"]["bottom_right"]["lng"],
        "y0": payload["bbox"]["top_left"]["lat"],
        "y1": payload["bbox"]["bottom_right"]["lat"],
        "x_size": payload["x_size"],
        "y_size": payload["y_size"]
    }
    user_location_args = {
        "lat": payload["lat"],
        "lng": payload["lng"]
    }

    self = LinearInference(**inference_args, **user_location_args)
    self.plot_3d()

