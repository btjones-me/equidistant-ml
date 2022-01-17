from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


@dataclass
class PredictGrid:
    lat: float
    lng: float
    x0: float
    x1: float
    y0: float
    y1: float

    # set by the predicting function
    # 2d array where one axis is invariant
    # eg [[0, 1, 2]
    #     [0, 1, 2]
    #     [0, 1, 2]]
    xs2d: np.ndarray = None  # expected to be a 2d array
    # 2d array where one axis is invariant
    # eg [[0, 0, 0]
    #     [2, 2, 2]
    #     [4, 4, 4]]
    ys2d: np.ndarray = None  # expected to be a 2d array
    # 2d array of results
    # eg [[0, 1, 2]
    #     [1, 3, 4]
    #     [2, 4, 5]]
    zs2d: np.ndarray = None  # expected to be a 2d array

    # size of the grid
    x_size: int = 50  # longitudes
    y_size: int = 50  # latitudes

    # set by the respective get function
    df: pd.DataFrame = None

    def __post_init__(self):
        self.xs = np.linspace(self.x0, self.x1, self.x_size)
        self.ys = np.linspace(self.y0, self.y1, self.y_size)

        # produces 2x2d arrays where one axis is invariant
        # eg [[0, 1, 2]     [[0, 0, 0]
        #     [0, 1, 2]      [2, 2, 2]
        #     [0, 1, 2]]     [4, 4, 4]]
        self.xs2d, self.ys2d = np.meshgrid(self.xs, self.ys)

    def _generate_gaussian(self, sigma_x: float = None, sigma_y: float = None):
        """Generate Gaussian.

        Centered on the user coordinates and scaled by the given range of the bbox.
        """
        if sigma_x is None:
            sigma_x = np.abs(self.x1 - self.x0)
        if sigma_y is None:
            sigma_y = np.abs(self.y0 - self.y1)

        gaussian = (1 / (2 * np.pi * sigma_x * sigma_y)
                    * np.exp(-((self.xs2d - self.lng) ** 2 / (2 * sigma_x ** 2)
                               + (self.ys2d - self.lat) ** 2 / (2 * sigma_y ** 2))))
        # invert the gaussian such that the lowest point is nearby the user
        self.zs2d = -1 * gaussian

    def get_gaussian(self, sigma_x=None, sigma_y=None):
        """Entry point - Get the gaussian and return the dataframe."""
        self._generate_gaussian(sigma_x, sigma_y)

        self.df = pd.DataFrame(self.zs2d)
        self.df.columns = np.round(self.xs, 5)
        self.df.set_index(np.round(self.ys, 5), inplace=True)

        return self.df

    def _generate_linear(self):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).
        """
        # Convert decimal degrees to Radians:
        lon1 = np.radians(self.xs2d)
        lat1 = np.radians(self.ys2d)
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

        self.zs2d = c * r

    def get_linear(self):
        """Entry point - Get the linear interpolation."""
        self._generate_linear()

        self.df = pd.DataFrame(self.zs2d)
        self.df.columns = np.round(self.xs, 5)
        self.df.set_index(np.round(self.ys, 5), inplace=True)

        return self.df

    def plot_contour(self, xs2d=None, ys2d=None, zs2d=None):
        if xs2d is None:
            xs2d = self.xs2d
        if ys2d is None:
            ys2d = self.ys2d
        if zs2d is None:
            zs2d = self.zs2d
        # plot as contour plot
        fig = plt.figure()
        plt.contourf(xs2d, ys2d, zs2d, cmap='Blues')
        plt.colorbar()
        return fig

    def plot_3d(self, xs2d=None, ys2d=None, zs2d=None):
        if xs2d is None:
            xs2d = self.xs2d
        if ys2d is None:
            ys2d = self.ys2d
        if zs2d is None:
            zs2d = self.zs2d
        # plot in 3D
        from matplotlib import cm
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.plot_surface(xs2d, ys2d, zs2d, rstride=3, cstride=3, linewidth=1, antialiased=True,
                         cmap=cm.viridis)
        return fig


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

    self = PredictGrid(**inference_args, **user_location_args)
    self._generate_gaussian(sigma_x=(inference_args['x1'] - inference_args['x0']),
                            sigma_y=(inference_args['y0'] - inference_args['y1']))
    self.plot_3d()
    self.plot_contour()

    self = PredictGrid(**inference_args, **user_location_args)
    self._generate_linear()
    self.plot_3d()
    self.plot_contour()
