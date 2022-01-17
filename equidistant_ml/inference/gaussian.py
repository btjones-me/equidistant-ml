"""Generates noise. Placeholder for a real model."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from equidistant_ml.base_grid import BaseGrid


@dataclass
class GenerateGaussian(BaseGrid):
    lat: float = None
    lng: float = None
    x0: float = None
    x1: float = None
    y0: float = None
    y1: float = None
    x_size: int = 50
    y_size: int = 50


    # def __init__(self, x_bounds=(-10, 10), y_bounds=(-10, 10), sigma_x=5., sigma_y=5., x_size=1000, y_size=1000):
    #     self.x_size = x_size
    #     self.y_size = y_size
    #     self.xs, self.ys, self.zs = self.get_gaussian(x_bounds, y_bounds, sigma_x, sigma_y, x_size, y_size)

    def __init__(self, x0=None, x1=None, y0=None, y1=None, x_size=1000, y_size=1000):


    @staticmethod
    def get_gaussian(x_bounds=(-10, 10), y_bounds=(-10, 10), sigma_x=5., sigma_y=5., x_size=1000, y_size=1000):
        x = np.linspace(x_bounds[0], x_bounds[1], x_size)
        y = np.linspace(y_bounds[0], y_bounds[1], y_size)
        x, y = np.meshgrid(x, y)
        z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
             + y**2/(2*sigma_y**2))))
        return x, y, z

#
#
# class GenerateGaussian(BaseGrid):
#
#     def __init__(self, x_bounds=(-10, 10), y_bounds=(-10, 10), sigma_x=5., sigma_y=5., x_size=1000, y_size=1000):
#         self.x_size = x_size
#         self.y_size = y_size
#         self.xs, self.ys, self.zs = self.get_gaussian(x_bounds, y_bounds, sigma_x, sigma_y, x_size, y_size)
#
#     @staticmethod
#     def get_gaussian(x_bounds=(-10, 10), y_bounds=(-10, 10), sigma_x=5., sigma_y=5., x_size=1000, y_size=1000):
#         x = np.linspace(x_bounds[0], x_bounds[1], x_size)
#         y = np.linspace(y_bounds[0], y_bounds[1], y_size)
#         x, y = np.meshgrid(x, y)
#         z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
#              + y**2/(2*sigma_y**2))))
#         return x, y, z
#
#
# class CoordinateDummy(GenerateGaussian):
#
#     def __init__(self, x0=None, x1=None, y0=None, y1=None, x_size=1000, y_size=1000):
#         super().__init__(x_size=x_size, y_size=y_size)
#
#         if any(val is None for val in (x0, x1, y0, y1)):
#             # use defaults
#             from equidistant_ml.config import cricklewood_coords, kidbrooke_coords
#             x0 = cricklewood_coords['lng']
#             x1 = kidbrooke_coords['lng']
#             y0 = cricklewood_coords['lat']
#             y1 = kidbrooke_coords['lat']
#
#         self.xs = np.linspace(x0, x1, self.x_size)
#         self.ys = np.linspace(y0, y1, self.y_size)
#
#         self.df = pd.DataFrame(self.zs)
#         self.df.columns = np.round(self.xs, 5)
#         self.df.set_index(np.round(self.ys, 5), inplace=True)


if __name__ == "__main__":
    c = CoordinateDummy()
    c.plot_contour()
