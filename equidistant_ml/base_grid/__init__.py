from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class BaseGrid:

    # to be set by parent class
    xs = None  # expected to be a 2d array
    ys = None  # expected to be a 2d array
    zs = None  # expected to be a 2d array

    def plot_contour(self, xs=None, ys=None, zs=None):
        if xs is None:
            xs = self.xs
        if ys is None:
            ys = self.ys
        if zs is None:
            zs = self.zs
        # plot as contour plot
        fig = plt.figure()
        plt.contourf(xs, ys, zs, cmap='Blues')
        plt.colorbar()
        return fig

    def plot_3d(self, xs=None, ys=None, zs=None):
        if xs is None:
            xs = self.xs
        if ys is None:
            ys = self.ys
        if zs is None:
            zs = self.zs
        # plot in 3D
        from matplotlib import cm
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.plot_surface(xs, ys, zs, rstride=3, cstride=3, linewidth=1, antialiased=True,
                        cmap=cm.viridis)
        return fig
