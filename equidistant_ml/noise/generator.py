"""Generates noise. Placeholder for a real model."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GenerateGaussian:

    def __init__(self, x_bounds=(-10, 10), y_bounds=(-10, 10), sigma_x=5., sigma_y=5., x_size=1000, y_size=1000):
        self.x_size = x_size
        self.y_size = y_size
        self.x, self.y, self.z = self.get_gaussian(x_bounds, y_bounds, sigma_x, sigma_y, x_size, y_size)

    @staticmethod
    def get_gaussian(x_bounds=(-10, 10), y_bounds=(-10, 10), sigma_x=5., sigma_y=5., x_size=1000, y_size=1000):
        x = np.linspace(x_bounds[0], x_bounds[1], x_size)
        y = np.linspace(y_bounds[0], y_bounds[1], y_size)
        x, y = np.meshgrid(x, y)
        z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
             + y**2/(2*sigma_y**2))))
        return x, y, z

    def plot_contour(self, x=None, y=None, z=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if z is None:
            z = self.z
        # plot as contour plot
        fig = plt.figure()
        plt.contourf(x, y, z, cmap='Blues')
        plt.colorbar()
        return fig

    def plot_3d(self, x=None, y=None, z=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if z is None:
            z = self.z
        # plot in 3D
        from matplotlib import cm
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.plot_surface(x, y, z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                        cmap=cm.viridis)
        return fig


class CoordinateDummy(GenerateGaussian):

    def __init__(self, x0=None, x1=None, y0=None, y1=None, x_size=1000, y_size=1000):
        super().__init__(x_size=x_size, y_size=y_size)

        if any(val is None for val in (x0, x1, y0, y1)):
            # use defaults
            from equidistant_ml.config import cricklewood_coords, kidbrooke_coords
            x0 = cricklewood_coords['lng']
            x1 = kidbrooke_coords['lng']
            y0 = cricklewood_coords['lat']
            y1 = kidbrooke_coords['lat']

        xs = np.linspace(x0, x1, self.x_size)
        ys = np.linspace(y0, y1, self.y_size)

        self.df = pd.DataFrame(self.z)
        self.df.columns = np.round(xs, 5)
        self.df.set_index(np.round(ys, 5), inplace=True)


if __name__ == "__main__":
    c = CoordinateDummy()
    c.plot_contour()
