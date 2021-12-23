"""Generates random noise. Placeholder for a real model."""
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# def f(x, y, z):
#     return 2 * x**3 + 3 * y**2 - z
#
#
# x = np.linspace(1, 4, 11)
# y = np.linspace(4, 7, 22)
# z = np.linspace(7, 9, 33)
# xg, yg, zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
#
# data = f(xg, yg, zg)
#
#
# my_interpolating_function = RegularGridInterpolator((x, y, z), data)
#

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]


rng = np.random.default_rng()
points = rng.random((20, 2))
values = func(points[:, 0], points[:, 1])

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

plt.subplot(221)
plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()




##
# df = None
def assign_random_values_to_coordinates(df_coords: pd.DataFrame) -> pd.DataFrame:
    df_coords['values'] = np.round(np.random.rand(len(df_coords)), 4)
    return df_coords


upper_left = (51.5350, -0.1939)
lower_right = (51.5232, -0.1569)
grid_x, grid_y = np.mgrid[upper_left[1]:lower_right[1]:100j, lower_right[0]:upper_left[0]:200j]

x = np.linspace(upper_left[1],lower_right[1],10)
y = np.linspace(lower_right[0],upper_left[0],10)
z = np.zeros(10)

b = pd.DataFrame({'x': x, 'y': y, 'z': z})


xgygzg = np.meshgrid(x, y, z, indexing='ij', sparse=False)
xg, yg, zg = xgygzg
plt.plot(yg[0], xg[1], marker='.', color='k', linestyle='none')

def f(x,y,z):
    return z

plt.imshow(zg.T, extent=(upper_left[1],lower_right[1], lower_right[0],upper_left[0]), origin='lower')


df_values = assign_random_values_to_coordinates(df)


def get_big_grid(xs, ys):

    xs_expand = np.linspace(min(xs), max(xs), 100)
    ys_expand = np.linspace(min(ys), max(ys), 100)

    xg, yg = np.meshgrid(xs_expand, ys_expand, indexing='ij', sparse=False)
    return xg, yg

xg, yg = get_big_grid(df_values['Latitude'], df_values['Longitude'])

grid_z2 = griddata(df_values[['Latitude', 'Longitude']].head(4).values, df_values['values'].head(4).values*10**2, (xg, yg), method='cubic')

plt.scatter(df_values['Longitude'].head(4), df_values[['Latitude']].head(4),  s=df_values['values'].head(4).values*10**2)
plt.imshow(grid_z2.T, extent=(upper_left[1], lower_right[1], lower_right[0],upper_left[0]), origin='lower')

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def get_big_grid(xs, ys):

    xs_expand = np.linspace(min(xs), max(xs), 100)
    ys_expand = np.linspace(min(ys), max(ys), 100)

    xg, yg = np.meshgrid(xs_expand, ys_expand, indexing='ij', sparse=False)
    return xg, yg

def imshow_ext(x, y, *args, **kwargs):
    return plt.imshow(*args, extent=(min(x), max(x), min(y), max(y)), **kwargs)


n = 1000
df = pd.DataFrame({'lat': np.random.normal(size=n)+5, 'lon': np.random.normal(size=n)+5})

df['weights'] = (df['lat'].mean() * df['lon'].mean())**2 \
                - ((df['lat'].mean()-df['lat']) * (df['lon'].mean()-df['lon'])) ** 2
df['weights'] = (1/(2*np.pi*df['lat'].std()*df['lon'].std())) * np.exp(-(df['lat']**2 + df['lon']**2) / (2*(df['lat'].std()*df['lon'].std()))**2)
# df['weights'] = np.random.normal(1, 1, size=n)**2

xg, yg = get_big_grid(df['lat'], df['lon'])

# zgrid = griddata(points=(df['lat'], df['lon']), values=df['weights'], xi=(xg, yg), method='linear')

# plt.scatter(df_values['Longitude'].head(4), df_values[['Latitude']].head(4),  s=df_values['values'].head(4).values*10**2)
# imshow_ext(df['lat'], df['lon'], zgrid.T, origin='lower')
plt.scatter(df['lat'], df['lon'], s=df['weights']*100)
plt.title("(df['lat'].mean() * df['lon'].mean())**2 \
                - ((df['lat'].mean()-df['lat']) * (df['lon'].mean()-df['lon'])) ** 2")

plt.scatter(np.arange(0, len(df['weights'])), df['weights'])

df['weights'].hist(bins=100)
plt.title("(df['lat'].mean() * df['lon'].mean())**2 - ((df['lat'].mean()-df['lat']) * (df['lon'].mean()-df['lon'])) ** 2")



#
#
#
#
# grid_z2 = griddata(df_values[['Latitude', 'Longitude']].head(4).values, df_values['values'].head(4).values*10**2, (xg, yg), method='cubic')
#
# plt.scatter(df_values['Longitude'].head(4), df_values[['Latitude']].head(4),  s=df_values['values'].head(4).values*10**2)
# plt.imshow(grid_z2.T, extent=(upper_left[1], lower_right[1], lower_right[0],upper_left[0]), origin='lower')

from scipy.stats import multivariate_normal

x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]
# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])

mu = np.array([0.0, 0.0])

sigma = np.array([.025, .025])
covariance = np.diag(sigma**2)

z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

# Reshape back to a (30, 30) grid.
z = z.reshape(x.shape)
