"""Interpolate measured points onto regular grid in phase space."""
import sys
import os
from os.path import join
import time
from datetime import datetime
import importlib
from pprint import pprint
import numpy as np
import pandas as pd
import h5py
import imageio
from scipy import ndimage
from scipy import interpolate
import skimage
from tqdm import tqdm
from tqdm import trange
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from plotly import graph_objects as go
import proplot as pplt
from ipywidgets import interactive

sys.path.append('../..')
from tools import image_processing as ip
from tools import plotting as mplt
from tools import utils
from tools.energyVS06 import EnergyCalculate

pplt.rc['grid'] = False
pplt.rc['cmap.discrete'] = False
pplt.rc['cmap.sequential'] = 'viridis'


# Setup
# ------------------------------------------------------------------------------
folder = '_output'

info = utils.load_pickle(join(folder, 'info.pkl'))
print('info')
pprint(info)

datadir = info['datadir']
filename = info['filename']
datadir = info['datadir']
file = h5py.File(join(datadir, 'preproc-' + filename + '.h5'), 'r')
data_sc = file['/scalardata']
data_wf = file['/wfdata']
data_im = file['/imagedata']

print('Attributes:')
print()
for data in [data_sc, data_wf, data_im]:
    print(data.name)
    for item in data.dtype.fields.items():
        print(item)
    print()

keys = list(info['variables'])
nsteps = np.array([variables[key]['steps'] for key in keys])

acts = info['acts']
points = np.vstack([data_sc[act] for act in acts]).T

## Convert to beam frame coordinates.
cam = info['cam']
if cam.lower() not in ['cam06', 'cam34']:
    raise ValueError('Unknown camera name!')

# y slit is inserted from above, always opposite y_beam.
points[:, 0] = -points[:, 0]

# VT04/VT06 are same sign as x_beam. From the BTF diagram, VT34a and VT34b 
# appear to be opposite sign; however, I think that is wrong. 
if cam.lower() == 'cam06':
    pass
elif cam.lower() == 'cam34':
    # points[:, 1:] *= -1.0
    pass
    
# Screen (x3, y3) are always opposite (x_beam, y_beam). (The beam is moving
# into the screen; the first row of the image is the maximum y.)
image_shape = info['image_shape']
y3grid = -np.arange(image_shape[0]) * info['image_pix2mm_y']
x3grid = -np.arange(image_shape[1]) * info['image_pix2mm_x']

# Dipole bend radius is positive/negative at VS06/VS34.
rho_sign = None
if cam.lower() == 'cam06':
    rho_sign = +1.0
elif cam.lower() == 'cam34':
    rho_sign = -1.0

# Build transfer matrices between slits/screens.
a2mm = 1.009  # assume same for both dipoles
rho = 0.3556  # bend radius
GL05 = 0.0  # quad 
GL06 = 0.0  # quad
l1 = 0.0
l2 = 0.0
l3 = 0.774
L2 = 0.311  # second slit to dipole face
l = 0.129  # dipole face to screen
LL = l1 + l2 + l3 + L2  # distance from emittance plane to dipole entrance
ecalc = EnergyCalculate(l1=l1, l2=l2, l3=l3, L2=L2, l=l, 
                        amp2meter=a2mm*1e3, rho_sign=rho_sign)
Mslit = ecalc.getM1()  # slit-slit
Mscreen = ecalc.getM()  # slit-screen


# Interpolation
# ------------------------------------------------------------------------------

## Setup interpolation grids

### y grid
y_scale = 1.1
ygrid = np.linspace(
    np.min(points[:, 0]), 
    np.max(points[:, 0]), 
    int(y_scale * (nsteps[0] + 1)),
)

### x-x' grid

# Assume that $x$ and $x'$ do not change on each iteration (or that the 
# only variation is noise in the readback value). Select an $x$ and $x'$
# for each $\left\{y, y_3, x_3\right\}$.
iterations = data_sc['iteration']
iteration_nums = np.unique(iterations)
n_iterations = len(iteration_nums)
XXP = np.zeros((n_iterations, 2))
for iteration in iteration_nums:
    idx = iterations == iteration
    x2, x1 = np.mean(points[idx, 1:], axis=0)
    x = x1
    xp = 1e3 * ecalc.calculate_xp(x1 * 1e-3, x2 * 1e-3, Mslit)
    XXP[iteration - 1] = (x, xp)

# Define the $x$-$x'$ interpolation grid. Tune `x_scale` and `xp_scale` 
# to roughly align the grid points with the measured points.
x_scale = 1.1
xp_scale = 1.5
x_min, xp_min = np.min(XXP, axis=0)
x_max, xp_max = np.max(XXP, axis=0)
xgrid = np.linspace(x_min, x_max, int(x_scale * (nsteps[2] + 1)))
xpgrid = np.linspace(xp_min, xp_max, int(xp_scale * (nsteps[1] + 1)))


fig, ax = pplt.subplots(figwidth=4)
line_kws = dict(color='lightgray', lw=0.7)
for x in xgrid:
    g1 = ax.axvline(x, **line_kws)
for xp in xpgrid:
    ax.axhline(xp, **line_kws)
ax.plot(XXP[:, 0], XXP[:, 1], color='pink7', lw=0, marker='.', ms=2)
ax.format(xlabel='x [mm]', ylabel='xp [mrad]')
plt.savefig('_output/xxp_interp_grid.png')


### y' grid
yp_scale = 1.25  # scales resolution of y' interpolation grid
_Y, _Y3 = np.meshgrid(ygrid, y3grid, indexing='ij')
_YP = 1e3 * ecalc.calculate_yp(_Y * 1e-3, _Y3 * 1e-3, Mscreen)  # [mrad]
ypgrid = np.linspace(
    np.min(_YP),
    np.max(_YP), 
    int(yp_scale * image_shape[0]),
)

fig, ax = pplt.subplots(figwidth=4)
for yp in ypgrid:
    ax.axhline(yp, **line_kws)
ax.plot(_Y.ravel(), _YP.ravel(), color='pink7', lw=0, marker='.', ms=2)
ax.format(xlabel='y [mm]', ylabel='yp [mrad]')
plt.savefig('_output/yyp_interp_grid.png')

### w grid
w_scale = 1.1
_W = np.zeros((len(xgrid), len(xpgrid), image_shape[1]))
for i in range(_W.shape[0]):
    for j in range(_W.shape[1]):
        x = xgrid[i]
        xp = xpgrid[j]
        _W[i, j, :] = ecalc.calculate_dE_screen(x3grid * 1e-3, 0.0, x * 1e-3, xp * 1e-3, Mscreen)
wgrid = np.linspace(np.min(_W), np.max(_W), int(w_scale * image_shape[1]))


## Interpolate

# Interpolate the image stack along the $y$ axis on each iteration. 
print('Interpolating y')
images = data_im[cam + '_Image'].reshape((len(data_im), image_shape[0], image_shape[1]))
images_3D = np.zeros((n_iterations, len(ygrid), len(y3grid), len(x3grid))
for count, iteration in enumerate(tqdm(iteration_nums)):
    idx, = np.where(iterations == iteration)
    _points = points[idx, 0]
    _values = images[idx, :, :]
    _, uind = np.unique(_points, return_index=True)                     
    fint = interpolate.interp1d(
        _points[uind], 
        _values[uind], 
        axis=0,
        kind='linear', 
        bounds_error=False,
        fill_value=0.0, 
        assume_sorted=False,
    )
    images_3D[count] = fint(ygrid) 

# Interpolate $x$-$x'$ for each $\left\{y, y_3, x_3\right\}$.
print('Interpolating x-xp')
_new_points = utils.get_grid_coords(xgrid, xpgrid, indexing='ij')
shape = (len(xgrid), len(xpgrid), len(ygrid), len(y3grid), len(x3grid))
f = np.zeros(shape)
for k in trange(shape[2]):
    for l in trange(shape[3]):
        for m in range(shape[4]):
            new_values = interpolate.griddata(
                XXP, 
                images_3D[:, k, l, m],
                new_points,
                method='linear',
                fill_value=False,
            )
            f[:, :, k, l, m] = new_values.reshape((shape[0], shape[1]))

# The $y$ coordinate is already on a grid. For each $\left\{x, x', y, x_3\right\}$,
# transform $y_3 \rightarrow y'$ and interpolate onto `ypgrid`. 
print('Interpolating yp')
shape = (len(xgrid), len(xpgrid), len(ygrid), len(ypgrid), len(x3grid))
f_new = np.zeros(shape)
for i in trange(shape[0]):
    for j in trange(shape[1]):
        for k in range(shape[2]):
            for m in range(shape[4]): 
                y = ygrid[k]
                yp = 1e3 * ecalc.calculate_yp(y * 1e-3, y3grid * 1e-3, Mscreen)
                fint = interpolate.interp1d(
                    yp,
                    f[i, j, k, :, m], 
                    kind='linear', 
                    fill_value=0.0, 
                    bounds_error=False,
                    assume_sorted=False,
                )
                f_new[i, j, k, :, m] = fint(ypgrid)
f = f_new.copy()


# Interpolate energy spread $w$ for each $\left\{x, x', y, y'\right\}$.
print('Interpolating w')
shape = (len(xgrid), len(xpgrid), len(ygrid), len(ypgrid), len(wgrid))
savefilename = f'_output/f_{filename}.mmp'
f_new = np.memmap(savefilename, shape=shape, dtype='float', mode='w+') 
for i in trange(shape[0]):
    for j in trange(shape[1]):
        for k in range(shape[2]):
            for l in range(shape[3]):
                x = xgrid[i]
                xp = xpgrid[j]
                w = ecalc.calculate_dE_screen(x3grid * 1e-3, 0.0, x * 1e-3, xp * 1e-3, Mscreen)
                fint = interpolate.interp1d(
                    w,
                    f[i, j, k, l, :],
                    kind='linear',
                    fill_value=0.0, 
                    bounds_error=False,
                    assume_sorted=False,
                )
                f_new[i, j, k, l, :] = fint(wgrid)
                
# Save grid coordinates.
coords = [xgrid, xpgrid, ygrid, ypgrid, wgrid]
for i in range(5):
    coords[i] = coords[i] - np.mean(coords[i])
utils.save_stacked_array(f'_output/coords_{filename}.npz', coords)

# Save info
info['int_shape'] = shape

print('info:')
pprint(info)

# Save as pickled dictionary for loading.
utils.save_pickle('_output/info.pkl', info)

# Save as file for viewing.
file = open('_output/info.txt', 'w')
for key, value in info.items():
    file.write(f'{key}: {value}\n')
file.close()

print('Done.')