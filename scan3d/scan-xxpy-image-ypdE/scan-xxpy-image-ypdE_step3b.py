import sys
import os
from os.path import join
import time
from datetime import datetime
import importlib
import numpy as np
import pandas as pd
import h5py
from pprint import pprint
import imageio
from scipy import ndimage
from scipy import interpolate
import skimage
from tqdm import tqdm
from tqdm import trange
from matplotlib import pyplot as plt
from matplotlib import colors
import plotly.graph_objs as go
from ipywidgets import interact
import proplot as pplt

sys.path.append('../..')
from tools import energyVS06 as energy
from tools import image_processing as ip
from tools import plotting as mplt
from tools import utils
from tools.utils import project

pplt.rc['grid'] = False
pplt.rc['cmap.discrete'] = False
pplt.rc['cmap.sequential'] = 'viridis'


# Load data
# ------------------------------------------------------------------------------
folder = '_output'
filenames = os.listdir(folder)
for filename in filenames:
    if filename.startswith('rawgrid'):
        print(filename)

filename = 'rawgrid_220429190854-scan-xxpy-image-ypdE.mmp'
coordfilename = 'rawgrid_coordinates_220429190854-scan-xxpy-image-ypdE.npy'

info = utils.load_pickle('_output/info.pkl')
print('info:')
pprint(info)

shape = info['rawgrid_shape']  # (x1, x2, y1, y3, x3)
dtype = info['im_dtype']
cam = info['cam']

f_raw = np.memmap(join(folder, filename), shape=shape, dtype=dtype, mode='r')
print(np.info(f_raw))

# Use the Right Hand Rule to determine the beam coordinates. [Insert image here].
# * Cam06 
#     * x_slit (x1, x2) = x_beam
#     * y_slit (y1) = -y_beam
#     * y_screen (y3) = -y_beam
#     * x_screen (x3) = -x_beam    
# * Cam34
#     * x_slit (x1, x2) = -x_beam (Are you sure??? Seems to give the wrong answer.)
#     * y_slit (y1) = -y_beam
#     * y_screen (y3) = -y_beam
#     * x_screen (x3) = +x_beam
if cam.lower() == 'cam06':
    f_raw = f_raw[:, :, ::-1, ::-1, ::-1]
elif cam.lower() == 'cam34':
    # a5d = a5d[::-1, ::-1, ::-1, ::-1, :]
    f_raw = f_raw[:, :, ::-1, ::-1, :]


## Load slit coordinates
coords_3d = np.load(join(folder, coordfilename))  # [X1, X2, Y1]
print('coords_3d.shape:', coords_3d.shape)

dims = ["x1", "x2", "y1", "y3", "x3"]
dim_to_int = {dim: i for i, dim in enumerate(dims)}

fig, axes = pplt.subplots(nrows=3, ncols=3, figwidth=6, spanx=False, spany=False)
for i in range(3):
    for j in range(3):
        U = coords_3d[j]
        V = coords_3d[i]
        ax = axes[i, j]
        ax.scatter(U.ravel(), V.ravel(), s=1, color='black')
        ax.axvline(np.mean(U), color='red', alpha=0.15)
        ax.axhline(np.mean(V), color='red', alpha=0.15)
    axes[i, 0].format(ylabel=dims[i])
    axes[-1, i].format(xlabel=dims[i])
    
# Copy the grids to cover the five-dimensional space.
X1, X2, Y1 = coords_3d
X1 = utils.copy_into_new_dim(X1, shape[3:], axis=-1)
X2 = utils.copy_into_new_dim(X2, shape[3:], axis=-1)
Y1 = utils.copy_into_new_dim(Y1, shape[3:], axis=-1)
print('X1.shape =', X1.shape)
print('X2.shape =', X2.shape)
print('Y1.shape =', Y1.shape)

Y3, X3 = np.meshgrid(np.arange(shape[3]), np.arange(shape[4]), indexing='ij')
Y3 = utils.copy_into_new_dim(Y3, shape[:3], axis=0)
X3 = utils.copy_into_new_dim(X3, shape[:3], axis=0)
print('Y3.shape =', Y3.shape)
print('X3.shape =', X3.shape)


# Make lists of centered coordinates `coords_`.
X1 = X1 - np.mean(X1)
X2 = X2 - np.mean(X2)
Y1 = Y1 - np.mean(Y1)
Y3 = Y3 - np.mean(Y3)
X3 = X3 - np.mean(X3)


# Crop x3
cut = 20
idx = (slice(None), slice(None), slice(None), slice(None), slice(cut, -cut))
X1 = X1[idx]
X2 = X2[idx]
Y1 = Y1[idx]
Y3 = Y3[idx]
X3 = X3[idx]
f_raw = f_raw[idx]


coords_ = [X1, X2, Y1, Y3, X3]

for i, dim in enumerate(dims):
    print('dim =', dim)
    print('slice of coordinate array:')
    U = coords_[i]
    axes = [k for k in range(U.ndim) if k != i]
    idx = utils.make_slice(U.ndim, axes, ind=[0, 0, 0, 0])
    print(U[idx])
    print()

    
# ## View 5D array in slit-screen coordinates

# Correlation between planes are removed... units are dimensionless. Need to be careful interpreting these plots.

# ### Projections 

f_raw_min = np.min(f_raw)
if f_raw_min < 0:
    print(f'min(f_raw) = {f_raw_min}. Clipping to 0.')
    f_raw = np.clip(f_raw, 0, None)

frac_thresh = 1e-5
for norm in [None, 'log']:
    axes = mplt.corner(
        f_raw,
        labels=dims,
        norm=norm,
        diag_kind='None',
        prof=True,
        prof_kws=dict(lw=1.0, alpha=0.5, scale=0.12),
        fig_kws=dict(),
        frac_thresh=frac_thresh,
    )
    plt.savefig(f"_output/slitscreen_corner_log{norm == 'log'}.png")


# ### Slices

# Compute the indices of the maximum pixel in the 5D array.
ind_max = np.unravel_index(np.argmax(f_raw), f_raw.shape)
print('Indices of max(f_raw):', ind_max)

axes_slice = [(k, j, i) for i in range(f_raw.ndim) for j in range(i) for k in range(j)]
axes_view = [tuple([i for i in range(f_raw.ndim) if i not in axis])
             for axis in axes_slice]
for axis, axis_view in zip(axes_slice, axes_view):
    idx = utils.make_slice(5, axis, [ind_max[i] for i in axis])
    f_raw_slice = f_raw[idx]
    f_raw_slice = f_raw_slice / np.max(f_raw_slice)

    dim1, dim2 = [dims[i] for i in axis_view]
    
    fig, plot_axes = pplt.subplots(ncols=2)
    for ax, norm in zip(plot_axes, [None, 'log']):
        mplt.plot_image(f_raw_slice, ax=ax, frac_thresh=frac_thresh, norm=norm, colorbar=True)
    plot_axes.format(xlabel=dim1, ylabel=dim2)
    string = '_output/slitscreen_slice_'
    for i in axis:
        string += f'_{dims[i]}-{ind_max[i]}'
    plt.savefig(string + '.png')
    # plt.show()

# ## Transformation to phase space coordinates
print('Transformation to phase space coordinates...')
# Convert x3 and y3 from pixels to mm.
cam_settings = ip.CameraSettings(cam)
cam_settings.set_zoom(info['cam_zoom'])
pix2mm_x = info['cam_pix2mm_x']
pix2mm_y = info['cam_pix2mm_y']
print(f"pix2mm_x = {pix2mm_x} (zoom = {info['cam_zoom']}, downscale={info['image_downscale']})")
print(f"pix2mm_y = {pix2mm_y} (zoom = {info['cam_zoom']}, downscale={info['image_downscale']})")
X3 = X3 * pix2mm_x
Y3 = Y3 * pix2mm_y


# Build the transfer matrices between the slits and the screen.
a2mm = 1.009  # assume same for both dipoles
rho = 0.3556
GL05 = 0.0
GL06 = 0.0
l1 = 0.0
l2 = 0.0
l3 = 0.774
L2 = 0.311  # slit2 to dipole face
l = 0.129  # dipole face to VS06 screen (assume same for first/last dipole-screen)
LL = l1 + l2 + l3 + L2  # distance from emittance plane to dipole entrance

ecalc = energy.EnergyCalculate(l1=l1, l2=l2, l3=l3, L2=L2, l=l, amp2meter=a2mm*1e3)
Mslit = ecalc.getM1()  # slit-slit
Mscreen = ecalc.getM()  # slit-screen


# Compute x', y', and energy w.
Y = Y1.copy()  # [mm]
YP = ecalc.calculate_yp(Y1 * 1e-3, Y3 * 1e-3, Mscreen)  # [rad]
YP *= 1e3  # [mrad]
print('Done with yp.')

X = X1.copy()  # [mm]
XP = ecalc.calculate_xp(X1 * 1e-3, X2 * 1e-3, Mslit)  # [rad]
XP *= 1e3  # [mrad]
print('Done with xp.')

W = ecalc.calculate_dE_screen(X3 * 1e-3, 0.0, X * 1e-3, XP * 1e-3, Mscreen)  # [MeV]
print('Done with w.')

del(X1, X2, Y1, X3, Y3)


# Make lists of centered phase space coordinate grids.
coords = [X, XP, Y, YP, W]
for coord in tqdm(coords):
    coord = coord - np.mean(coord)


# Interpolation
# ------------------------------------------------------------------------------
# It makes sense to increase the resolution along some axes of the interpolation grid since we are moving from a tilted grid to an regular grid grid.
new_shape = np.array(shape).astype(float)
new_shape[0] *= 1.1
new_shape[1] *= 1.8
new_shape[2] *= 1.1
new_shape = tuple(new_shape.astype(int))
info['int_shape'] = new_shape
print('shape of interpolated array f:', new_shape)

# Define regular phase space grid.
x_gv_new = np.linspace(np.min(X), np.max(X), new_shape[0])
xp_gv_new = np.linspace(np.min(XP), np.max(XP), new_shape[1])
y_gv_new = np.linspace(np.min(Y), np.max(Y), new_shape[2])
yp_gv_new = np.linspace(np.min(YP), np.max(YP), new_shape[3])
w_gv_new = np.linspace(np.min(W), np.max(W), new_shape[4])
new_coords = [x_gv_new, xp_gv_new, y_gv_new, yp_gv_new, w_gv_new]
utils.save_stacked_array('_output/coords.npz', new_coords)


# ### Test: put 2D projected phase spaces projection on upright grid
grid = True
contour = False
norm = None
gvs = [x_gv_new, xp_gv_new, y_gv_new, yp_gv_new]
pdims = ["x [mm]", "xp [mrad", "y [mm]", "yp [mrad]", "w [MeV]"]
for plane, (i, j) in zip(['x', 'y'], [(0, 1), (2, 3)]):
    if plane == 'x':
        U = X[:, :, 0, 0, 0]
        V = XP[:, :, 0, 0, 0]
    elif plane == 'y':
        U = Y[0, 0, :, :, 0]
        V = YP[0, 0, :, :, 0]
    H = utils.project(f_raw, axis=(i, j))
    H = H / np.max(H)
    
    points = (U.ravel(), V.ravel())
    values = H.ravel()
    U_new, V_new = np.meshgrid(gvs[i], gvs[j], indexing='ij')
    new_points = (U_new.ravel(), V_new.ravel())
    new_values = interpolate.griddata(points, values, new_points, fill_value=0.0, method='linear')
    H_new = new_values.reshape(len(gvs[i]), len(gvs[j]))
    print(f'H_new.min() = {H_new.min()}')
    H_new = np.clip(H_new, 0.0, None)
    H_new = H_new / np.max(H_new)

    fig, axes = pplt.subplots(ncols=2)
    mplt.plot_image(H, x=U, y=V, ax=axes[0], colorbar=True, norm=norm)
    mplt.plot_image(H_new, x=gvs[i], y=gvs[j], ax=axes[1], colorbar=True, norm=norm)
    if grid:
        kws = dict(c='grey', lw=0.4, alpha=0.5)
        for g in gvs[i]:
            axes[0].axvline(g, **kws)
        for g in gvs[j]:
            axes[0].axhline(g, **kws)
    if contour:
        axes[1].contour(U.T, V.T, H.T, color='white', alpha=0.2, lw=0.75)
    axes.format(xlabel=pdims[i], ylabel=pdims[j], toplabels=['Original', 'Interpolated'])
    plt.savefig('_output/interp2dsliceplane.png')
    # plt.show()

    
# Interpolate x-x'-w for each (y, y') 
print("interpolating x-x'-w for each y-y'")
temp_shape = (new_shape[0], new_shape[1], shape[2], shape[3], new_shape[4])
f = np.copy(f_raw)
f_new = np.zeros(temp_shape)
_X, _XP, _W = np.meshgrid(new_coords[0], new_coords[1], new_coords[4], indexing='ij')
new_points = (_X.ravel(), _XP.ravel(), _W.ravel())
for k in trange(shape[2]):
    for l in trange(shape[3]):
        idx = (slice(None), slice(None), k, l, slice(None))
        points = (
            coords[0][idx].ravel(),
            coords[1][idx].ravel(),
            coords[4][idx].ravel(),
        )
        values = f[idx].ravel()
        new_values = interpolate.griddata(
            points, 
            values, 
            new_points, 
            fill_value=0.0, 
            method='linear',
        )
        f_new[idx] = new_values.reshape((new_shape[0], new_shape[1], new_shape[4]))

# Redefine the grid coordinates.
X = np.zeros(temp_shape)
XP = np.zeros(temp_shape)
W = np.zeros(temp_shape)
for k in range(shape[2]):
    for l in range(shape[3]):
        idx = (slice(None), slice(None), k, l, slice(None))
        X[idx] = _X
        XP[idx] = _XP
        W[idx] = _W
coords = [X, XP, Y, YP, W]

print('Updated coordinate grid shapes:')
for C in coords:
    print(C.shape)

    
print("interpolating x-x'-w for each y-y'")
f = np.copy(f_raw)
f_new = np.memmap('_output/f.mmp', shape=new_shape, dtype='float', mode='w+') 
new_points = tuple([C.ravel() for C in np.meshgrid(new_coords[2], new_coords[3], indexing='ij')])
for i in trange(shape[0]):
    for j in trange(shape[1]):
        idx = (i, j, slice(None), slice(None), slice(None))
        points = (coords[2][idx].ravel(), coords[3][idx].ravel())
        values = f[idx].ravel()
        new_values = interpolate.griddata(
            points, 
            values, 
            new_points, 
            fill_value=0.0, 
            method='linear',
        )
        f_new[idx] = new_values.reshape((new_shape[2], new_shape[3]))
        

utils.save_pickle('_output/info.pkl', info)
file = open('_output/info.txt', 'w')
for key, value in info.items():
    file.write(f'{key}: {value}\n')
file.close()