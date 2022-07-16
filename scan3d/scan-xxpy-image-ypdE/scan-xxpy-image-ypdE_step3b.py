#!/usr/bin/env python
# coding: utf-8

# # Step 3b: Interpolate 5D phase space density on a regular grid

# In[ ]:


import sys
import os
from os.path import join
import time
from datetime import datetime
import importlib
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


# In[ ]:


pplt.rc['grid'] = False
pplt.rc['cmap.discrete'] = False
pplt.rc['cmap.sequential'] = 'viridis'


# ## Load data

# In[ ]:


folder = '_output'
filenames = os.listdir(folder)
for filename in filenames:
    if filename.startswith('f_raw') or filename.startswith('coordinates3d_raw'):
        print(filename)


# In[ ]:


filename = 'f_raw_220429190854-scan-xxpy-image-ypdE.mmp'
coordfilename = 'coordinates3d_raw_220429190854-scan-xxpy-image-ypdE.npy'


# In[ ]:


info = utils.load_pickle('_output/info.pkl')
info


# In[ ]:


shape = info['rawgrid_shape']  # (x1, x2, y1, y3, x3)
dtype = info['im_dtype']
cam = info['cam']


# In[ ]:


f_raw = np.memmap(join(folder, filename), shape=shape, dtype=dtype, mode='r')
print(np.info(f_raw))


# Use the Right Hand Rule to determine the beam coordinates. [Insert image here]. (NEED TO FIX DIPOLE TRANSFER MATRIX FOR VS34; CURRENT CALCULATION IS WRONG AND IS CAUSING SIGN ERROR.
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

# In[ ]:


if cam.lower() == 'cam06':
    f_raw = f_raw[:, :, ::-1, ::-1, ::-1]
elif cam.lower() == 'cam34':
    # a5d = a5d[::-1, ::-1, ::-1, ::-1, :]
    f_raw = f_raw[:, :, ::-1, ::-1, :]


# ## Load slit coordinates

# In[ ]:


coords_3d = np.load(join(folder, coordfilename))  # [X1, X2, Y1]
coords_3d.shape


# In[ ]:


dims = ["x1", "x2", "y1", "y3", "x3"]
dim_to_int = {dim: i for i, dim in enumerate(dims)}


# In[ ]:


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
plt.savefig('coordinates3d.png')


# Copy the grids into the five-dimensional space.

# In[ ]:


X1, X2, Y1 = coords_3d
X1 = utils.copy_into_new_dim(X1, shape[3:], axis=-1)
X2 = utils.copy_into_new_dim(X2, shape[3:], axis=-1)
Y1 = utils.copy_into_new_dim(Y1, shape[3:], axis=-1)


# In[ ]:


print('X1.shape =', X1.shape)
print('X2.shape =', X2.shape)
print('Y1.shape =', Y1.shape)


# In[ ]:


Y3, X3 = np.meshgrid(np.arange(shape[3]), np.arange(shape[4]), indexing='ij')
Y3 = utils.copy_into_new_dim(Y3, shape[:3], axis=0)
X3 = utils.copy_into_new_dim(X3, shape[:3], axis=0)


# In[ ]:


print('Y3.shape =', Y3.shape)
print('X3.shape =', X3.shape)


# Make lists of centered coordinates `coords_`.

# In[ ]:


X1 = X1 - np.mean(X1)
X2 = X2 - np.mean(X2)
Y1 = Y1 - np.mean(Y1)
Y3 = Y3 - np.mean(Y3)
X3 = X3 - np.mean(X3)
coords_ = [X1, X2, Y1, Y3, X3]


# In[ ]:


for i, dim in enumerate(dims):
    print('dim =', dim)
    U = coords_[i]
    axes = [k for k in range(U.ndim) if k != i]
    idx = utils.make_slice(U.ndim, axes, ind=[0, 0, 0, 0])
    print(U[idx])
    print()


# ## View 5D array in slit-screen coordinates

# Correlation between planes are removed... units are dimensionless.

# In[ ]:


f_raw_max = np.max(f_raw)
f_raw_min = np.min(f_raw)
if f_raw_min < 0:
    print(f'min(f_raw) = {f_raw_min}. Clipping to 0.')
    f_raw = np.clip(f_raw, 0, None)


# In[ ]:


mplt.interactive_proj2d(f_raw / f_raw_max, dims=['x1', 'x2', 'y1', 'y3', 'x3'], 
                        slider_type='int', default_ind=(4, 3))


# ## Transformation to phase space coordinates

# Convert x3 and y3 from pixels to mm.

# In[ ]:


cam_settings = ip.CameraSettings(cam)
cam_settings.set_zoom(info['cam_zoom'])
pix2mm_x = info['cam_pix2mm_x']
pix2mm_y = info['cam_pix2mm_y']
print(f"pix2mm_x = {pix2mm_x} (zoom = {info['cam_zoom']}, downscale={info['image_downscale']})")
print(f"pix2mm_y = {pix2mm_y} (zoom = {info['cam_zoom']}, downscale={info['image_downscale']})")


# In[ ]:


X3 = X3 * pix2mm_x
Y3 = Y3 * pix2mm_y


# Build the transfer matrices between the slits and the screen.

# In[ ]:


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


# Compute x', y', and energy deviation w.

# In[ ]:


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


# In[ ]:


del(X1, X2, Y1, X3, Y3)


# Make lists of centered phase space coordinate grids.

# In[ ]:


coords = [X, XP, Y, YP, W]
for coord in tqdm(coords):
    coord = coord - np.mean(coord)


# ## Interpolation 

# It makes sense to increase the resolution along some axes of the interpolation grid since we are moving from a tilted grid to an regular grid grid.

# In[ ]:


M = info['M']
M


# In[ ]:


new_shape = np.array(shape).astype(float)
new_shape[0] *= 1.1
new_shape[1] *= 2.0
new_shape[2] *= 1.1
new_shape = tuple(new_shape.astype(int))
info['int_shape'] = new_shape
print(new_shape)


# In[ ]:


x_gv_new = np.linspace(np.min(X), np.max(X), new_shape[0])
xp_gv_new = np.linspace(np.min(XP), np.max(XP), new_shape[1])
y_gv_new = np.linspace(np.min(Y), np.max(Y), new_shape[2])
yp_gv_new = np.linspace(np.min(YP), np.max(YP), new_shape[3])
w_gv_new = np.linspace(np.min(W), np.max(W), new_shape[4])
new_coords = [x_gv_new, xp_gv_new, y_gv_new, yp_gv_new, w_gv_new]
utils.save_stacked_array('_output/coords.npz', new_coords)


# ### Test: put 2D projected phase spaces projection on upright grid

# In[ ]:


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
    plt.savefig('_output/interp2d_test.png')


# ### Interpolate w for each (x, x', y, y').

# In[ ]:


f = np.copy(f_raw)
f_new = np.zeros((shape[0], shape[1], shape[2], shape[3], new_shape[4]))
new_points = new_coords[4]
for i in trange(shape[0]):
    for j in trange(shape[1]):
        for k in range(shape[2]):
            for l in range(shape[3]):
                idx = (i, j, k, l, slice(None))
                points = coords[4][idx].ravel()
                values = f[idx].ravel()
                f_new[idx] = interpolate.griddata(
                    points, 
                    values, 
                    new_points, 
                    fill_value=0.0, 
                    method='linear',
                )        


# Redefine the grid coordinates: copy the x, x', y, and y' grids along the new w axis.

# In[ ]:


idx = (slice(None), slice(None), slice(None), slice(None), 0)
X = utils.copy_into_new_dim(X[idx], (new_shape[4],), axis=-1)
Y = utils.copy_into_new_dim(Y[idx], (new_shape[4],), axis=-1)
XP = utils.copy_into_new_dim(XP[idx], (new_shape[4],), axis=-1)
YP = utils.copy_into_new_dim(YP[idx], (new_shape[4],), axis=-1)
W = utils.copy_into_new_dim(new_coords[4], (shape[0], shape[1], shape[2], shape[3]), axis=0)
coords = [X, XP, Y, YP, W]


# In[ ]:


for C in coords:
    print(C.shape)


# ### Interpolate x-x' for each (y, y', w)

# In[ ]:


f = np.copy(f_new)
f_new = np.zeros((new_shape[0], new_shape[1], shape[2], shape[3], new_shape[4]))
new_points = tuple([C.ravel() for C in np.meshgrid(new_coords[0], new_coords[1], indexing='ij')])
for k in trange(shape[2]):
    for l in trange(shape[3]):   
        for m in trange(new_shape[4]):
            idx = (slice(None), slice(None), k, l, m)
            points = (coords[0][idx].ravel(), coords[1][idx].ravel())
            values = f[idx].ravel()
            new_values = interpolate.griddata(
                points,
                values,
                new_points,
                fill_value=0.0,
                method='linear',
            )
            f_new[idx] = new_values.reshape((new_shape[0], new_shape[1]))


# Same thing with the coordinates. We now need to copy the x-x' grid along all other dimensions, and y, y', and w along the x and x' dimensions.

# In[ ]:


_X, _XP = np.meshgrid(new_coords[0], new_coords[1], indexing='ij')
X = utils.copy_into_new_dim(_X, (shape[2], shape[3], new_shape[4]), axis=-1)
XP = utils.copy_into_new_dim(_XP, (shape[2], shape[3], new_shape[4]), axis=-1)
Y = utils.copy_into_new_dim(Y[0, 0, :, :, :], (new_shape[0], new_shape[1]), axis=0)
YP = utils.copy_into_new_dim(YP[0, 0, :, :, :], (new_shape[0], new_shape[1]), axis=0)
W = utils.copy_into_new_dim(W[0, 0, :, :, :], (new_shape[0], new_shape[1]), axis=0)
coords = [X, XP, Y, YP, W]


# In[ ]:


for C in coords:
    print(C.shape)


# ### Interpolate y-y' for each (x, x', w)

# In[ ]:


f = f_new.copy()
f_new = np.memmap('_output/f.mmp', shape=new_shape, dtype='float', mode='w+') 
new_points = tuple([G.ravel() for G in np.meshgrid(y_gv_new, yp_gv_new, indexing='ij')])
for i in trange(new_shape[0]):
    for j in trange(new_shape[1]):   
        for m in trange(new_shape[4]):
            idx = (i, j, slice(None), slice(None), m)
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


# In[ ]:


del f_new


# In[ ]:


utils.save_pickle('_output/info.pkl', info)
file = open('_output/info.txt', 'w')
for key, value in info.items():
    file.write(f'{key}: {value}\n')
file.close()


# In[ ]:




