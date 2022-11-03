#!/usr/bin/env python
# coding: utf-8

# # Step 1

# * Load scalar, waveform and image h5 files.
# * For each sweep, interpolate images on regular y grid. 
# * For each y and image pixel, interpolate x-x'. (f(x, x', y, y3, x3))
# * For each (x, xp, y, x3), interpolate yp. (f(x, x', y, y', x3))
# * For each (x, xp, y, yp), inteprolate w. (f(x, x', y, y', w)).

# In[ ]:


import sys
import os
from os.path import join
import time
from datetime import datetime
import itertools
import importlib
import json
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
from ipywidgets import widgets

sys.path.append('../..')
from tools.energyVS06 import EnergyCalculate
from tools import image_processing as ip
from tools import plotting as mplt
from tools import utils


# In[ ]:


pplt.rc['grid'] = False
pplt.rc['cmap.discrete'] = False
pplt.rc['cmap.sequential'] = 'viridis'


# ## Setup

# In[ ]:


folder = '_saved/2022-07-15-VS06/'


# In[ ]:


info = utils.load_pickle(join(folder, 'info.pkl'))
print('info')
pprint(info)


# In[ ]:


# Save as pickled dictionary for loading.
utils.save_pickle('_output/info.pkl', info)

# Save as file for viewing.
file = open('_output/info.txt', 'w')
for key, value in info.items():
    file.write(f'{key}: {value}\n')
file.close()


# ### TEMP: correct for bad pix2mm.**
# 
# Cam06 was replaced in April 2022, and we did not get a mm/pixel calibration. Apply correction factor estimated by K. Ruisard.

# In[ ]:


fac = 0.4006069802731412
info['image_pix2mm_x'] *= fac
info['image_pix2mm_y'] *= fac


# End TEMP

# In[ ]:


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


# In[ ]:


variables = info['variables']
keys = list(variables)
nsteps = np.array([variables[key]['steps'] for key in keys])

acts = info['acts']
print(acts)
points = np.vstack([data_sc[act] for act in acts]).T
points = points[:, ::-1]  # y, x2, x --> x, x2, y
nsteps = nsteps[::-1]

cam = info['cam']
print(f"cam = '{cam}'")
if cam.lower() not in ['cam06', 'cam34']:
    raise ValueError(f"Unknown camera name '{cam}'.")


# ## Convert to beam frame coordinates

# In[ ]:


## y_slit is inserted from above, always opposite y_beam.
points[:, 2] = -points[:, 2]

## Screen coordinates (x3, y3) are always opposite beam (x, y).
image_shape = info['image_shape']
x3grid = -np.arange(image_shape[1]) * info['image_pix2mm_x'] 
y3grid = -np.arange(image_shape[0]) * info['image_pix2mm_y']

# VT04/VT06 are same sign as x_beam; VT34a and VT34b are opposite.
if cam.lower() == 'cam34':
    points[:, :2] = -points[:, :2]


# ## Setup interpolation grids

# Build the transfer matrices between the slits and the screen. 

# In[ ]:


# Eventually switch to saving metadata dict as pickled file in Step 0, then loading here.
_file = h5py.File(join(datadir, filename + '.h5'), 'r')
if 'config' in _file:
    config = _file['config']
    metadata = dict()
    for name in config['metadata'].dtype.names:
        metadata[name] = config['metadata'][name]
else:
    # Older measurement; metadata is in json file.
    metadata = json.load(open(join(datadir, filename + '-metadata.json'), 'r'))
    _metadata = dict()
    for _dict in metadata.values():
        for key, value in _dict.items():
            _metadata[key] = value
    metadata = _metadata
pprint(metadata)
_file.close()


# In[ ]:


dipole_current = 0.0  # deviation of dipole current from nominal
l = 0.129  # dipole face to screen (assume same for first/last dipole-screen)
if cam.lower() == 'cam06':
    GL05 = 0.0  # QH05 integrated field strength (1 [A] = 0.0778 [Tm])
    GL06 = 0.0  # QH06 integrated field strength (1 [A] = 0.0778 [Tm])
    l1 = 0.280  # slit1 to QH05 center
    l2 = 0.210  # QH05 center to QV06 center
    l3 = 0.457  # QV06 center to slit2
    L2 = 0.599  # slit2 to dipole face    
    rho_sign = +1.0  # dipole bend radius sign
    if GL05 == 0.0 and metadata['BTF_MEBT_Mag:PS_QH05:I_Set'] != 0.0:
        print('Warning: QH05 is turned on according to metadata.')
    if GL05 != 0.0 and metadata['BTF_MEBT_Mag:PS_QH05:I_Set'] == 0.0:
        print('Warning: QH05 is turned off according to metadata.')
    if GL06 == 0.0 and metadata['BTF_MEBT_Mag:PS_QV06:I_Set'] != 0.0:
        print('Warning: QH06 is turned on according to metadata.')
    if GL06 != 0.0 and metadata['BTF_MEBT_Mag:PS_QV06:I_Set'] == 0.0:
        print('Warning: QH06 is turned off according to metadata.')
elif cam.lower() == 'cam34':
    GL05 = 0.0  # QH05 integrated field strength
    GL06 = 0.0  # QH06 integrated field strength
    l1 = 0.000  # slit1 to QH05 center
    l2 = 0.000  # QH05 center to QV06 center
    l3 = 0.774  # QV06 center to slit2
    L2 = 0.311  # slit2 to dipole face
    # Weird... I can only get the right answer for energy if I *do not* flip rho,
    # x1, x2, and x3. I then flip x and xp at the very end.
    rho_sign = +1.0  # dipole bend radius sign
    x3grid = -x3grid
    points[:, :2] = -points[:, :2]
LL = l1 + l2 + l3 + L2  # distance from emittance plane to dipole entrance
ecalc = EnergyCalculate(l1=l1, l2=l2, l3=l3, L2=L2, l=l, rho_sign=rho_sign)
Mslit = ecalc.getM1(GL05=GL05, GL06=GL06)  # slit-slit
Mscreen = ecalc.getM(GL05=GL05, GL06=GL06)  # slit-screen


# Convert to x'.

# In[ ]:


points[:, 1] = 1e3 * ecalc.calculate_xp(points[:, 0] * 1e-3, points[:, 1] * 1e-3, Mslit) 


# Center points at zero.

# In[ ]:


points -= np.mean(points, axis=0)


# Make grids.

# In[ ]:


mins = np.min(points, axis=0)
maxs = np.max(points, axis=0)
scales = [1.1, 1.6, 1.1]
ns = np.multiply(scales, nsteps + 1).astype(int)
xgrid, xpgrid, ygrid = [np.linspace(umin, umax, n) for (umin, umax, n) in zip(mins, maxs, ns)]

YP = np.zeros((len(ygrid), len(y3grid)))
for k, y in enumerate(ygrid):
    YP[k] = 1e3 * ecalc.calculate_yp(1e-3 * y, 1e-3 * y3grid, Mscreen)
ypgrid = np.linspace(np.min(YP), np.max(YP), int(1.1 * len(y3grid)))

W = np.zeros((len(xgrid), len(xpgrid), len(x3grid)))
for i, x in enumerate(xgrid):
    for j, xp in enumerate(xpgrid):
        W[i, j] = ecalc.calculate_dE_screen(1e-3 * x3grid, dipole_current, 1e-3 * x, 1e-3 * xp, Mscreen)
wgrid = np.linspace(np.min(W), np.max(W), int(1.1 * len(x3grid)))


# ## Interpolate 

# In[ ]:


iterations = data_sc['iteration'].copy()
iteration_nums = np.unique(iterations)
n_iterations = len(iteration_nums)
kws = dict(kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)


# ### Interpolate y

# In[ ]:

print("Interpolating y.")
images_yy3x3 = []
for iteration in tqdm(iteration_nums):
    idx, = np.where(iterations == iteration)
    _points = points[idx, 2]
    _values = data_im[idx, cam + '_Image'].reshape((len(idx), len(y3grid), len(x3grid)))
    _, uind = np.unique(_points, return_index=True)
    fint = interpolate.interp1d(_points[uind], _values[uind], axis=0, **kws)
    images_yy3x3.append(fint(ygrid))


# ### Interpolate y'

# In[ ]:

print("Interpolating y'.")
images_yypx3 = []
for image_yy3x3 in tqdm(images_yy3x3):
    image_yypx3 = np.zeros((len(ygrid), len(ypgrid), len(x3grid)))
    for k in range(len(ygrid)):
        _points = YP[k]
        _values = image_yy3x3[k, :, :]
        fint = interpolate.interp1d(_points, _values, axis=0, **kws)
        image_yypx3[k, :, :] = fint(ypgrid)
    images_yypx3.append(image_yypx3)
del(images_yy3x3)


# ### Interpolate w

# Also save x,x' on each sweep.

# In[ ]:

print("Interpolating w.")
XXP = []
images_yypw = []
for iteration, image_yypx3 in enumerate(tqdm(images_yypx3), start=1):
    x, xp = np.mean(points[iterations == iteration, :2], axis=0)
    _points = ecalc.calculate_dE_screen(1e-3 * x3grid, dipole_current, 1e-3 * x, 1e-3 * xp, Mscreen)
    _values = image_yypx3
    fint = interpolate.interp1d(_points, _values, axis=-1, **kws)
    images_yypw.append(fint(wgrid))
    XXP.append([x, xp])
del(images_yypx3)
XXP = np.array(XXP)
images_yypw = np.array(images_yypw)


# In[ ]:


fig, ax = pplt.subplots(figwidth=4)
ax.scatter(XXP[:, 0], XXP[:, 1], c=np.arange(1, n_iterations + 1), s=2, cmap='flare_r',
           colorbar=True, colorbar_kw=dict(label='iteration'))
ax.format(xlabel="x [mm]", ylabel="xp [mrad]")
plt.savefig('_output/x-xp_iterations')
plt.show()


# ### Interpolate x-x'

# Since we move in vertical lines in the x-xp plane, we could separate the x and xp interpolations. If we moved in diagonal lines in the x-xp plane, we would need to perform a 2D interpolation for each y, yp, w. The following variables determines which method to use.

# In[ ]:


xxp_interp = '2D'  # {'1D', '2D'}


# Try grouping iterations by x step by looping through each iteration and checking if x has changed significantly (within each x step, x should only change by a small amount due to noise in the readback). Find a good cutoff `max_abs_delta`.

# In[ ]:


max_abs_delta = 0.05
X, steps = [], []
x_last = np.inf
for iteration in trange(1, n_iterations + 1):
    x, xp = XXP[iteration - 1]
    if np.abs(x - x_last) > max_abs_delta:
        X.append(x)
        steps.append([])
    steps[-1].append(iteration)
    x_last = x

fig, ax = pplt.subplots(figsize=(4, 2))
ax.hist(np.abs(np.diff(points[:, 0])), bins=50, color='black')
ax.axvline(max_abs_delta, color='red')
ax.format(yscale='log', xlabel=r'$\Delta x$ [mm]', ylabel='Number of steps')
plt.savefig('_output/delta_x')
plt.show()

fig, ax = pplt.subplots(figsize=(4, 2))
for _iterations in steps:
    _idx = np.array(_iterations) - 1
    ax.scatter(XXP[_idx][:, 0], XXP[_idx][:, 1], s=1)
    ax.format(xlabel="x [mm]", ylabel="x' [mrad]")
plt.savefig('_output/x_groups')
plt.show()


# Run the interpolation.

# In[ ]:

shape = (len(xgrid), len(xpgrid), len(ygrid), len(ypgrid), len(wgrid))
f = np.memmap(f'_output/f_{filename}.mmp', dtype='float', mode='w+', shape=shape)


# In[ ]:


if xxp_interp == '1D':
    # Interpolate x'-y-y'-w image along x'
    print("Interpolating x'.")
    images_xpyypw = []
    for _iterations in tqdm(steps):
        idx = np.array(_iterations) - 1
        _points = XXP[idx, 1]
        _values = images_yypw[idx]
        fint = interpolate.interp1d(_points, _values, axis=0, **kws)
        images_xpyypw.append(fint(xpgrid))
    del(images_yypw)

    # Interpolate the xp-y-yp-w image stack along x.
    print("Interpolating x.")
    ## Passing a very large array to `scipy.interpolate.interp1d` can give 
    ## memory errors for large arrays. In that case, loop through {x', y, y', w}
    ## and perform a 1D interpolation at each index. 
    n_loop = 1
    _points = X
    if n_loop == 0:
        _values = images_xpyypw
        fint = interpolate.interp1d(_points, _values, axis=0, **kws)
        f[:, j, k, l, m] = fint(xgrid)
    else:
        images_xpyypw = np.array(images_xpyypw)
        axis = list(range(1, n_loop + 1))
        ranges = [range(s) for s in shape[1: n_loop + 1]]
        for ind in tqdm(itertools.product(*ranges)):
            idx = utils.make_slice(5, axis=axis, ind=ind)
            _values = images_xpyypw[idx]
            fint = interpolate.interp1d(_points, _values, axis=0, **kws)
            f[idx] = fint(xgrid) 
        del(images_xpyypw)
else:
    # 2D interpolation of x-x' for each {y, y', w}.
    print("Interpolating x-x'.")
    _points = XXP
    _new_points = utils.get_grid_coords(xgrid, xpgrid)
    for k in trange(shape[2]):
        for l in trange(shape[3]):
            for m in range(shape[4]):
                _values = images_yypw[:, k, l, m]
                new_values = interpolate.griddata(
                    _points,
                    _values,
                    _new_points,
                    method='linear',
                    fill_value=False,
                )
                f[:, :, k, l, m] = new_values.reshape((shape[0], shape[1]))


# ## Shutdown

# Hack: flip x-x' if we are at Cam34.

# In[ ]:


if cam.lower() == 'cam34':
    ## This may give a memory error...
    f[:, :, :, :, :] = f[::-1, ::-1, :, :, :] 
    
    ## This should not...
    # for k in trange(shape[2]):
    #     f[:, :, k, :, :] = f[::-1, ::-1, k, :, :]


# Write changes to the memory map.

# In[ ]:


f.flush()


# Save the grid coordinates.

# In[ ]:


coords = [xgrid, xpgrid, ygrid, ypgrid, wgrid]
coords = [c.copy() - np.mean(c) for c in coords]
utils.save_stacked_array(f'_output/coords_{filename}.npz', coords)


# Briefly examine the interpolated array.

# In[ ]:


dims = ["x", "x'", "y", "y'", "w"]
units = ['mm', 'mrad', 'mm', 'mrad', 'MeV']
dims_units = [f'{dim} [{unit}]' for dim, unit in zip(dims, units)]
prof_kws = dict(kind='step')
mplt.interactive_proj2d(f, coords=coords, dims=dims, units=units, prof_kws=prof_kws)


# Save info.

# In[ ]:


info['dims'] = dims
info['units'] = units
info['int_shape'] = shape

# Save as pickled dictionary for loading.
utils.save_pickle('_output/info.pkl', info)

# Save as file for viewing.
file = open('_output/info.txt', 'w')
for key, value in info.items():
    file.write(f'{key}: {value}\n')
file.close()


file.close()