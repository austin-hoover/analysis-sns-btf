#!/usr/bin/env python
# coding: utf-8

# # Step 1

# * Load scalar, waveform and image h5 files.
# * For each sweep, interpolate images on regular y grid. 
# * For each y and image pixel, interpolate x-x'. (f(x, x', y, y3, x3))
# * For each (x, xp, y, x3), interpolate yp. (f(x, x', y, y', x3))
# * For each (x, xp, y, yp), inteprolate w. (f(x, x', y, y', w)).

# In[1]:


import sys
import os
from os.path import join
import time
from datetime import datetime
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


# In[2]:


pplt.rc['grid'] = False
pplt.rc['cmap.discrete'] = False
pplt.rc['cmap.sequential'] = 'viridis'


# ## Setup

# In[3]:


folder = '_output'


# In[4]:


info = utils.load_pickle(join(folder, 'info.pkl'))
print('info')
pprint(info)


# In[5]:


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


# In[6]:


variables = info['variables']
keys = list(variables)
nsteps = np.array([variables[key]['steps'] for key in keys])

acts = info['acts']
print(acts)
points = np.vstack([data_sc[act] for act in acts]).T
points = points[:, ::-1]  # x, x2, y
nsteps = nsteps[::-1]

cam = info['cam']
print(f"cam = '{cam}'")
if cam.lower() not in ['cam06', 'cam34']:
    raise ValueError(f"Unknown camera name '{cam}'.")


# ## Convert to beam frame coordinates

# In[7]:


## y slit is inserted from above, always opposite y beam.
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

# In[8]:


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


# In[9]:


dipole_current = 0.0  # deviation of dipole current from nominal
l = 0.129  # dipole face to screen (assume same for first/last dipole-screen)
if cam.lower() == 'cam06':
    GL05 = 0.0  # QH05 integrated field strength (1 [A] = 0.0778 [Tm])
    GL06 = 0.0778  # QH06 integrated field strength (1 [A] = 0.0778 [Tm])
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

# In[10]:


points[:, 1] = 1e3 * ecalc.calculate_xp(points[:, 0] * 1e-3, points[:, 1] * 1e-3, Mslit) 


# Center points at zero.

# In[11]:


points -= np.mean(points, axis=0)


# Make grids.

# In[12]:


mins = np.min(points, axis=0)
maxs = np.max(points, axis=0)
scales = [1.1, 1.6, 1.1]
ns = np.multiply(scales, nsteps + 1).astype(int)
xgrid, xpgrid, ygrid = [np.linspace(umin, umax, n) 
                        for (umin, umax, n) in zip(mins, maxs, ns)]

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

# In[13]:


iterations = data_sc['iteration']
iteration_nums = np.unique(iterations)
n_iterations = len(iteration_nums)
kws = dict(kind='linear', copy=True, bounds_error=False, fill_value=0.0, assume_sorted=False)


# Interpolate y-y3-x3 image along y for each sweep.

# In[16]:
print('Interpolating y-y3-x3 images along y.')

images = data_im[cam + '_Image'].reshape((len(data_im), len(y3grid), len(x3grid)))
images_yy3x3 = []
for iteration in tqdm(iteration_nums):
    idx, = np.where(iterations == iteration)
    _points = points[idx, 2]
    _values = images[idx, :, :]
    _, uind = np.unique(_points, return_index=True)
    fint = interpolate.interp1d(_points[uind], _values[uind], axis=0, **kws)
    images_yy3x3.append(fint(ygrid))


# Convert y3 to y'.

# In[17]:
print('Converting y3 --> yp.')

images_yypx3 = []
for image_yy3x3 in tqdm(images_yy3x3):
    image_yypx3 = np.zeros((len(ygrid), len(ypgrid), len(x3grid)))
    for k in range(len(ygrid)):
        _points = YP[k]
        _values = image_yy3x3[k, :, :]
        fint = interpolate.interp1d(_points, _values, axis=0, **kws)
        image_yypx3[k, :, :] = fint(ypgrid)
    images_yypx3.append(image_yypx3)


# Convert x3 to w.

# In[18]:
print('Converting x3 --> w.')

XXP = []
images_yypw = []
for iteration, image_yypx3 in enumerate(tqdm(images_yypx3), start=1):
    x, xp = np.mean(points[iterations == iteration, :2], axis=0)
    _points = ecalc.calculate_dE_screen(1e-3 * x3grid, dipole_current, 1e-3 * x, 1e-3 * xp, Mscreen)
    _values = image_yypx3
    fint = interpolate.interp1d(_points, _values, axis=-1, **kws)
    images_yypw.append(fint(wgrid))
    XXP.append([x, xp])


# Since we move in vertical lines in the x-xp plane, we can separate the x and xp interpolations. If we moved in diagonal lines in the x-xp plane, we would need to perform a 2D interpolation for each y, yp, w; we currently do not do this. 2D interpolation is in the older notebooks in this directory... should add as an option here.

# In[19]:

XXP = np.zeros((n_iterations, 2))
for iteration in iteration_nums:
    idx, = np.where(iterations == iteration)
    XXP[iteration - 1] = np.mean(points[idx, :2], axis=0)
    
fig, ax = pplt.subplots(figwidth=4)
ax.scatter(XXP[:, 0], XXP[:, 1], c=np.arange(1, n_iterations + 1), s=2, cmap='flare_r',
           colorbar=True, colorbar_kw=dict(label='iteration'))
ax.format(xlabel="x [mm]", ylabel="xp [mrad]")
plt.savefig('_output/x-xp_iterations.png')
plt.show()


# We need to group the iterations by x step. To do this, loop through each iteration and check if x has changed by a significant amount (within the same x step, x will only change by very small amounts due to noise in the readback). Find a good number for this.

# In[21]:


max_abs_delta = 0.075
deltas = np.diff(points[:, 0])
fig, ax = pplt.subplots()
ax.hist(np.abs(deltas), bins=50, color='black')
ax.axvline(max_abs_delta, color='red')
ax.format(yscale='log', xlabel='delta_x [mm]')
plt.show()


# Group the iterations.

# In[22]:


X = []
steps = []
x_last = np.inf
for iteration in trange(1, n_iterations + 1):
    x, xp = XXP[iteration - 1]
    if np.abs(x - x_last) > max_abs_delta:
        X.append(x)
        steps.append([])
    steps[-1].append(iteration)
    x_last = x


# In[23]:


fig, ax = pplt.subplots()
for _iterations, x in zip(steps, X):
    ax.plot(_iterations, len(_iterations) * [x])
ax.format(xlabel='Iteration', ylabel='x [mm]')
plt.savefig('_output/x_groups.png')


# Interpolate each xp-y-yp-w image along xp.

# In[ ]:
print('Interpolating xp-y-yp-w images along xp.')

images_yypw = np.array(images_yypw)
images_xpyypw = []
for _iterations in tqdm(steps):
    idx = np.array(_iterations) - 1
    _points = XXP[idx, 1]
    _values = images_yypw[idx]
    fint = interpolate.interp1d(_points, _values, axis=0, **kws)
    images_xpyypw.append(fint(xpgrid))


# Interpolate the x-xp-y-yp-w image along x.

# In[ ]:
print('Interpolating x-xp-y-yp-w image long x')

# Create memory map
shape = (len(xgrid), len(xpgrid), len(ygrid), len(ypgrid), len(wgrid))
f = np.memmap(f'_output/f_{filename}.mmp', dtype='float', mode='w+', shape=shape)

# Interpolate 5D image along x.
_points = X
_values = images_xpyypw
fint = interpolate.interp1d(_points, _values, axis=0, **kws)
f[:, :, :, :, :] = fint(xgrid)

# Hack: flip x and x'.
if cam.lower() == 'cam34':
    f[:, :, :, :, :] = f[::-1, ::-1, :, :, :]

# Write changes to the memory map. 
f.flush()


# In[ ]:


coords = [xgrid, xpgrid, ygrid, ypgrid, wgrid]
coords = [c.copy() - np.mean(c) for c in coords]
utils.save_stacked_array(f'_output/coords_{filename}.npz', coords)


# Examine the array (Step 2 notebook makes more plots).

# In[ ]:


dims = ['x', 'xp', 'y', 'yp', 'w']
units = ['mm', 'mrad', 'mm', 'mrad', 'MeV']
dims_units = [f'{dim} [{unit}]' for dim, unit in zip(dims, units)]
prof_kws = dict(kind='step')
mplt.interactive_proj2d(f, coords=coords, dims=dims, units=units, prof_kws=prof_kws)


# In[ ]:


axes = mplt.corner(
    f,
    coords=coords,
    labels=dims_units,
    diag_kind='None',
    prof='edges',
    prof_kws=prof_kws,
)


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

print('info:')
print(info)


print('Done.')