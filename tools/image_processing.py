import numpy as np
from scipy import ndimage
from scipy import interpolate

from . import utils


def crop(image, l=None, r=None, b=None, t=None):
    return image[b:-t, l:-r]


def thresh(image, thresh=None, val=0, mask=False):
    im = np.copy(image)
    if thresh:
        idx = image < thresh
        if mask:
            im = np.ma.masked_array(im, mask=idx)
        else:
            im[image < thresh] = val
    return im


def get_image_3d(ys, images, gridy, ny, nx, smooth=True):
    """Create 3D array from a set of screen images.
    
    Each image corresponds to a different position of the y slit.
    
    ys : ndarray, shape (n,)
        Y slit positions corresponding to each image.
    images : ndarray, shape (n, ny * nx)
        Flattened camera images.
    gridy : ndarray, shape
        Grid values along the y axis.
    ny{nx} : float
        Number of rows{columns} in the images.
    """
    ypix = np.arange(ny)
    xpix = np.arange(nx)
    # Build up the 3D image (camera image for every data-point in one sweep).
    n_frames = len(ys)
    im3d = np.zeros((n_frames, ny, nx))
    for i, im in enumerate(images):
        im3d[i, :, :] = im.reshape(ny, nx)
    # Sort by increasing y value (needed for interpolation)
    idx_sort = np.argsort(ys)
    ys = ys[idx_sort]
    im3d = im3d[idx_sort, :, :]
    # Deal with any repeating points (otherwise interpolate won't work)
    ys = utils.avoid_repeats(ys, pad=1e-7)
    # Apply filter along 1st dimension.
    im3d_smooth = im3d
    if smooth:
        im3d_smooth = ndimage.median_filter(im3d, size=(3,1,1), mode='nearest')
    # Interpolate
    Y, YP, X = np.meshgrid(gridy, ypix, xpix, indexing='ij')
    new_points = np.vstack([Y.ravel(), YP.ravel(), X.ravel()]).T
    a3d = interpolate.interpn(
        (ys, ypix, xpix), 
        im3d_smooth, 
        new_points, 
        method='linear', 
        bounds_error=False, 
        fill_value=0.0,
    )   
    a3d = a3d.reshape(len(gridy), ny, nx)
    return a3d