import numpy as np
from scipy import ndimage
from scipy import interpolate

from .utils import avoid_repeats
from .utils import get_grid_coords


class CamSettings:
    def __init__(self, name):
        self.name = name
        self.name_lowercase = name.lower()
        self.ny = None
        self.nx = None
        if self.name_lowercase == 'cam34':
            self.ny = 512
            self.nx = 612
            

def camera_settings(cam):
    return CamSettings(cam)


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


def get_image_3d(images, points, grid, ny, nx, smooth=True):
    """Interpolate to obtain a 3D array from a set of 2D images.
    
    This should be extended to > 3 dimensions in the future.
    
    Parameters
    ----------
    images : ndarray, shape (m, ny * nx)
        Flattened camera images.
    points : ndarray, shape (m,)
        Positions on the w axis (w could be whatever), one per image.
    grid : ndarray, shape (g,)
        Grid points on the w axis.
    ny{nx} : float
        Number of rows{columns} in each images.
        
    Returns
    -------
    im3d : ndarray, shape (g, ny, nx)
    """
    grid_im_y = np.arange(ny)
    grid_im_x = np.arange(nx)
    n_frames = len(points)
    im3d = np.zeros((n_frames, ny, nx))
    for i, im in enumerate(images):
        im3d[i, :, :] = im.reshape(ny, nx)
    # Sort by increasing y value (needed for interpolation)
    idx_sort = np.argsort(points)
    points = points[idx_sort]
    im3d = im3d[idx_sort, :, :]
    # Deal with any repeating points (otherwise interpolate won't work).
    points = avoid_repeats(points, pad=1e-7)
    # Apply filter along 1st dimension.
    if smooth:
        im3d_smooth = ndimage.median_filter(im3d, size=(3, 1, 1), mode='nearest')
    else:
        im3d_smooth = im3d
    # Interpolate
    new_points = get_grid_coords(grid, grid_im_y, grid_im_x)
    im3d = interpolate.interpn(
        (points, grid_im_y, grid_im_x), 
        im3d_smooth, 
        new_points, 
        method='linear', 
        bounds_error=False, 
        fill_value=0.0,
    )   
    return im3d.reshape(len(grid), ny, nx)