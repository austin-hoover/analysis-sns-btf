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


def crop(image, y1=None, y2=None, x1=None, x2=None):
    return image[y1:y2, x1:x2]


def thresh(image, thresh=None, val=0, mask=False):
    im = np.copy(image)
    if thresh:
        idx = image < thresh
        if mask:
            im = np.ma.masked_array(im, mask=idx)
        else:
            im[image < thresh] = val
    return im


def interpolate_3d(im3d, points, grid, ny, nx, smooth=True):
    if len(im3d) != len(points):
        raise ValueError('Must have same number of images and points.')
    n_frames, ny, nx = im3.shape
    grid_im_y = np.arange(ny)
    grid_im_x = np.arange(nx)
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
    # Interpolate.
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