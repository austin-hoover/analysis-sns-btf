import numpy as np
from scipy import ndimage
from scipy import interpolate

from . import utils


class CamSettings:
    def __init__(self, name):
        self.name = name
        self.name_lowercase = name.lower()
        self.shape = None
        self.pix2mm = None 
        self.zoom = 1.0
        if self.name_lowercase == 'cam06':
#             self.shape = (258, 346)
            self.shape = (512, 612)
            self.pix2mm = 0.0659  # at zoom=1.0?
        elif self.name_lowercase == 'cam34':
            self.shape = (512, 612)
            self.pix2mm = 0.05  # at zoom=1.0
        self.ny, self.nx = self.shape
        
    def set_zoom(self, zoom):
        self.zoom = zoom
        self.pix2mm /= zoom
            

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


def interp_along_axis(image, x, xnew, axis=0, **kws):
    """Interpolate N-dimensional image along along one axis.
    
    It just calls `scipy.interpolate.interp1d`. Before doing so, `x` and
    `image` are flipped if `x` is decreasing (such as when a slit is moving 
    backwards). Also, we make sure that `x` has no duplicate elements.
    
    Parameters
    ----------
    array : ndarray, shape (..., N, ...)
        An N-D array of real values. The length along the interpolation 
        axis must be equal to the length of x.
    x : ndarray, shape (N,)
        The coordinates of the data points along the interpolation axis.
    xnew : ndarray, shape (M,)
        The coordinates at which to evaluate the interpolated image along
        the interpolation axis.
    axis : int
        The axis of `image` along which to interpolate.
    **kws
        Key word arguments to be passes to `scipy.interpolate.interp1d`.
    """
    kws.setdefault('bounds_error', False)
    kws.setdefault('fill_value', 0.0)
    idx_sort = np.argsort(x)
    x = x[idx_sort]
    image = image[idx_sort, :, :]
    x = utils.avoid_repeats(x)
    f = interpolate.interp1d(x, image, axis=0, **kws)
    return f(xnew)