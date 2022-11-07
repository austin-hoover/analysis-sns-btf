import numpy as np
from scipy import ndimage
from scipy import interpolate
from skimage import transform
import proplot as pplt

from . import utils


class CameraSettings:
    def __init__(self, name):
        self.name = name
        self.name_lowercase = name.lower()
        self.shape = None
        self.pix2mm = None 
        self.zoom = 1.0
        if self.name_lowercase == 'cam06':
            self.shape = (512, 612)
            self.pix2mm = 0.0274  # zoom=1.0
        if self.name_lowercase == 'cam06_old':
            self.shape = (258, 346)
            self.pix2mm = 0.0659  # zoom=1.0
        elif self.name_lowercase == 'cam34':
            self.shape = (512, 612)
            self.pix2mm = 0.050  # zoom=1.0
        self.ny, self.nx = self.shape
        
    def set_zoom(self, zoom):
        self.zoom = zoom
        self.pix2mm /= zoom
    

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


def downscale(image, down=1):
    return skimage.downscale_local_mean(image, (down, down))


def to_uint8(image, cmap):
    if type(cmap) is str:
        cmap = pplt.Colormap(cmap)
    return np.uint8(cmap(image) * np.iinfo(np.uint8).max)