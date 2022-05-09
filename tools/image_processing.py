import numpy as np



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