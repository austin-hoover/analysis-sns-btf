import numpy as np


def avoid_repeats(array, pad=1e-7):
    """Avoid repeating points in an array.
    
    Adds a small random number to each duplicate element.
    """
    repeat_idx, = np.where(np.diff(a) == 0)
    for i in reversed(repeat_idx):
        array[i] += np.random.uniform(-pad, pad)
    return array


def apply(M, X):
    """Apply M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def max_indices(array):
    """Return indices of maximum element in array."""
    return np.unravel_index(np.argmax(array), array.shape)    


def slice_array(array, axis=0, ind=0):
    """Slice array along one or more axes.
    
    array : ndarray
        The array to slice.
    axis : int or list
        The axis(axes) along which to slice.
    ind : int or list
        Locations on the specified axis(axes).
    """
    idx = array.ndim * [slice(None)]
    if type(axis) is int:
        axis = [axis]
    if type(ind) is int:
        ind = [ind]
    for i, k in zip(ind, axis):
        idx[k] = i
    return array[tuple(idx)]


def project(array, axis=0):
    """Project array onto one or more axes.
    
    array : ndarray
        The distribution.
    axis : int or list
        Axis(axes) along which to project the array.
    """
    if type(axis) is int:
        axis = [axis]
    sum_axis = tuple([k for k in range(array.ndim) if k not in axis])
    return np.sum(array, axis=sum_axis)


def get_grid_coords(*xi, indexing='ij'):
    """Return array of shape (N, D), where N is the number of points on 
    the grid and D is the number of dimensions."""
    return np.vstack([X.ravel() for X in np.meshgrid(*xi, indexing=indexing)]).T


def snap(array, n=165, pad=0.1):
    """[Description here.]
    
    Parameters
    ----------
    array : ndarray, shape (n,)
    n_bins : int
        The number of bins.
    n_bins_mult : int
        Multiplicative factor on n_bins. I need to figure out why this is
        necessary. Quote from K. Ruisard: "This parameter can be tuned
        for good performance."
        
    Returns
    -------
    gv : ndarray, shape (?)
        The grid values, i.e., bin centers.
    idx : ndarray, shape (?)
    """
    bins = np.linspace(np.min(array) - pad, np.max(array) + pad, 1 + n)
    counts, bins = np.histogram(array, bins=bins)
    idx = np.digitize(array, bins, right=False)
    idx_unique = np.unique(idx)
    for i in range(len(idx_unique)):
        idx[idx == idx_unique[i]] = i
    gv = bins[np.nonzero(counts)] + 0.5 * (bins[1] - bins[0])
    return gv, idx