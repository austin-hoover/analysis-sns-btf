import numpy as np


def avoid_repeats(a, pad=1e-7):
    """Function written by K. Ruisard to avoid repeating points in array.
    
    I'm nots sure why the counter is included.
    """
    repeat_idx, = np.where(np.diff(a) == 0)
    counter = 1
    for i in reversed(repeat_idx):
        a[i] -= pad * counter
        counter += 1
    return a


# Numpy arrays
# -----------------------------------------------------------------------------
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


def get_bins(coords, axis=0, n_bins=10, pad=0.1, n_bins_mult=None):
    """Return bin centers along the specified axis.
    
    Parameters
    ----------
    coords : ndarray, shape (n, d)
        An array of n points in d-dimensional space.
    dim : int
        The dimension index.
    n_bins : int
        The number of bins
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
    # Bin the points.
    bins = np.linspace(
        np.min(coords[:, axis]) - pad,
        np.max(coords[:, axis]) + pad,
        1 + n_bins * n_bins_mult,
    )
    counts, bins = np.histogram(coords[:, axis], bins=bins)
    # Assign a bin index to every point.
    idx = np.digitize(coords[:, axis], bins, right=False)
    idx_unique = np.unique(idx)
    for i in range(len(idx_unique)):
        idx[idx == idx_unique[i]] = i
    # Keep grid values (i.e. bin centers) of bins with nonzero counts.
    gv = bins[np.nonzero(counts)] + 0.5 * (bins[1] - bins[0])
    return gv, idx