import numpy as np


def avoid_repeats(array, pad=1e-7):
    """Avoid repeating points in an array.
    
    Adds a small number to each duplicate element.
    """
    repeat_idx, = np.where(np.diff(array) == 0)
    counter = 1
    for i in reversed(repeat_idx):
        array[i] -= pad * counter
        counter += 1
    return array


def apply(M, X):
    """Apply M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def max_indices(array):
    """Return indices of maximum element in array."""
    return np.unravel_index(np.argmax(array), array.shape)    


def flatten_index(ind, shape):
    """Return index in flattend array from multi-dimensional index.
    
    Example
    -------
    >>> x = np.array([[1.0, 1.5, 3.0], [-1.4, 3.6, 2.1]])
    >>> ind = (1, 2)
    >>> i = flatten_array_index(ind, x.shape)
    >>> print(i, x.ravel()[i], x[ind])
    5 2.1 2.1
    """
    multi_index = tuple([[i,] for i in ind])
    i, = np.ravel_multi_index(multi_index, shape)
    return i


def copy_into_new_dim(array, shape, axis=-1, method='broadcast', copy=False):
    """Copy an array into one or more new dimensions.
    
    The 'broadcast' method is much faster since it works with views instead of copies. 
    See 'https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times'
    """
    if type(shape) is int:
        shape = (shape,)
    if method == 'repeat':
        for i in range(len(shape)):
            array = np.repeat(np.expand_dims(array, axis), shape[i], axis=axis)
        return array
    elif method == 'broadcast':
        if axis == 0:
            new_shape = shape + array.shape
        elif axis == -1:
            new_shape = array.shape + shape
        for _ in range(len(shape)):
            array = np.expand_dims(array, axis)
        if copy:
            return np.broadcast_to(array, new_shape).copy()
        else:
            return np.broadcast_to(array, new_shape)
    return None


def make_slice(n, axis=0, ind=0):
    idx = n * [slice(None)]
    if type(axis) is int:
        axis = [axis]
    if type(ind) is int or ind is None or ind is np.newaxis:
        ind = [ind]
    for i, k in zip(ind, axis):
        idx[k] = i
    return tuple(idx)


def slice_array(array, axis=0, ind=0):
    """Slice array along one or more axes.
    
    array : ndarray
        The array to slice.
    axis : int or list
        The axis(axes) along which to slice.
    ind : int or list
        Locations on the specified axis(axes).
    """
    return array[make_slice(array.ndim, axis, ind)]


def project(array, axis=0):
    """Project array onto one or more axes.
    
    array : ndarray
        The distribution.
    axis : int or list
        Axis(axes) along which to project the array.
    """
    if type(axis) is int:
        axis = [axis]
    axis_sum = tuple([i for i in range(array.ndim) if i not in axis])
    proj = np.sum(array, axis=axis_sum)
    # Handle out of order projection. Right now it just handles 2D, but
    # it should be extended to any number of dimension.
    if proj.ndim == 2 and axis[0] > axis[1]:
        proj = np.moveaxis(proj, 0, 1)
    return proj


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