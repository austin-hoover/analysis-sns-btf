import numpy as np
import pickle


def volume_unit_sphere(n=2):
    from scipy.special import gamma
    return np.pi**(0.5 * n) / gamma(1.0 + 0.5 * n)

def volume_sphere(n=2, r=1.0):
    return volume_unit_sphere(n=n) * r**n

def volume_unit_box(n=2):
    return 2.0**n

def volume_box(n=2, r=1.0):
    return volume_unit_box(n=n) * r**n


def save_pickle(filename, item):
    """Convenience function to save pickled file."""
    with open(filename, 'wb') as file:
        pickle.dump(item, file)
        
        
def load_pickle(filename):
    """Convenience function to load pickled file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def cov2corr(cov_mat):
    """Form correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = np.linalg.inv(D)
    corr_mat = np.linalg.multi_dot([Dinv, cov_mat, Dinv])
    return corr_mat


def symmetrize(M):
    """Return a symmetrized version of M.
    
    M : A square upper or lower triangular matrix.
    """
    return M + M.T - np.diag(M.diagonal())


def is_sorted(a):
    return np.all(a[:-1] <= a[1:])


def avoid_repeats(a, pad=1e-12):
    """Avoid repeating points in an array.
    
    Adds a small number to each duplicate element.
    """
    ind, = np.where(np.diff(a) == 0)
    counter = 1
    for i in ind:
        a[i] += counter * pad
        # a[i] += np.random.uniform(0.0, counter * pad)
        counter += 1
    return a


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
    if type(shape) in [int, np.int32, np.int64]:
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
        else:
            raise ValueError('Cannot yet handle axis != 0, -1.')
        for _ in range(len(shape)):
            array = np.expand_dims(array, axis)
        if copy:
            return np.broadcast_to(array, new_shape).copy()
        else:
            return np.broadcast_to(array, new_shape)
    return None


def make_slice(n, axis=0, ind=0):
    """Return a slice index."""
    if type(axis) is int:
        axis = [axis]
    if type(ind) is int:
        ind = [ind]
    idx = n * [slice(None)]
    for k, i in zip(axis, ind):
        if i is None:
            continue
        elif type(i) is tuple and len(i) == 2:
            idx[k] = slice(i[0], i[1])
        else:
            idx[k] = i
    return tuple(idx)


def project(array, axis=0):
    """Project array onto one or more axes."""
    if type(axis) is int:
        axis = [axis]
    axis_sum = tuple([i for i in range(array.ndim) if i not in axis])
    proj = np.sum(array, axis=axis_sum)
    # Handle out of order projection. Right now it just handles 2D, but
    # it should be extended to higher dimensions.
    if proj.ndim == 2 and axis[0] > axis[1]:
        proj = np.moveaxis(proj, 0, 1)
    return proj


def get_grid_coords(*xi, indexing='ij'):
    """Return array of shape (N, D), where N is the number of points on 
    the grid and D is the number of dimensions."""
    return np.vstack([X.ravel() for X in np.meshgrid(*xi, indexing=indexing)]).T


def snap(array, n=165, pad=0.1, tol=0.0):
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


# The following three functions are from Tony Yu's blog (https://tonysyu.github.io/ragged-arrays.html#.YKVwQy9h3OR). They allow fast saving/loading of ragged arrays.
def stack_ragged(array_list, axis=0):
    """Stacks list of arrays along first axis.
    
    Example: (25, 4) + (75, 4) -> (100, 4). It also returns the indices at
    which to split the stacked array to regain the original list of arrays.
    """
    lengths = [np.shape(array)[axis] for array in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx
    

def save_stacked_array(filename, array_list, axis=0):
    """Save list of ragged arrays as single stacked array. The index from
    `stack_ragged` is also saved."""
    stacked, idx = stack_ragged(array_list, axis=axis)
    np.savez(filename, stacked_array=stacked, stacked_index=idx)
    
    
def load_stacked_arrays(filename, axis=0):
    """"Load stacked ragged array from .npz file as list of arrays."""
    npz_file = np.load(filename)
    idx = npz_file['stacked_index']
    stacked = npz_file['stacked_array']
    return np.split(stacked, idx, axis=axis)


def get_boundary_points(iterations, points, signal, thresh, pad=3.0, tol=0.01):
    lb = []  # "left" boundary points
    ub = []  # "right" boundary points
    for iteration in np.unique(iterations):
        # Get points for this sweep.
        idx_sweep, = np.where(iterations == iteration)
        _points = points[idx_sweep]
        # Make sure only the first actuator is sweeping.
        if tol:
            if not np.all(np.abs(_points[:, 1:] - _points[0, 1:]) <= tol):
                print("More than one sweeper!")
                break
        _signal = signal[idx_sweep].copy()
        # Sort by sweeper coordinate.
        idx_sort = np.argsort(_points[:, 0])
        _points = _points[idx_sort, :]
        _signal = _signal[idx_sort]
        # Find min/max sweeper coordinate with signal > thresh.
        _valid, = np.where(_signal >= thresh)
        if len(_valid) == 0:
            continue
        elif len(_valid) == 1:
            xl = xr = _points[_valid[0], 0] 
        else:
            xl = _points[_valid[0], 0] 
            xr = _points[_valid[-1], 0]
        # Add some padding.
        delta = pad * np.mean(np.diff(_points[:, 0]))
        lb.append(np.hstack([xl - delta, _points[0, 1:]]))
        ub.append(np.hstack([xr + delta, _points[0, 1:]]))
    return np.array([lb, ub])