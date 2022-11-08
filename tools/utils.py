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


def flatten_index(ind, shape):
    """Return index in flattened array from multi-dimensional index.
    
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


def copy_into_new_dim(a, shape, axis=-1, method='broadcast', copy=False):
    """Copy an array into one or more new dimensions.
    
    The 'broadcast' method is much faster since it works with views instead of copies. 
    See 'https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times'
    """
    if type(shape) in [int, np.int32, np.int64]:
        shape = (shape,)
    if method == 'repeat':
        for i in range(len(shape)):
            a = np.repeat(np.expand_dims(a, axis), shape[i], axis=axis)
        return a
    elif method == 'broadcast':
        if axis == 0:
            new_shape = shape + a.shape
        elif axis == -1:
            new_shape = a.shape + shape
        else:
            raise ValueError('Cannot yet handle axis != 0, -1.')
        for _ in range(len(shape)):
            a = np.expand_dims(a, axis)
        if copy:
            return np.broadcast_to(a, new_shape).copy()
        else:
            return np.broadcast_to(a, new_shape)
    return None


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


def permutations_with_replacement(elements, n):
    """Return unique permutations of elements.
    
    https://stackoverflow.com/questions/6284396/permutations-with-unique-values
    """
    def permutations_helper(elements, result_list, d):
        if d < 0 :
            yield tuple(result_list)
        else:
            for element in elements:
                result_list[d] = element
                for g in permutations_helper(elements, result_list, d - 1):
                    yield g
                    
    return permutations_helper(elements, [0] * n, n - 1)


def multiset_permutations(elements):
    from sympy.utilities.iterables import multiset_permutations
    return multiset_permutations(elements)