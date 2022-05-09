import numpy as np


def project(Z, indices):
    """Project distribution onto lower-dimensional subspace.
    
    Parameters
    ----------
    Z : ndarray, n dimensions
        The distribution.
    indices : int, list, or tuple
        The distribution is projected onto these indices.
        
    Returns
    -------
    ndarray, n - len(indices) dimensions
        The projection of the distribution.
    """
    if type(indices) is int:
        indices = [indices]
    axis = tuple([k for k in range(Z.ndim) if k not in indices])
    return np.sum(Z, axis=axis)