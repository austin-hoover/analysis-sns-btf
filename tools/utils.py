import numpy as np


def max_indices(array):
    """Return indices of maximum element in array."""
    return np.unravel_index(np.argmax(array), array.shape)


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