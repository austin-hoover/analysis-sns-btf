import numpy as np
import numpy.linalg as la
from tqdm import trange

from . import utils


def dist_cov(f, coords, disp=False):
    """Compute the distribution covariance matrix.
    
    Parameters
    ----------
    f : ndarray
        The distribution function (weights).
    coords : list[ndarray]
        List of coordinates along each axis of `H`. Can also
        provide meshgrid coordinates.
        
    Returns
    -------
    Sigma : ndarray, shape (n, n)
    means : ndarray, shape (n,)
    """
    if disp:
        print(f'Forming {f.shape} meshgrid...')
    if coords[0].ndim == 1:
        COORDS = np.meshgrid(*coords, indexing='ij')
    n = f.ndim
    f_sum = np.sum(f)
    if f_sum == 0:
        return np.zeros((n, n)), np.zeros((n,))
    if disp:
        print('Averaging...')
    means = np.array([np.average(C, weights=f) for C in COORDS])
    Sigma = np.zeros((n, n))
    _range = trange if disp else range
    for i in _range(Sigma.shape[0]):
        for j in _range(i + 1):
            X = COORDS[i] - means[i]
            Y = COORDS[j] - means[j]
            EX = np.sum(X * f) / f_sum
            EY = np.sum(Y * f) / f_sum
            EXY = np.sum(X * Y * f) / f_sum
            Sigma[i, j] = EXY - EX * EY
    Sigma = utils.symmetrize(Sigma)
    if disp:
        print('Done.')
    return Sigma, means


def rms_ellipse_dims(sig_xx, sig_yy, sig_xy):
    """Return semi-axes and tilt angle of the RMS ellipse in the x-y plane.
    
    Parameters
    ----------
    sig_xx, sig_yy, sig_xy : float
        Covariance between x-x, y-y, x-y.
    
    Returns
    -------
    angle : float
        Tilt angle of the ellipse below the x axis (radians).
    cx, cy : float
        Semi-axes of the ellipse.
    """
    angle = -0.5 * np.arctan2(2 * sig_xy, sig_xx - sig_yy)
    sn, cs = np.sin(angle), np.cos(angle)
    cx = np.sqrt(abs(sig_xx*cs**2 + sig_yy*sn**2 - 2*sig_xy*sn*cs))
    cy = np.sqrt(abs(sig_xx*sn**2 + sig_yy*cs**2 + 2*sig_xy*sn*cs))
    return angle, cx, cy


def intrinsic_emittances(Sigma):
    """Return intrinsic emittances from covariance matrix."""
    Sigma = Sigma[:4, :4]
    U = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    trSU2 = np.trace(la.matrix_power(np.matmul(Sigma, U), 2))
    detS = la.det(Sigma)
    eps_1 = 0.5 * np.sqrt(-trSU2 + np.sqrt(trSU2**2 - 16 * detS))
    eps_2 = 0.5 * np.sqrt(-trSU2 - np.sqrt(trSU2**2 - 16 * detS))
    return eps_1, eps_2
    
    
def apparent_emittances(Sigma):
    """Return apparent emittances from covariance matrix."""
    Sigma = Sigma[:4, :4]
    eps_x = _emittance(Sigma[:2, :2])
    eps_y = _emittance(Sigma[2:, 2:])
    return eps_x, eps_y


def _emittance(Sigma):
    return np.sqrt(la.det(Sigma))


def _twiss(Sigma):
    eps = _emittance(Sigma)
    alpha = -Sigma[0, 1] / eps
    beta = Sigma[0, 0] / eps
    return (alpha, beta)


def emittances(Sigma):
    """Return rms emittances from covariance matrix."""
    Sigma = Sigma[:4, :4]
    eps_x, eps_y = apparent_emittances(Sigma)
    eps_1, eps_2 = intrinsic_emittances(Sigma)
    return eps_x, eps_y, eps_1, eps_2
        
def twiss(Sigma):
    """Return 2D Twiss parameters from covariance matrix."""
    Sigma = Sigma[:4, :4]
    eps_x, eps_y = apparent_emittances(Sigma)
    beta_x = Sigma[0, 0] / eps_x
    beta_y = Sigma[2, 2] / eps_y
    alpha_x = -Sigma[0, 1] / eps_x
    alpha_y = -Sigma[2, 3] / eps_y
    return alpha_x, alpha_y, beta_x, beta_y