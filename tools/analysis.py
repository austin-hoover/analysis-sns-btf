import numpy as np
import numpy.linalg as la
from tqdm import trange

from . import utils


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
    if Sigma.shape[0] < 4:
        raise ValueError("Cannot calculate intrinsic emittances of 2x2 covariance matrix.")
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
    eps_x = emittance2D(Sigma[:2, :2])
    eps_y = emittance2D(Sigma[2:, 2:])
    return eps_x, eps_y


def emittance2D(Sigma):
    return np.sqrt(la.det(Sigma))


def twiss2D(Sigma):
    eps = emittance2D(Sigma)
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
    alpha_x, beta_x = twiss2D(Sigma[:2, :2])
    alpha_y, beta_y = twiss2D(Sigma[2:, 2:])
    return alpha_x, alpha_y, beta_x, beta_y


def beam_stats(cov):
    """Return dictionary of parameters from NxN covariance matrix.
    
    Parameters
    ----------
    cov : ndarray, shape (N, N)
        The NxN covariance matrix.
    
    Returns
    -------
    dict
        cov : ndarray, shape (N, N)
            The covariance matrix.
        corr : ndarray, shape (N, N)
            The correlation matrix.
        'eps_x': apparent emittance in x-x' plane
        'eps_y': apparent emittance in y-y' plane
        'eps_1': intrinsic emittance
        'eps_2': intrinsic emittance
        'eps_4D': 4D emittance = eps_1 * eps_2
        'eps_4D_app': apparent 4D emittance = eps_x * eps_y
        'C': coupling coefficient = sqrt((eps_1 * eps_2) / (eps_x * eps_y))
        'beta_x': beta_x = <x^2> / eps_x
        'beta_y': beta_y = <y^2> / eps_y
        'alpha_x': alpha_x = -<xx'> / eps_x
        'alpha_y': alpha_y = -<yy'> / eps_y
    """
    stats = dict()
    stats["cov"] = cov
    stats["corr"] = utils.cov2corr(cov)
    if cov.shape[0] < 4:
        stats["alpha"], stats["beta"] = twiss2D(cov)
        stats["eps"] = emittance2D(cov)
    else:
        stats["alpha_x"], stats["alpha_y"], stats["beta_x"], stats["beta_y"] = twiss(cov)
        stats["eps_x"], stats["eps_y"], stats["eps_1"], stats["eps_2"] = emittances(cov)
        stats["eps_4D"] = stats["eps_1"] * stats["eps_2"]
        stats["eps_4D_app"] = stats["eps_x"] * stats["eps_y"]
        stats["C"] = np.sqrt(stats["eps_4D"] / stats["eps_4D_app"])
    return stats