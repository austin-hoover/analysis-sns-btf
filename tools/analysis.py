import numpy as np
import numpy.linalg as la


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