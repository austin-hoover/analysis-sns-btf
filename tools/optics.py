"""BTF optics calculations."""
import numpy as np
from scipy.constants import speed_of_light


def matrix_dipole(rho, theta):
    return np.matrix([
        [0, rho, 0, 0, 0, rho],
        [-1 / rho, 0, 0, 0, 0, 1],
        [0, 0, 1, rho * theta, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, rho, 0, 0, 1, rho * (theta - 1)],
        [0, 0, 0, 0, 0, 1]
    ])


def matrix_drift(l):
    return np.matrix([
        [1, l, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, l, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])


def matrix_thin_quad(kappa, Lq):
    return np.matrix([
        [1, 0, 0, 0, 0, 0],
        [-kappa * Lq, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, kappa * Lq, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])


def get_xp(x1, x2, M):
    """Return x' [rad].
    
    x1 : float
        The x coordinate at point 1 [m].
    x2 : float
        The x coordinate at point 2 [m].
    M : ndarray, shape (6, 6)
        The transfer matrix from point 1 to point 2.
    """
    return (x2 - M[0,0] * x1) / M[0,1]


def get_yp(y1, y2, M):
    """Return y' [rad].
    
    y1 : float
        The y coordinate at point 1 [m].
    y2 : float
        The y coordinate at point 2 [m].
    M : ndarray, shape (6, 6)
        The transfer matrix from point 1 to point 2.
    """
    return (y2 - M[2, 2] * y1) / M[2, 3]


class BTFOptics():
    """BTF optics calculations."""
    def __init__(
        self, 
        E0=2.5,
        m0=939.3014,
        freq=402.5e6,
        C=0.737,
        amp2meter=1.007e3,
        rho=0.3556,
        theta=0.5*np.pi,
        GL05=0.0,
        GL06=0.0,
        l1=0.248,
        l2=0.210,
        l3=0.489,
        L2=0.567,
        l=0.129,
    ):
        """Constructor.
        
        Parameters
        ----------
        E0 : float
            Kinetic energy of synchronous particle [MeV].
        m0 : float
            Mass per particle [MeV/c^2].
        freq : float
            RF frequency [Hz].
        C : float
            Measured saturation correction to dipole field.
        amp2meter : float
            Relation of x_screen to delta_I (difference from nominal dipole current).
        rho : float
            Dipole bend radius [m].
        theta : float
            Dipole bend angle [radians]
        GL05 : float
            Integrated field strength of QH05 (positive=F).
        GL06 : float
            Integrated field strength of QV06 (positive=D). 
        l1 : float
            Distance from first slit to QH05 center.
        l2 : float
            Distance from QH05 center to QV06 center.
        l3 : float
            Distance from QV06 center to second slit.
        L2 : float
            Distance from second slit to dipole face.
        l : float
            Distance from dipole face to screen.
        """
        self.E0 = E0
        self.m0 = m0
        self.freq = freq
        self.rho = rho
        self.theta = theta
        self.gamma = (self.E0 / self.m0) + 1.0  # Lorentz factor
        self.beta = np.sqrt(1.0 - (1.0 / (self.gamma**2)))  # Lorentz factor
        self.P0 = self.gamma * self.m0 * self.beta
        self.brho = self.gamma * self.beta * self.m0 * 1.0e6 / speed_of_light
        self.z2phase = 360.0 * self.freq / (self.beta * speed_of_light) * 1.0e-3 

    def get_M1(self, Lq=0.106):
        """Return slit-slit transfer matrix.
        
        Lq : float
            Quadrupole length [m].
        """
        kappa5 = self.GL05 / (Lq * self.brho)
        kappa6 = self.GL06 / (Lq * self.brho)
        quad5 = matrix_thin_quad(kappa5, Lq)
        quad6 = matrix_thin_quad(kappa6, Lq)
        drl1 = matrix_drift(self.l1)
        drl2 = matrix_drift(self.l2)
        drl3 = matrix_drift(self.l3)
        return drl3 * quad6 * drl2 * quad5 * drl1
    
    def get_M2(self):
        """Return transfer matrix from second slit to screen."""
        M2 = matrix_drift(self.L2)
        Mrho = matrix_dipole(self.rho, self.thetha)
        Ml = matrix_drift(self.l)
        return Ml * Mrho * M2
    
    def get_M(self):
        """Return transfer matrix from first slit to screen."""
        M1 = self.get_M1()
        M2 = matrix_drift(self.L2)
        Mrho = matrix_dipole(self.rho, self.theta)
        Ml = matrix_drift(self.l)
        return Ml * Mrho * M2 * M1
    
    def get_dE_screen(self, x3, current, x, xp):
        """Return energy deviation dE from the screen position.
        
        Parameters
        ----------
        x3 : float
            Position on screen with respect to beam center [m].
        current : float
            Deviation from nominal dipole current (I - I0) [A].
        x : float
            The x [m] coordinate at the measurement plane.
        xp : float
            The xp [rad] coordinate at the measurement plane.
            
        Returns
        -------
        dE : float
            Deviation from the synchronous particle energy [MeV].
        """
        M = self.get_M()
        dpp = (1.0 / M[0, 5]) * (x3 - M[0, 0] * x - M[0, 1] * xp)
        P = self.P0 * (dpp + 1.0)
        b = np.sqrt(1.0 / (1.0 + (self.m0**2) / (P**2)))
        gamma = P / (self.m0 * b)
        return (gamma - self.gamma) * self.m0
    
    def get_dphi(self, de, L=None, l=0.695):
        """Phase conversion.
        
        Parameters
        ----------
        de : float
            [...]
        L : float
            Distance from emittance plane to dipole entrance (L=1.545 for HZ04 location).
        l : float
            Distance from dipole exit to BSM.
            
        Returns
        -------
        dphi : float
            Phase shift due to path length effect when projecting back to earlier 
            plane. (0 when considering BSM plane.)
        """
        if L is None:
            L = self.L2        
        prefactor = 2.0 * np.pi * self.freq / (self.beta * self.gamma * self.gamma * speed_of_light) 
        path_length = L + l + self.rho * 0.5 * np.pi
        x_correction = x0 + xp0 * (L + self.rho)
        DEE = de / self.E0
        DPP = 1.0 / self.beta**2 * (1.0 - 1.0 / self.gamma) * DEE 
        return np.rad2deg(prefactor * DPP * (path_length + x_correction + self.rho*(0.5 * np.pi - 1.0) * DPP))