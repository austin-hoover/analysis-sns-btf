import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings


class EnergyCalculate():
    
    def __init__(self, **kwargs):
        """
        l1 = 0.248  # slit1 to QH05 center
        l2 = 0.210  # Qh05 center to QV06 center
        l3 = 0.489  # QV06 center to slit2
        L2 = 0.567  # slit2 to dipole face
        l = 0.129  # dipole face to VS06 screen
        amp2meter=1.007*1e-3 -- relation of x_vs06 to delta-I
        C = 0.737 is saturation correction to dipole field (this is a measured quantity)
        rho_sign: float
            Sign of bending radius. (Positive bends toward negative x, which is true 
            at VS06; negative bends toward positive x, which is true at VS34.)
        """
        self.E0 = kwargs.get('E0', 2.5)
        self.m0 = kwargs.get('m0', 939.3014)
        self.freq = kwargs.get('freq', 402.5e6)
        self.rho_sign = kwargs.get('rho_sign', +1.0)
        
        # -- derived quantities
        self.speed_of_light = 2.99792458e+8
        self.gamma = self.E0 / self.m0 + 1.0
        self.beta = np.sqrt(1.0 - 1.0 / (self.gamma**2))
        self.P0 = self.gamma * self.m0 * self.beta
        self.brho = self.gamma * self.beta * self.m0 * 1e6 / self.speed_of_light
        
        # -- calibrated values
        self.C = kwargs.get('C', 0.737)
        self.amp2meter = kwargs.get('amp2meter', 1.007 * 1e3)
        
        # -- geometry terms for matrix calculations
        # For slits, I use location of HZ04, HZ06. VT04,VT06 shifted by ~0.04 m.
        self.l1 = kwargs.get('l1', 0.280)
        self.l2 = kwargs.get('l2', 0.210)
        self.l3 = kwargs.get('l3', 0.457)
        self.L2 = kwargs.get('L2', 0.599)
        self.l = kwargs.get('l', 0.129)
        self.L= self.l1 + self.l2 + self.l3 + self.L2
                
        # Bend radius. 
        self.rho = 0.3556 * self.rho_sign
        
        # mm to deg conversion
        self.z2phase = 360.0 * self.freq / (self.beta * self.speed_of_light) * 1e-3 
 
    def getthinquad(self, kappa, Lq):
        quadM = np.matrix([[1, 0, 0, 0, 0, 0],
                           [-kappa * Lq, 1, 0, 0, 0, 0],
                           [0,0,1,0,0,0],
                           [0, 0, kappa * Lq, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        return quadM
    
    def getdrift(self,l):
        drM = np.matrix([[1, l, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, l, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        return drM
    
    def getdipole(self, rho, theta):
        M = np.matrix([
            [0, rho, 0, 0, 0, rho],
            [-1 / rho, 0, 0, 0, 0, 1],
            [0, 0, 1, rho * theta, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, rho, 0 , 0, 1, rho * (theta - 1)],
            [0, 0, 0, 0, 0, 1]
        ])
        return M

    def getM1(self, GL05=0, GL06=0):
        """Get slit-slit matrix
        
        GL05 is integrated field of QH05 (positive=F)
        GL06 is integrated field of QV06 (positive=D)
        """
        # quads
        Lq = 0.106 # quad length 
        kappa5 = +GL05 / (Lq * self.brho)
        kappa6 = -GL06 / (Lq * self.brho)

        quad5 = self.getthinquad(kappa5, Lq)
        quad6 = self.getthinquad(kappa6, Lq)

        # drifts
        drl1 = self.getdrift(self.l1)
        drl2 = self.getdrift(self.l2)
        drl3 = self.getdrift(self.l3)

        # M1: slit to slit
        M1 = drl3 * quad6 * drl2 * quad5 * drl1
        M1rev = drl1 * quad5 * drl2 * quad6 * drl3
        return M1
    
    def getM2(self, rho=None):
        """Get 2nd slit-screen matrix."""
        if not rho:
            rho = self.rho
            theta = np.pi / 2.0
        else: 
            theta = np.arccos(1.0 - self.rho / rho)
            
        # M2: 2nd slit to dipole face
        M2 = self.getdrift(self.L2)

        # dipole 
        Mrho = self.getdipole(rho, theta)

        # Ml is just a short drift
        Ml = self.getdrift(self.l)

        M = Ml * Mrho * M2
        return M
    
    def getM(self, GL05=0.0, GL06=0.0, rho=None):
        """
        Get matrix slit-to-screen (VT04 to VS06)
        GL05 = 0 is integrated field of QH05 (positive=F)
        GL06 = 0. is integrated field of QV06 (positive=D)    
        rho = 0.3556 # dipole radius of curvature
        """
        M1 = self.getM1(GL05=GL05, GL06=GL06)
        M2 = self.getM2(rho=rho)
        M = M2 * M1
        return M
    
    def calculate_xp(self, x1, x2, M):
        """
        x1 is position of first slit (in meters)
        x2 is position of 2nd slit (meters)
        """
        xp = (x2 - M[0, 0] * x1) / M[0, 1]
        return xp
    
    def calculate_yp(self, y1, y2, M):
        """
        x1 is position of first slit (in meters)
        x2 is position of 2nd slit (meters)
        """
        yp = (y2 - M[2, 2] * y1) / M[2, 3]
        return yp

    def calculate_dE_slit(self, current, x, xp, GL05=0, GL06=0, rho=None, amp2meter=1.007*1e-3):
        """
        returns delta-energy dE in MeV
        current [A]; dipole current, actually I-I0 (so, it's delta-I)
        x [m]; x, x_slit-<x_slit>
        xp [rad]; xp, use calculate_xp above to get from x_slit1, x_slit2
        
        geometry:
        amp2meter=1.007*1e-3 -- relation of x_vs06 to delta-I
        GL05 = 0 is integrated field of QH05 (positive=F)
        GL06 = 0. is integrated field of QV06 (positive=D)
        rho = 0.3556 # dipole radius of curvature
        """
        if not(rho):
            rho = self.rho
        x3 = current * self.amp2meter
        M = self.getM(GL05=GL05, GL06=GL06, rho=rho)
        dpp = (1.0 / M[0, 5]) * (x3 - M[0, 0] * x - M[0, 1] * x)
        P = self.P0 * (dpp + 1.0)
        b = np.sqrt(1.0 / (1.0 + (self.m0 * self.m0) / (P * P)))
        gamma = P / (self.m0 * b)
        dE = (gamma - self.gamma) * self.m0
        return dE
    
    def rho_adjust(self, current, I0=359.0):
        rho = self.rho * (1.0 - self.C * current / I0)
        return rho 
    
    def calculate_dE_screen(self, x3, current, x, xp, M):
        """
        returns delta-energy dE in MeV
        x3 [m]; position on VS06 screen with respect to beam center, x_screen - <x_screen>. 
        current [A]; dipole current, actually I-I0 (so, it's delta-I)
        x [m]; x_slit - <x_slit>
        xp [rad]; use `calculate_xp` above to get from x_slit1, x_slit2
        """
        dpp = (1.0 / M[0, 5]) * (x3 - M[0, 0] * x - M[0, 1] * xp)
        P = self.P0 * (dpp + 1.0)
        b = np.sqrt(1.0 / (1.0 + (self.m0 * self.m0) / (P * P)))
        gamma = P / (self.m0 * b)
        dE = (gamma - self.gamma) * self.m0
        return dE
    
    # -- phase conversion.    
    def calculate_dphi(self, de, **kwargs):
        """ 
        L is distance between emittance plane at dipole entrance (L=1.545 for HZ04 location)
        l is distance dipole exit to bsm
        DPHI_L is phase shift due to path length effect when projecting back to earlier plane.
        DPHI_L= 0 when considering BSM plane.
        """
        L = kwargs.get('L',self.L2)
        l = kwargs.get('l',0.695)
        print('%.3f %.3f %.3f m'%(L,self.rho,l))
        
        prefactor = 2*np.pi*self.freq / (self.beta*self.gamma*self.gamma*self.speed_of_light) 
        path_length = L + l + self.rho*np.pi/2
        x_correction = x0 + xp0*(L + self.rho)

        DEE = de / self.E0
        DPP = 1/self.beta**2 *(1-1/self.gamma)*DEE 

        dphi_l = np.rad2deg(prefactor* DPP * (path_length + x_correction + self.rho*(np.pi/2-1)*DPP)) 
        
        return dphi_l 
    
    def averageRows(self,df):
        """
        average rows without applying rejection algorithm
        
        input:
        thisdf: dataframe of n rows; will be averaged along rows
        
        output:
        1D array: averaged data 
        """    
        dropind = np.array([])
        return df.mean(axis=0), dropind