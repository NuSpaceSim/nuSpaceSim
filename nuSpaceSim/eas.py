import numpy as np
from nuSpaceSim.EAScherGen.cphotang import CphotAng
import nuSpaceSim.radio as radio

#import matplotlib.pyplot as plt

class EAS:
    """
    Electromagnetic Air Shower wrapper class.

    Vectorized computation of photo-electrons and Cherenkov angles.
    """

    def __init__(self, config):
        self.config = config
        self.CphotAng = CphotAng(config)

    def altDec(self, beta, tauBeta, tauLorentz, u=None):
        """
        get decay altitude
        """

        u = np.random.uniform(0, 1, len(beta)) if u is None else u

        tDec = (-1.0 * tauLorentz /
                self.config.fundcon.inv_mean_Tau_life) * np.log(u)

        lenDec = tDec * tauBeta * self.config.fundcon.c

        # brad = beta * (self.config.fundcon.pi / 180.0)
        brad = np.radians(beta)

        altDec = np.sqrt(self.config.EarthRadius**2 + lenDec**2 +
                         2.0 * self.config.EarthRadius * lenDec * np.sin(brad))

        altDec -= self.config.EarthRadius

        return altDec, lenDec

            
    def call_optical(self, beta, theta, tauBeta, tauLorentz, showerEnergy):
        """
        Electromagnetic Air Shower operation.
        """

        altDec, lenDec = self.altDec(beta, tauBeta, tauLorentz)

        mask = (altDec < 0.0) | (altDec > 20.0)
        mask |= beta < 0.0
        mask |= beta > 25.0
        mask = ~mask

        dphots = np.zeros_like(beta)
        thetaCh = np.full(beta.shape, 1.5)

        dphots[mask], thetaCh[mask] = self.CphotAng(beta[mask], altDec[mask])

        numPEs = dphots * showerEnergy * self.config.detAeff * \
            self.config.detQeff

        enhanceFactor = numPEs / self.config.detPEthres
        # logenhanceFactor = np.where(enhanceFactor > 2.0, np.log(enhanceFactor), 0.5)
        logenhanceFactor = np.empty_like(enhanceFactor)
        efMask = enhanceFactor > 2.0
        logenhanceFactor[efMask] = np.log(enhanceFactor[efMask])
        logenhanceFactor[~efMask] = 0.5

        #print(enhanceFactor)
        #print(logenhanceFactor)

        hwfm = np.sqrt(2. * logenhanceFactor)
        thetaChEnh = np.multiply(thetaCh, hwfm)
        thetaChEff = np.where(thetaChEnh >= thetaCh, thetaChEnh, thetaCh)

        #print(thetaCh)
        #print(thetaChEff)

        #costhetaCh = np.cos(np.degrees(thetaCh))
        #costhetaCh = np.cos(np.radians(thetaCh))
        costhetaChEff = np.cos(np.radians(thetaChEff))

        return numPEs, costhetaChEff

    def decay_to_detector_dist(self, beta, altDec, detectAlt):
        Re = self.config.EarthRadius
        r1 = detectAlt + Re
        r2 = altDec + Re
        exit = np.pi/2. - beta
        r2squared = r2**2
        cosexit = np.cos(exit)
        return np.sqrt(r2squared*cosexit*cosexit - r2squared + r1*r1) - r2*cosexit

    def get_decay_view(self, exitView, losDist, lenDec):
        '''
        get view angle from shower to detector
        '''
        #sin^2 of our decay view angle
        s2phi = (losDist*np.sin(exitView))**2.0 / (
                lenDec * lenDec + losDist * losDist - 2 * losDist * lenDec * np.cos(exitView)
                )
        ang = np.arcsin(np.sqrt(s2phi))
        return ang

    def call_radio(self, beta, theta, pathLen, tauBeta, tauLorentz, showerEnergy):
        '''
        EAS radio output from ZHAires lookup tables
        '''
        altDec, lenDec = self.altDec(beta, tauBeta, tauLorentz)
        radioParams = radio.RadioEFieldParams(self.config.detFreqRange)
        mask = (altDec < 0.0) | (altDec > 10.0) #TODO set a reasonable cut for max shower height
        mask = ~mask
        
        #rudimentary distance scaling TODO investigate that this actually works with zhaires
        nssDist = self.decay_to_detector_dist(np.deg2rad(beta[mask]), altDec[mask], self.config.detectAlt)
        zhairesDist = self.decay_to_detector_dist(np.deg2rad(beta[mask]), altDec[mask], 525.0)

        viewAngles = np.zeros_like(theta)
        viewAngles[mask] = self.get_decay_view(theta[mask], pathLen[mask], lenDec[mask])

        #plt.hist(np.degrees(viewAngles[mask]), label='decay view', bins=100)
        #plt.hist(np.degrees(theta[mask]), label='decay view', bins=100)
        #plt.xlabel('angle (deg)')
        #plt.legend()
        #plt.yscale('log')
        #plt.show()

        EFields = np.zeros_like(beta)
        EFields[mask] = radioParams(90. - beta[mask], np.rad2deg(viewAngles[mask]), altDec[mask])

        #scale by the energy of the shower (all zhaires files are for 10^18 eV shower)
        #shower energy is in units of 100 PeV, we want in GeV
        EFields *= showerEnergy/10.
        distScale = zhairesDist/nssDist
        EFields[mask] *= distScale

        return EFields, altDec
    
    def __call__(self, beta, theta, pathLen, tauBeta, tauLorentz, showerEnergy):
        if self.config.method == "Radio":
            return self.call_radio(beta, theta, pathLen, tauBeta, tauLorentz, showerEnergy)
        if self.config.method == "Optical":
            return self.call_optical(beta, theta, tauBeta, tauLorentz, showerEnergy)

