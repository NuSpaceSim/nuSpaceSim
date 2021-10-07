import numpy as np
from .cphotang import CphotAng
from ... import NssConfig
from ..eas_radio.radio import RadioEFieldParams, IonosphereParams

__all__ = ["EAS"]

class EAS:
    """
    Electromagnetic Air Shower wrapper class.

    Vectorized computation of photo-electrons and Cherenkov angles.
    """

    def __init__(self, config: NssConfig):
        self.config = config
        self.CphotAng = CphotAng()

    def altDec(self, beta, tauBeta, tauLorentz, u=None):
        """
        get decay altitude
        """

        u = np.random.uniform(0, 1, len(beta)) if u is None else u

        tDec = (-1.0 * tauLorentz /
                self.config.constants.inv_mean_Tau_life) * np.log(u)

        lenDec = tDec * tauBeta * self.config.constants.c

        altDec = np.sqrt(self.config.constants.earth_radius**2 + lenDec**2 +
                         2.0 * self.config.constants.earth_radius * lenDec * np.sin(beta))

        altDec -= self.config.constants.earth_radius

        return altDec, lenDec

            
    def call_optical(self, beta, altDec, lenDec, theta, showerEnergy, store):
        """
        Electromagnetic Air Shower operation.
        """

        mask = (altDec < 0.0) | (altDec > 20.0)
        mask |= beta < 0.0
        mask |= beta > np.radians(25.0)
        mask = ~mask

        dphots = np.zeros_like(beta)
        thetaCh = np.full(beta.shape, 1.5)

        dphots[mask], thetaCh[mask] = self.CphotAng(beta[mask], altDec[mask])

        numPEs = dphots * showerEnergy * self.config.detector.telescope_effective_area * \
            self.config.detector.quantum_efficiency

        enhanceFactor = numPEs / self.config.detector.photo_electron_threshold
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
        if store is not None:
            store(
                ["altDec", "numPEs", "costhetaChEff"], [altDec, numPEs, costhetaChEff]
            )

        return numPEs, costhetaChEff

    def decay_to_detector_dist(self, beta, altDec, detectAlt, lenDec, viewAngle):
        Re = self.config.constants.earth_radius
        r1 = detectAlt + Re
        r2 = altDec + Re
        exit = np.pi/2. - beta
        r2squared = r2**2
        thetaE = (Re**2 + (Re + altDec)**2 - lenDec**2)/(2*Re*(Re+altDec))
        thetaE[thetaE<0] = 0
        thetaE[thetaE>1] = 1
        thetaE = np.arccos(thetaE)
        thetaRel = exit - thetaE + viewAngle
        cosexit = np.cos(thetaRel)
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

    def call_radio(self, beta, altDec, lenDec, theta, pathLen, showerEnergy, store):
        '''
        EAS radio output from ZHAires lookup tables
        '''
        FreqRange = (self.config.detector.low_freq, self.config.detector.high_freq)
        radioParams = RadioEFieldParams(FreqRange)
        mask = (altDec < 0.0) | (altDec > 10.0) #TODO set a reasonable cut for max shower height
        mask = ~mask
        
        viewAngles = np.zeros_like(theta)
        viewAngles[mask] = self.get_decay_view(theta[mask], pathLen[mask], lenDec[mask])
        
        #rudimentary distance scaling TODO investigate that this actually works with zhaires
        nssDist = self.decay_to_detector_dist(beta[mask], altDec[mask], self.config.detector.altitude, lenDec[mask], viewAngles[mask])
        zhairesDist = self.decay_to_detector_dist(beta[mask], altDec[mask], 525.0, lenDec[mask], viewAngles[mask])

        #plt.hist(np.degrees(viewAngles[mask]), label='decay view', bins=100)
        #plt.hist(np.degrees(theta[mask]), label='decay view', bins=100)
        #plt.xlabel('angle (deg)')
        #plt.legend()
        #plt.yscale('log')
        #plt.show()

        #EFields = np.zeros_like(beta).astype(float)
        #EFields[mask] = radioParams(90. - beta[mask], np.rad2deg(viewAngles[mask]), altDec[mask])
        EFields = np.zeros_like(beta)
        EFields = radioParams(np.degrees(np.pi/2. - beta), np.rad2deg(viewAngles), altDec)
        EFields = (EFields.T * mask).T

        #scale by the energy of the shower (all zhaires files are for 10^18 eV shower)
        #shower energy is in units of 100 PeV, we want in GeV
        EFields[mask] = (EFields[mask].T * showerEnergy[mask]/10.).T
        distScale = zhairesDist/nssDist
        EFields[mask] = (EFields[mask].T * distScale).T
        if self.config.simulation.model_ionosphere:
            if self.config.simulation.TEC < 0:
                print("TEC should be positive!! continuing without ionospheric dispersion")
            else:
                ionosphere = IonosphereParams(FreqRange, self.config.simulation.TECerr, self.config.simulation.TEC)
                ionosphereScaling = ionosphere(EFields[mask])
                EFields[mask] *= ionosphereScaling
        if store is not None:
            store(
                ["altDec", "EFields"], [altDec, EFields]
            )

        return EFields, altDec
    
    def __call__(self, beta, altDec, lenDec, theta, pathLen, showerEnergy, store=None):
        if self.config.detector.method == "Radio":
            return self.call_radio(beta, altDec, lenDec, theta, pathLen, showerEnergy, store)
        if self.config.detector.method == "Optical":
            return self.call_optical(beta, altDec, lenDec, theta, showerEnergy, store)
        if self.config.detector.method == "Both":
            NPE, costhetaChEff=self.call_optical(beta, altDec, lenDec, theta, showerEnergy, store)
            EFields, altDec=self.call_radio(beta, altDec, lenDec, theta, pathLen, showerEnergy, store)
            return [NPE, EFields], costhetaChEff


