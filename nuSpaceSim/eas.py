import numpy as np
from nuSpaceSim.EAScherGen.cphotang import CphotAng


class EAS:
    """
    Electromagnetic Air Shower wrapper class.

    Vectorized computation of photo-electrons and Cherenkov angles.
    """

    def __init__(self, config):
        self.config = config
        self.CphotAng = CphotAng()

    def altDec(self, beta, tauBeta, tauLorentz, u=None):
        """
        altDec
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

        return altDec

    def __call__(self, beta, tauBeta, tauLorentz, showerEnergy):
        """
        Electromagnetic Air Shower operation.
        """

        altDec = self.altDec(beta, tauBeta, tauLorentz)

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

        #print (enhanceFactor, logenhanceFactor)
        
        hwfm = np.sqrt(2. * logenhanceFactor)
        thetaChEnh = np.multiply(thetaCh, hwfm)
        thetaChEff = np.where(thetaChEnh >= thetaCh, thetaChEnh, thetaCh)

        #print(thetaCh, thetaChEff)

        #costhetaCh = np.cos(np.degrees(thetaCh))
        #costhetaCh = np.cos(np.radians(thetaCh))
        costhetaChEff = np.cos(np.radians(thetaChEff))

        return numPEs, costhetaChEff
