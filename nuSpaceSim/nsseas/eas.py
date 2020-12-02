import numpy as np
from .cphotang import CphotAng


class EAS:
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

        altDec = np.sqrt(
            self.config.EarthRadius ** 2 +
            lenDec ** 2 +
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

        # costhetaCh = np.cos(thetaCh * self.config.fundcon.pi / 180.0)
        costhetaCh = np.cos(np.degrees(thetaCh))

        numPEs = dphots * showerEnergy * self.config.detAeff * \
            self.config.detQeff

        return numPEs, costhetaCh
