import numpy as np
from nuspacesim.EAScherGen.cphotang import CphotAng
from nuspacesim.configuration import NssConfig

__all__ = ["EAS"]


class EAS:
    """
    Electromagnetic Air Shower wrapper class.

    Vectorized computation of photo-electrons and Cherenkov angles.
    """

    def __init__(self, config: NssConfig):
        self.config = config
        self.CphotAng = CphotAng()

    def altDec(self, beta, tauBeta, tauLorentz, u=None) -> float:
        """
        alt Decay
        """

        u = np.random.uniform(0, 1, len(beta)) if u is None else u

        tDec = (-1.0 * tauLorentz / self.config.constants.inv_mean_Tau_life) * np.log(u)

        lenDec = tDec * tauBeta * self.config.constants.c

        # brad = beta * (self.config.fundcon.pi / 180.0)
        brad = np.radians(beta)

        altDec = np.sqrt(
            self.config.constants.earth_radius ** 2
            + lenDec ** 2
            + 2.0 * self.config.constants.earth_radius * lenDec * np.sin(brad)
        )

        altDec -= self.config.constants.earth_radius

        return altDec

    def __call__(self, beta, altDec, showerEnergy, store=None) -> tuple:
        """
        Electromagnetic Air Shower operation.
        """

        # Mask out-of-bounds events. Do not pass to CphotAng. Instead use
        # Default values for dphots and thetaCh
        mask = (altDec < 0.0) | (altDec > 20.0)
        mask |= beta < 0.0
        mask |= beta > 25.0
        mask = ~mask

        # phots and theta arrays with default 0 and 1.5 values.
        dphots = np.full(beta.shape, np.nan)
        thetaCh = np.full(beta.shape, np.nan)

        # Run CphotAng on in-bounds events
        dphots[mask], thetaCh[mask] = self.CphotAng(beta[mask], altDec[mask])

        numPEs = (
            dphots
            * showerEnergy
            * self.config.detector.telescope_effective_area
            * self.config.detector.quantum_efficiency
        )

        enhanceFactor = numPEs / self.config.detector.photo_electron_threshold
        logenhanceFactor = np.empty_like(enhanceFactor)
        efMask = enhanceFactor > 2.0
        logenhanceFactor[efMask] = np.log(enhanceFactor[efMask])
        logenhanceFactor[~efMask] = 0.5

        hwfm = np.sqrt(2.0 * logenhanceFactor)
        thetaChEnh = np.multiply(thetaCh, hwfm)
        thetaChEff = np.where(thetaChEnh >= thetaCh, thetaChEnh, thetaCh)

        costhetaChEff = np.cos(np.radians(thetaChEff))

        if store is not None:
            store(
                ["altDec", "numPEs", "costhetaChEff"], [altDec, numPEs, costhetaChEff]
            )

        return numPEs, costhetaChEff
