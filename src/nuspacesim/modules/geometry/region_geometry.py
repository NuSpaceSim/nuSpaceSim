from nuspacesim.modules.geometry import nssgeometry as nssgeo
from nuspacesim.core import NssConfig
import numpy as np

__all__ = ["RegionGeom"]


class RegionGeom(nssgeo.Geom_params):
    """
    Region Geometry class.

    Wrapper of nssgeometry module for easier integration in nuspacesim.
    """

    def __init__(self, config: NssConfig):
        super().__init__(
            radE=config.constants.earth_radius,
            detalt=config.detector.altitude,
            detra=config.detector.ra_start,
            detdec=config.detector.dec_start,
            delAlpha=config.simulation.ang_from_limb,
            maxsepangle=config.simulation.theta_ch_max,
            delAziAng=config.simulation.max_azimuth_angle,
            ParamPi=config.constants.pi,
        )
        self.config = config

    def throw(self, numtrajs):
        """Generate Events."""
        uranarray = np.random.rand(numtrajs, 4)
        super().run_geo_dmc_from_ran_array_nparray(uranarray)

    def betas(self):
        """Create array of Earth-emergence angles for valid events."""
        betaArr = super().evArray["betaTrSubN"][super().evMasknpArray]

        return betaArr

    def beta_rad(self):
        """Create array of Earth-emergence angles for valid events."""
        return np.radians(self.betas())

    def __call__(self, numtrajs, store=None):
        """Throw numtrajs events and return valid betas in radians."""
        self.throw(numtrajs)

        if store is not None:
            store(["beta_tr"], [self.beta_rad()])

        return self.beta_rad()

    def mcintegral(self, numPEs, costhetaCh, tauexitprob, store=None):
        """Monte Carlo integral."""
        cossepangle = super().evArray["costhetaTrSubV"][super().evMasknpArray]

        # Geometry Factors
        mcintfactor = np.where(cossepangle - costhetaCh < 0, 0.0, 1.0)
        mcintfactor = np.multiply(
            mcintfactor, super().evArray["costhetaTrSubN"][super().evMasknpArray]
        )
        mcintfactor = np.divide(
            mcintfactor, super().evArray["costhetaNSubV"][super().evMasknpArray]
        )
        mcintfactor = np.divide(
            mcintfactor, super().evArray["costhetaTrSubV"][super().evMasknpArray]
        )
        mcintegralgeoonly = np.mean(mcintfactor) * super().mcnorm

        # Multiply by tau exit probability
        mcintfactor *= tauexitprob

        # PE threshold
        mcintfactor *= np.where(
            numPEs - self.config.detector.photo_electron_threshold < 0, 0.0, 1.0
        )

        numEvPass = np.count_nonzero(mcintfactor)

        mcintegral = np.mean(mcintfactor) * super().mcnorm

        if store is not None:
            store.add_meta("mcint", mcintegral, "MonteCarlo Integral")
            store.add_meta(
                "mcintgeo", mcintegralgeoonly, "MonteCarlo Integral, GEO Only"
            )
            store.add_meta("nEvPass", numEvPass, "Number of Passing Events")

        return mcintegral, mcintegralgeoonly, numEvPass
