import nuSpaceSim.nssgeometry as nssgeo
import numpy as np


class RegionGeom(nssgeo.Geom_params):
    """
    Region Geometry class.

    Wrapper of nssgeometry module for easier integration in nuSpaceSim.
    """

    def __init__(self, config):
        super().__init__(radE=config.EarthRadius,
                         detalt=config.detectAlt,
                         detra=config.raStart,
                         detdec=config.decStart,
                         delAlpha=config.AngFrLimb,
                         maxsepangle=config.thetaChMax,
                         delAziAng=config.maxaziang,
                         ParamPi=config.fundcon.pi)
        self.config = config
        self.detPEthres = config.detPEthres

    def throw(self, numtrajs):
        """ Generate Events.  """
        uranarray = np.random.rand(numtrajs, 4)
        super().run_geo_dmc_from_ran_array_nparray(uranarray)

    def betas(self):
        """ Create array of Earth-emergence angles for valid events.  """
        betaArr = super().evArray["betaTrSubN"][super().evMasknpArray]

        return betaArr

    def __call__(self, numtrajs):
        """ Throw numtrajs events and return valid betas.  """
        self.throw(numtrajs)
        return self.betas()

    def mcintegral(self, numPEs, costhetaCh, tauexitprob):
        """ Monte Carlo integral.  """
        #cossepangle = super().localevent.costhetaTrSubV
        cossepangle = super().evArray["costhetaTrSubV"][super().evMasknpArray]

        #print(cossepangle,costhetaCh)

        # Geometry Factors
        mcintfactor = np.where(cossepangle - costhetaCh < 0, 0.0, 1.0)
        mcintfactor = np.multiply(
            mcintfactor,
            super().evArray["costhetaTrSubN"][super().evMasknpArray])
        mcintfactor = np.divide(
            mcintfactor,
            super().evArray["costhetaNSubV"][super().evMasknpArray])
        mcintfactor = np.divide(
            mcintfactor,
            super().evArray["costhetaTrSubV"][super().evMasknpArray])

        mcintegralgeoonly = np.mean(mcintfactor) * super().mcnorm

        # Multiply by tau exit probability
        mcintfactor *= tauexitprob

        # PE threshold
        mcintfactor *= np.where(numPEs - self.detPEthres < 0, 0.0, 1.0)

        numEvPass = np.count_nonzero(mcintfactor)

        mcintegral = np.mean(mcintfactor) * super().mcnorm

        return mcintegral, mcintegralgeoonly, numEvPass
