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
        self.detSNRthres = config.detSNRthres

    def throw(self, numtrajs):
        """ Generate Events.  """
        uranarray = np.random.rand(numtrajs, 4)
        super().run_geo_dmc_from_ran_array_nparray(uranarray)

    def betas(self):
        """ Create array of Earth-emergence angles for valid events.  """
        betaArr = super().evArray["betaTrSubN"][super().evMasknpArray]

        return betaArr
    
    def thetas(self):
        """ Create array of view angles for valid events.  """
        thetaArr = super().evArray["thetaTrSubV"][super().evMasknpArray]
        return thetaArr
    
    def pathLens(self):
        """ Create array of view angles for valid events.  """
        pathLenArr = super().evArray["losPathLen"][super().evMasknpArray]
        return pathLenArr
    
    def __call__(self, numtrajs):
        """ Throw numtrajs events and return valid betas.  """
        self.throw(numtrajs)
        betas = self.betas()
        thetas = self.thetas()
        pathLens = self.pathLens()
        return betas, thetas, pathLens

    def mcintegral(self, numPEs, costhetaCh, tauexitprob):
        """ Monte Carlo integral.  
            numPEs is actually SNR in the radio case
        """
        #cossepangle = super().localevent.costhetaTrSubV
        cossepangle = super().evArray["costhetaTrSubV"][super().evMasknpArray]

        #print(cossepangle)
        #print(costhetaCh)

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
        if self.config.method == 'Radio':
            mcintfactor *= np.where(numPEs - self.detSNRthres < 0, 0.0, 1.0)
        else:
            mcintfactor *= np.where(numPEs - self.detPEthres < 0, 0.0, 1.0)

        numEvPass = np.count_nonzero(mcintfactor)

        mcintegral = np.mean(mcintfactor) * super().mcnorm

        return mcintegral, mcintegralgeoonly, numEvPass
