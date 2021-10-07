import numpy as np
from .nssgeometry import Geom_params
from ... import NssConfig

__all__ = ["RegionGeom"]

class RegionGeom(Geom_params):
    """
    Region Geometry class.

    Wrapper of nssgeometry module for easier integration in nuSpaceSim.
    """

    def __init__(self, config):
        super().__init__(radE=config.constants.earth_radius,
                         detalt=config.detector.altitude,
                         detra=config.detector.ra_start,
                         detdec=config.detector.dec_start,
                         delAlpha=config.simulation.ang_from_limb,
                         maxsepangle=config.simulation.theta_ch_max,
                         delAziAng=config.simulation.max_azimuth_angle,
                         ParamPi=config.constants.pi)
        self.config = config
        self.detPEthres = config.detector.photo_electron_threshold
        self.detSNRthres = config.detector.det_SNR_thres

    def throw(self, numtrajs):
        """ Generate Events.  """
        uranarray = np.random.rand(numtrajs, 4)
        super().run_geo_dmc_from_ran_array_nparray(uranarray)

    def betas(self):
        """ Create array of Earth-emergence angles for valid events."""
        betaArr = super().evArray["betaTrSubN"][super().evMasknpArray]
        return betaArr

    def beta_rad(self):
        """ Create array of Earth-emergence angles for valid events."""
        return np.radians(self.betas())
    
    def thetas(self):
        """ Create array of view angles for valid events."""
        thetaArr = super().evArray["thetaTrSubV"][super().evMasknpArray]
        return thetaArr
    
    def pathLens(self):
        """ Create array of view angles for valid events."""
        pathLenArr = super().evArray["losPathLen"][super().evMasknpArray]
        return pathLenArr
    
    def __call__(self, numtrajs, store=None):
        """ Throw numtrajs events and return valid betas."""
        self.throw(numtrajs)
        betas = self.beta_rad()
        thetas = self.thetas()
        pathLens = self.pathLens()
        if store is not None:
            store(["beta_tr"], [self.beta_rad()])
        return betas, thetas, pathLens

    def mcintegral(self, numPEs, costhetaCh, tauexitprob, store=None):
        """ Monte Carlo integral.  
            numPEs is actually SNR in the radio case
        """
        if self.config.detector.method == 'Radio' or self.config.detector.method == 'Optical':
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

            mcint_notrigger = mcintfactor.copy()
            # PE threshold
            if self.config.detector.method == 'Radio':
                mcintfactor *= np.where(numPEs - self.detSNRthres < 0, 0.0, 1.0)
            if self.config.detector.method == 'Optical':
                mcintfactor *= np.where(numPEs - self.detPEthres < 0, 0.0, 1.0)
        if self.config.detector.method == 'Both':
            cossepangle = super().evArray["costhetaTrSubV"][super().evMasknpArray]

            npe = numPEs[0]
            snr = numPEs[1]
            opt_costheta = costhetaCh[0]
            rad_costheta = costhetaCh[1]
            # Geometry Factors
            #Optical first
            mcintfactor = np.ones(opt_costheta.shape)
            mcintfactor = np.multiply(
                mcintfactor,
                super().evArray["costhetaTrSubN"][super().evMasknpArray])
            mcintfactor = np.divide(
                mcintfactor,
                super().evArray["costhetaNSubV"][super().evMasknpArray])
            mcintfactor = np.divide(
                mcintfactor,
                super().evArray["costhetaTrSubV"][super().evMasknpArray])
            
            mcintfactor_opt = np.where(cossepangle - opt_costheta < 0, 0.0, 1.0)
            mcintfactor_rad = np.where(cossepangle - rad_costheta < 0, 0.0, 1.0)
            mcintfactor_opt *= mcintfactor
            mcintfactor_rad *= mcintfactor

            mcintegralgeoonly = np.mean(mcintfactor_rad) * super().mcnorm

            # Multiply by tau exit probability
            mcintfactor_opt *= tauexitprob
            mcintfactor_rad *= tauexitprob

            mcint_notrigger = mcintfactor_rad.copy()
            # PE threshold
            mcintfactor_opt *= np.where(npe - self.detPEthres < 0, 0.0, 1.0)
            mcintfactor_rad *= np.where(snr - self.detSNRthres < 0, 0.0, 1.0)
            mcintfactor = np.where(mcintfactor_opt > mcintfactor_rad, mcintfactor_opt, mcintfactor_rad)

        numEvPass = np.count_nonzero(mcintfactor)

        mcintegral = np.mean(mcintfactor) * super().mcnorm

        if store is not None:
            store.add_meta("mcint", mcintegral, "MonteCarlo Integral")
            store.add_meta(
                "mcintgeo", mcintegralgeoonly, "MonteCarlo Integral, GEO Only"
            )
            store.add_meta("nEvPass", numEvPass, "Number of Passing Events")

        #return mcintegral, mcintegralgeoonly, numEvPass, mcint_notrigger, super().mcnorm
        return mcintegral, mcintegralgeoonly, numEvPass
