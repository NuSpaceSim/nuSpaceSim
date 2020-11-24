import nssgeometry as nssgeo
import numpy as np
import numpy.ma as ma


class DetectorGeometry(nssgeo.Geom_params):
    def __init__(self, config):
        # self detector = nssgeo.Geom_params(
        super().__init__(
            radE=config.EarthRadius,
            detalt=config.detectAlt,
            detra=config.raStart,
            detdec=config.decStart,
            delAlpha=config.AngFrLimb,
            maxsepangle=config.thetaChMax,
            delAziAng=config.maxaziang,
            ParamPi=config.fundcon.pi)

    def throw(self, numtrajs):
        uranarray = np.random.rand(numtrajs, 4)
        super().run_geo_dmc_from_ran_array_nparray(uranarray)

    def betas(self):
        # Create array of Earth-emergence angles for valid events and tile it
        betaArrMA = ma.array(
            super().evArray["betaTrSubN"],
            mask=np.logical_not(
                super().evMasknpArray))

        betaArr = np.array(betaArrMA[~betaArrMA.mask])

        return betaArr
