# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np

from .nssgeometry import Geom_params
from ...utils import decorators
from ... import NssConfig
from .local_plots import *

__all__ = ["RegionGeom"]


class RegionGeom(Geom_params):
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

    def thetas(self):
        """Create array of view angles for valid events."""
        thetaArr = super().evArray["thetaTrSubV"][super().evMasknpArray]
        return thetaArr

    def pathLens(self):
        """Create array of view angles for valid events."""
        pathLenArr = super().evArray["losPathLen"][super().evMasknpArray]
        return pathLenArr

    @decorators.nss_result_plot(geom_beta_tr_hist, geom_beta_tr_hist_red)
    @decorators.nss_result_store("beta_rad", "theta_rad", "path_lens")
    def __call__(self, numtrajs):
        """Throw numtrajs events and return valid betas."""
        self.throw(numtrajs)
        return self.beta_rad(), self.thetas(), self.pathLens()

    @decorators.nss_result_store_scalar(
        ["mcint", "mcintgeo", "nEvPass"],
        [
            "MonteCarlo Integral",
            "MonteCarlo Integral, GEO Only",
            "Number of Passing Events",
        ],
    )
    def mcintegral(self, numPEs, costhetaCh, tauexitprob):
        """Monte Carlo integral.
        numPEs is actually SNR in the radio case
        """
        if (
            self.config.detector.method == "Radio"
            or self.config.detector.method == "Optical"
        ):
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

            mcint_notrigger = mcintfactor.copy()
            # PE threshold
            if self.config.detector.method == "Radio":
                mcintfactor *= np.where(
                    numPEs - self.config.detector.det_SNR_thres < 0, 0.0, 1.0
                )
            if self.config.detector.method == "Optical":
                mcintfactor *= np.where(
                    numPEs - self.config.detector.photo_electron_threshold < 0, 0.0, 1.0
                )
        if self.config.detector.method == "Both":
            cossepangle = super().evArray["costhetaTrSubV"][super().evMasknpArray]

            npe = numPEs[0]
            snr = numPEs[1]
            opt_costheta = costhetaCh[0]
            rad_costheta = costhetaCh[1]
            # Geometry Factors
            # Optical first
            mcintfactor = np.ones(opt_costheta.shape)
            mcintfactor = np.multiply(
                mcintfactor, super().evArray["costhetaTrSubN"][super().evMasknpArray]
            )
            mcintfactor = np.divide(
                mcintfactor, super().evArray["costhetaNSubV"][super().evMasknpArray]
            )
            mcintfactor = np.divide(
                mcintfactor, super().evArray["costhetaTrSubV"][super().evMasknpArray]
            )

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
            mcintfactor_opt *= np.where(
                npe - self.config.detector.photo_electron_threshold < 0, 0.0, 1.0
            )
            mcintfactor_rad *= np.where(
                snr - self.config.detector.det_SNR_thres < 0, 0.0, 1.0
            )
            mcintfactor = np.where(
                mcintfactor_opt > mcintfactor_rad, mcintfactor_opt, mcintfactor_rad
            )

        numEvPass = np.count_nonzero(mcintfactor)

        mcintegral = np.mean(mcintfactor) * super().mcnorm

        return mcintegral, mcintegralgeoonly, numEvPass
