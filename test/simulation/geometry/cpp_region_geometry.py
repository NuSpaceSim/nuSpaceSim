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

from nuspacesim.simulation.geometry.nssgeometry import Geom_params
from nuspacesim.utils import decorators
from nuspacesim import NssConfig
from nuspacesim.simulation.geometry.local_plots import *

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
    @decorators.nss_result_store("beta_rad", "theta_rad", "path_len")
    def __call__(self, numtrajs):
        """Throw numtrajs events and return valid betas."""
        self.throw(numtrajs)
        return self.beta_rad(), self.thetas(), self.pathLens()

    def mcintegral(self, triggers, costheta, tauexitprob, threshold):
        """Monte Carlo integral."""

        cossepangle = super().evArray["costhetaTrSubV"][super().evMasknpArray]

        # Geometry Factors
        mcintfactor = np.where(cossepangle - costheta < 0, 0.0, 1.0)
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

        # # PE threshold
        mcintfactor *= np.where(triggers - threshold < 0, 0.0, 1.0)

        # mcintfactor = np.where(
        #     mcintfactor_opt > mcintfactor_rad, mcintfactor_opt, mcintfactor_rad
        # )

        numEvPass = np.count_nonzero(mcintfactor)

        mcintegral = np.mean(mcintfactor) * super().mcnorm

        return mcintegral, mcintegralgeoonly, numEvPass