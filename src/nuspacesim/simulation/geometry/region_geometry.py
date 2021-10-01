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
from ... import NssConfig

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
