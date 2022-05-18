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

from ...utils import decorators
from .local_plots import geom_beta_tr_hist
from ...too import *

__all__ = ["RegionGeom"]


class RegionGeom:
    """Region Geometry of trajectories."""

    def __init__(self, config, source):
        self.config = config

        self.earth_rad_2 = self.config.constants.earth_radius**2

        self.core_alt = (
            self.config.constants.earth_radius + self.config.detector.altitude
        )

        self.source = source

        self.detRA = np.radians(config.detector.ra_start)
        self.detDec = np.radians(config.detector.dec_start)

        alphaHorizon = np.pi / 2 - np.arccos(
            self.config.constants.earth_radius / self.core_alt
        )  # why????
        # alphaMin = alphaHorizon - config.simulation.ang_from_limb #self ??
        alphaMin = np.radians(42)
        minChordLen = 2 * np.sqrt(
            self.earth_rad_2 - (self.core_alt * np.sin(alphaMin)) ** 2
        )
        self.Tobs = 24 * 60 * 60

        # Line of Sight path length
        self.minLOSpathLen = self.core_alt * np.cos(alphaMin) - minChordLen / 2
        self.maxLOSpathLen = np.sqrt(self.core_alt**2 - self.earth_rad_2)

        self.sinOfMaxThetaTrSubV = np.sin(config.simulation.theta_ch_max)  # self?

        self.maxPhiS = config.simulation.max_azimuth_angle / 2
        self.minPhiS = -config.simulation.max_azimuth_angle / 2

        normThetaTrSubV = 2 / self.sinOfMaxThetaTrSubV**2
        normPhiTrSubV = np.reciprocal(2 * np.pi)
        normPhiS = np.reciprocal(self.maxPhiS - self.minPhiS)

        bracketForNormThetaS = (
            (self.core_alt**2 - self.earth_rad_2) * self.maxLOSpathLen
            - (1.0 / 3.0) * self.maxLOSpathLen**3
            - (self.core_alt**2 - self.earth_rad_2) * self.minLOSpathLen
            + (1.0 / 3.0) * self.minLOSpathLen**3
        )

        normThetaS = 2.0 * self.core_alt * self.earth_rad_2 / bracketForNormThetaS

        pdfnorm = normThetaTrSubV * normPhiTrSubV * normPhiS * normThetaS
        self.mcnorm = self.earth_rad_2 / pdfnorm

    def throw(self, u=None):
        """Throw N events with 1 * u random numbers"""

        if isinstance(u, int):
            # fix to make closed in [0,1]
            u = np.random.rand(1, u)

        if u is None:
            raise RuntimeError(
                "Provide a number of trajectories, or a 2D set of uniform random "
                "numbers in [0, 1]"
            )

        if u.shape[0] != 1:
            raise RuntimeError("u random numbers must be of shape (1, N)")

        # Generate the random times
        u *= self.Tobs  # in s
        u = astropy.time.TimeDelta(u, format="sec")
        u = self.source.eventtime + u

        # Calculate the local nadir angle of the source
        self.sourceNadRad = np.pi / 2 - self.source.localcoords(u).alt.rad

        # Calculate the earth emergence angle from the nadir angle
        self.sourcebeta = np.arccos(
            ((self.core_alt) / self.config.constants.earth_radius)
            * np.sin(self.sourceNadRad)
        )

        # Calculate the pathlength through the atmosphere
        self.losPathLen = self.config.constants.earth_radius * np.cos(
            self.sourcebeta
        ) - self.core_alt * np.cos(self.sourceNadRad + self.sourcebeta)

        # Cut out any events that are outside the calc volume
        self.event_mask = np.logical_and(
            np.rad2deg(self.sourcebeta) >= 0, np.rad2deg(self.sourcebeta) < 42
        )

    def betas(self):
        """Earth-emergence angles for valid events."""
        return np.rad2deg(beta_rad())

    def beta_rad(self):
        """Radian Earth-emergence angles for valid events."""
        return self.sourcebeta[self.event_mask]

    def thetas(self):
        """View angles for valid events."""
        return self.sourceNadRad[self.event_mask]

    def pathLens(self):
        """View angles for valid events."""
        # pathLenArr = super().evArray["losPathLen"][super().evMasknpArray]
        # return pathLenArr
        return self.losPathLen[self.event_mask]

    def valid_costhetaTrSubN(self):
        return self.costhetaTrSubN[self.event_mask]

    def valid_costhetaNSubV(self):
        return self.costhetaNSubV[self.event_mask]

    def valid_costhetaTrSubV(self):
        return self.costhetaTrSubV[self.event_mask]

    @decorators.nss_result_plot(geom_beta_tr_hist)
    @decorators.nss_result_store("beta_rad", "theta_rad", "path_len")
    def __call__(self, numtrajs, *args, **kwargs):
        """Throw numtrajs events and return valid betas."""
        self.throw(numtrajs)
        return self.beta_rad(), self.thetas(), self.pathLens()

    def mcintegral(
        self,
        triggers,
        costheta,
        tauexitprob,
        threshold,
        spec_norm,
        spec_weights_sum,
    ):
        """Monte Carlo integral."""

        cossepangle = self.costhetaTrSubV[self.event_mask]

        mcintfactor = (
            self.valid_costhetaTrSubN()
            / self.valid_costhetaNSubV()
            / self.valid_costhetaTrSubV()
        )

        mcnorm = self.mcnorm

        # Geometry Factors
        mcintfactor[cossepangle < costheta] = 0
        mcintegralgeoonly = np.mean(mcintfactor) * mcnorm

        # Multiply by tau exit probability
        mcintfactor *= tauexitprob

        # Weighting by energy spectrum if other than monoenergetic spectrum
        mcintfactor /= spec_norm
        mcintfactor /= spec_weights_sum

        # PE threshold
        mcintfactor[triggers < threshold] = 0
        mcintegral = np.mean(mcintfactor) * mcnorm
        mcintegraluncert = (
            np.sqrt(np.var(mcintfactor, ddof=1) / len(mcintfactor)) * mcnorm
        )

        numEvPass = np.count_nonzero(mcintfactor)

        return mcintegral, mcintegralgeoonly, numEvPass, mcintegraluncert

    def tooMcIntegral(
        self,
        triggers,
        costhetaChEff,
        tauexitprob,
        threshold,
        spec_norm,
        spec_weights_sum,
    ):

        # calculate the Cherenkov angle
        thetaChEff = np.arccos(costhetaChEff)
        tanthetaChEff = np.tan(thetaChEff)

        # cossepangle = self.costhetaTrSubV[self.event_mask]

        mcintfactor = self.pathLens() * self.pathLens() * tanthetaChEff**2

        # Branching ratio set to 1 to be consistent
        Bshr = 1
        mcnorm = np.pi * Bshr

        # Geometry Factors
        # mcintfactor[cossepangle < costheta] = 0  # What does this do?
        mcintegralgeoonly = np.mean(mcintfactor) * mcnorm

        # Multiply by tau exit probability
        mcintfactor *= tauexitprob

        # Weighting by energy spectrum if other than monoenergetic spectrum
        mcintfactor /= spec_norm
        mcintfactor /= spec_weights_sum

        # PE threshold
        mcintfactor[triggers < threshold] = 0
        mcintegral = np.mean(mcintfactor) * mcnorm
        mcintegraluncert = (
            np.sqrt(np.var(mcintfactor, ddof=1) / len(mcintfactor)) * mcnorm
        )

        numEvPass = np.count_nonzero(mcintfactor)

        return mcintegral, mcintegralgeoonly, numEvPass, mcintegraluncert


def show_plot(sim, plot):
    plotfs = tuple([geom_beta_tr_hist])
    inputs = tuple([0])
    outputs = ("beta_rad", "theta_rad", "path_len")
    decorators.nss_result_plot_from_file(sim, inputs, outputs, plotfs, plot)
