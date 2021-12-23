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

__all__ = ["RegionGeom"]


class RegionGeom:
    """Region Geometry of trajectories."""

    def __init__(self, config):
        self.config = config

        self.earth_rad_2 = self.config.constants.earth_radius ** 2

        self.core_alt = (
            self.config.constants.earth_radius + self.config.detector.altitude
        )

        self.detRA = np.radians(config.detector.ra_start)
        self.detDec = np.radians(config.detector.dec_start)

        alphaHorizon = np.pi / 2 - np.arccos(
            self.config.constants.earth_radius / self.core_alt
        )
        alphaMin = alphaHorizon - config.simulation.ang_from_limb
        minChordLen = 2 * np.sqrt(
            self.earth_rad_2 - (self.core_alt * np.sin(alphaMin)) ** 2
        )

        # Line of Sight path length
        self.minLOSpathLen = self.core_alt * np.cos(alphaMin) - minChordLen / 2
        self.maxLOSpathLen = np.sqrt(self.core_alt ** 2 - self.earth_rad_2)

        self.sinOfMaxThetaTrSubV = np.sin(config.simulation.theta_ch_max)

        self.maxPhiS = config.simulation.max_azimuth_angle / 2
        self.minPhiS = -config.simulation.max_azimuth_angle / 2

        normThetaTrSubV = 2 / self.sinOfMaxThetaTrSubV ** 2
        normPhiTrSubV = np.reciprocal(2 * np.pi)
        normPhiS = np.reciprocal(self.maxPhiS - self.minPhiS)

        bracketForNormThetaS = (
            (self.core_alt ** 2 - self.earth_rad_2) * self.maxLOSpathLen
            - (1.0 / 3.0) * self.maxLOSpathLen ** 3
            - (self.core_alt ** 2 - self.earth_rad_2) * self.minLOSpathLen
            + (1.0 / 3.0) * self.minLOSpathLen ** 3
        )

        normThetaS = 2.0 * self.core_alt * self.earth_rad_2 / bracketForNormThetaS

        pdfnorm = normThetaTrSubV * normPhiTrSubV * normPhiS * normThetaS
        self.mcnorm = self.earth_rad_2 / pdfnorm

    def throw(self, u=None):
        """Throw N events with 4 * u random numbers"""

        if isinstance(u, int):
            # fix to make closed in [0,1]
            u = np.random.rand(4, u)

        if u is None:
            raise RuntimeError(
                "Provide a number of trajectories, or a 2D set of uniform random "
                "numbers in [0, 1]"
            )

        if u.shape[0] != 4:
            raise RuntimeError("u random numbers must be of shape (4, N)")

        u1, u2, u3, u4 = u

        self.thetaTrSubV = np.arcsin(self.sinOfMaxThetaTrSubV * np.sqrt(u1))
        self.costhetaTrSubV = np.cos(self.thetaTrSubV)
        self.phiTrSubV = 2.0 * np.pi * u2
        self.phiS = (self.maxPhiS - self.minPhiS) * u3 + self.minPhiS

        # Generate theta_s (the colatitude on the surface of the Earth in the
        # detector nadir perspective)

        b = (
            3 * (self.core_alt ** 2 - self.earth_rad_2) * self.maxLOSpathLen
            - self.maxLOSpathLen ** 3
            - 3 * (self.core_alt ** 2 - self.earth_rad_2) * self.minLOSpathLen
            + self.minLOSpathLen ** 3
        )
        q = -(self.core_alt ** 2 - self.earth_rad_2)
        r = (
            -1.5 * (self.core_alt ** 2 - self.earth_rad_2) * self.maxLOSpathLen
            + 0.5 * self.maxLOSpathLen ** 3
            + 0.5 * b * u4
        )

        psi = np.arccos(r / np.sqrt(-(q ** 3)))
        v1 = 2 * np.sqrt(-q) * np.cos(psi / 3)
        v2 = 2 * np.sqrt(-q) * np.cos((psi + 2 * np.pi) / 3)
        v3 = 2 * np.sqrt(-q) * np.cos((psi + 4 * np.pi) / 3)

        dscr = q * q * q + r * r

        dmsk = dscr <= 0
        v1_msk = (v1 > 0) & (v1 >= self.minLOSpathLen) & (v1 <= self.maxLOSpathLen)
        v2_msk = (v2 > 0) & (v2 >= self.minLOSpathLen) & (v2 <= self.maxLOSpathLen)
        v3_msk = (v3 > 0) & (v3 >= self.minLOSpathLen) & (v3 <= self.maxLOSpathLen)

        self.losPathLen = np.zeros_like(v1)
        self.losPathLen[dmsk & v1_msk] = v1[dmsk & v1_msk]
        self.losPathLen[dmsk & v2_msk] = v2[dmsk & v2_msk]
        self.losPathLen[dmsk & v3_msk] = v3[dmsk & v3_msk]

        s = np.cbrt(r[~dmsk] + np.sqrt(dscr[~dmsk]))
        t = np.cbrt(r[~dmsk] - np.sqrt(dscr[~dmsk]))
        self.losPathLen[~dmsk] = s + t

        # self.losPathLen[~dmsk] = np.sum(
        #     np.cbrt(r[~dmsk] + np.multiply.outer([1, -1], np.sqrt(dscr[~dmsk]))),
        #     axis=0,
        # )

        rvsqrd = self.losPathLen * self.losPathLen
        costhetaS = (self.core_alt ** 2 + self.earth_rad_2 - rvsqrd) / (
            2 * self.config.constants.earth_radius * self.core_alt
        )
        self.thetaS = np.arccos(costhetaS)

        self.costhetaNSubV = (self.core_alt ** 2 - self.earth_rad_2 - rvsqrd) / (
            2 * self.config.constants.earth_radius * self.losPathLen
        )

        thetaNSubV = np.arccos(self.costhetaNSubV)

        self.costhetaTrSubN = np.cos(self.thetaTrSubV) * self.costhetaNSubV - np.sin(
            self.thetaTrSubV
        ) * np.sin(thetaNSubV) * np.cos(self.phiTrSubV)

        self.thetaTrSubN = np.arccos(self.costhetaTrSubN)

        self.betaTrSubN = np.degrees(0.5 * np.pi - self.thetaTrSubN)

        rsindecS = np.sin(self.config.detector.dec_start) * costhetaS - np.cos(
            self.config.detector.dec_start
        ) * np.sin(self.thetaS) * np.cos(self.phiS)

        self.decS = np.degrees(np.arcsin(rsindecS))

        rxS = (
            np.sin(self.config.detector.dec_start)
            * np.cos(self.config.detector.ra_start)
            * np.sin(self.thetaS)
            * np.cos(self.phiS)
            - np.sin(self.config.detector.ra_start)
            * np.sin(self.thetaS)
            * np.sin(self.phiS)
            + np.cos(self.config.detector.dec_start)
            * np.cos(self.config.detector.ra_start)
            * np.cos(self.thetaS)
        )

        ryS = (
            np.sin(self.config.detector.dec_start)
            * np.sin(self.config.detector.ra_start)
            * np.sin(self.thetaS)
            * np.cos(self.phiS)
            + np.cos(self.config.detector.ra_start)
            * np.sin(self.thetaS)
            * np.sin(self.phiS)
            + np.cos(self.config.detector.dec_start)
            * np.sin(self.config.detector.ra_start)
            * np.cos(self.thetaS)
        )

        self.raS = np.degrees(np.arctan2(ryS, rxS)) % 360.0

        self.event_mask = np.logical_and(self.costhetaTrSubN >= 0, self.betaTrSubN < 42)

    def betas(self):
        """Earth-emergence angles for valid events."""
        return self.betaTrSubN[self.event_mask]

    def beta_rad(self):
        """Radian Earth-emergence angles for valid events."""
        return np.radians(self.betas())

    def thetas(self):
        """View angles for valid events."""
        return self.thetaTrSubV[self.event_mask]

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

        numEvPass = np.count_nonzero(mcintfactor)

        return mcintegral, mcintegralgeoonly, numEvPass


def show_plot(sim, plot):
    plotfs = tuple([geom_beta_tr_hist])
    inputs = tuple([0])
    outputs = ("beta_rad", "theta_rad", "path_len")
    decorators.nss_result_plot_from_file(sim, inputs, outputs, plotfs, plot)
