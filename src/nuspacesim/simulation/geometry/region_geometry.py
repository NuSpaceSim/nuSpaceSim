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

import astropy
import matplotlib.pyplot as plt
import numpy as np

from ...utils import decorators
from .local_plots import geom_beta_tr_hist
from .too import ToOEvent

__all__ = ["RegionGeom", "RegionGeomToO"]


class RegionGeom:
    """Region Geometry of trajectories."""

    def __init__(self, config):
        self.config = config

        self.earth_rad_2 = self.config.constants.earth_radius**2

        self.core_alt = (
            self.config.constants.earth_radius + self.config.detector.altitude
        )
        self.detection_mode = self.config.simulation.det_mode
        self.sun_moon_cut = self.config.detector.sun_moon_cuts

        # Detector definitions
        self.detlat = np.radians(config.detector.detlat)
        self.detlong = np.radians(config.detector.detlong)
        self.detalt = config.detector.altitude

        self.alphaHorizon = np.pi / 2 - np.arccos(
            self.config.constants.earth_radius / self.core_alt
        )

        alphaMin = self.alphaHorizon - config.simulation.ang_from_limb

        minChordLen = 2 * np.sqrt(
            self.earth_rad_2 - (self.core_alt * np.sin(alphaMin)) ** 2
        )

        # Line of Sight path length
        self.minLOSpathLen = self.core_alt * np.cos(alphaMin) - minChordLen / 2
        self.maxLOSpathLen = np.sqrt(self.core_alt**2 - self.earth_rad_2)

        self.sinOfMaxThetaTrSubV = np.sin(config.simulation.theta_ch_max)

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
            3 * (self.core_alt**2 - self.earth_rad_2) * self.maxLOSpathLen
            - self.maxLOSpathLen**3
            - 3 * (self.core_alt**2 - self.earth_rad_2) * self.minLOSpathLen
            + self.minLOSpathLen**3
        )
        q = -(self.core_alt**2 - self.earth_rad_2)
        r = (
            -1.5 * (self.core_alt**2 - self.earth_rad_2) * self.maxLOSpathLen
            + 0.5 * self.maxLOSpathLen**3
            + 0.5 * b * u4
        )

        psi = np.arccos(r / np.sqrt(-(q**3)))
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
        costhetaS = (self.core_alt**2 + self.earth_rad_2 - rvsqrd) / (
            2 * self.config.constants.earth_radius * self.core_alt
        )
        self.thetaS = np.arccos(costhetaS)

        self.costhetaNSubV = (self.core_alt**2 - self.earth_rad_2 - rvsqrd) / (
            2 * self.config.constants.earth_radius * self.losPathLen
        )

        thetaNSubV = np.arccos(self.costhetaNSubV)

        self.costhetaTrSubN = np.cos(self.thetaTrSubV) * self.costhetaNSubV - np.sin(
            self.thetaTrSubV
        ) * np.sin(thetaNSubV) * np.cos(self.phiTrSubV)

        self.thetaTrSubN = np.arccos(self.costhetaTrSubN)

        self.betaTrSubN = np.degrees(0.5 * np.pi - self.thetaTrSubN)

        rsindecS = np.sin(self.config.detector.detlong) * costhetaS - np.cos(
            self.config.detector.detlong
        ) * np.sin(self.thetaS) * np.cos(self.phiS)

        self.decS = np.degrees(np.arcsin(rsindecS))

        rxS = (
            np.sin(self.config.detector.detlong)
            * np.cos(self.config.detector.detlat)
            * np.sin(self.thetaS)
            * np.cos(self.phiS)
            - np.sin(self.config.detector.detlat)
            * np.sin(self.thetaS)
            * np.sin(self.phiS)
            + np.cos(self.config.detector.detlong)
            * np.cos(self.config.detector.detlat)
            * np.cos(self.thetaS)
        )

        ryS = (
            np.sin(self.config.detector.detlong)
            * np.sin(self.config.detector.detlat)
            * np.sin(self.thetaS)
            * np.cos(self.phiS)
            + np.cos(self.config.detector.detlat)
            * np.sin(self.thetaS)
            * np.sin(self.phiS)
            + np.cos(self.config.detector.detlong)
            * np.sin(self.config.detector.detlat)
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
        if self.detection_mode == "ToO":
            self.too_throw(numtrajs)
            return self.too_beta_rad(), self.too_thetas(), self.too_pathLens()
        else:
            self.throw(numtrajs)
            return self.beta_rad(), self.thetas(), self.pathLens()

    def McIntegral(
        self,
        triggers,
        costheta,
        tauexitprob,
        threshold,
        spec_norm,
        spec_weights_sum,
        **kwargs,
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


class RegionGeomToO:
    def __init__(self, config):
        self.config = config

        self.core_alt = (
            self.config.constants.earth_radius + self.config.detector.altitude
        )
        self.detection_mode = self.config.simulation.det_mode
        self.sun_moon_cut = self.config.detector.sun_moon_cuts

        self.sourceOBSTime = self.config.simulation.source_obst
        self.too_source = ToOEvent(self.config)

        self.alphaHorizon = np.pi / 2 - np.arccos(
            self.config.constants.earth_radius / self.core_alt
        )

    @decorators.nss_result_plot(geom_beta_tr_hist)
    @decorators.nss_result_store("beta_rad", "theta_rad", "path_len", "times")
    def __call__(self, numtrajs, *args, **kwargs):
        """Throw numtrajs events and return valid betas."""
        self.throw(numtrajs)
        return self.beta_rad(), self.thetas(), self.pathLens(), self.val_times()

    def throw(self, times=None) -> None:
        """Throw N events with 1 * u random numbers for the ToO detection mode"""

        # Calculate the local nadir angle of the source
        self.times = self.generate_times(times)
        local_coords = self.too_source.localcoords(self.times)
        self.sourceNadRad = np.pi / 2 + local_coords.alt.rad

        self.alt_deg = local_coords.alt.deg
        self.az_deg = local_coords.az.deg

        # Define a cut if the source is below the horizon
        self.horizon_mask = self.sourceNadRad < self.alphaHorizon

        # Calculate the earth emergence angle from the nadir angle
        self.sourcebeta = self.get_beta_angle(self.sourceNadRad[self.horizon_mask])

        # Define a cut if the source is below the horizon
        self.volume_mask = self.sourcebeta < np.min(
            [np.radians(42), self.get_beta_angle(self.config.simulation.ang_from_limb)]
        )

        # Calculate the pathlength through the atmosphere
        self.losPathLen = self.get_path_length(
            self.sourcebeta[self.volume_mask], self.event_mask(self.sourceNadRad)
        )
        # self.test_plot_nadir_angle()

    def generate_times(self, times) -> np.ndarray:
        """
        Function to generate random times within the simulation time period
        """
        if isinstance(times, int):
            times = np.random.rand(times)

        if times is None:
            raise RuntimeError(
                "Provide a number of trajectories, or a 2D set of uniform random "
                "numbers in [0, 1]"
            )

        times = np.sort(times)
        times *= self.sourceOBSTime  # in s
        times = astropy.time.TimeDelta(times, format="sec")
        times = self.too_source.eventtime + times
        return times

    def get_beta_angle(self, nadir_angle):
        return np.arccos(
            ((self.core_alt) / self.config.constants.earth_radius) * np.sin(nadir_angle)
        )

    def get_path_length(self, beta, nadir_angle):
        return self.config.constants.earth_radius * np.cos(
            beta
        ) - self.core_alt * np.cos(nadir_angle + beta)

    def event_mask(self, x):
        return x[self.horizon_mask][self.volume_mask]

    def val_times(self):
        return self.event_mask(self.times)

    def betas(self):
        """Earth-emergence angles for valid events in degrees."""
        return np.rad2deg(self.beta_rad())

    def beta_rad(self):
        """Radian Earth-emergence angles for valid events."""
        return self.sourcebeta[self.volume_mask]

    def thetas(self):
        """View angles for valid events in radians"""
        return self.event_mask(self.sourceNadRad)

    def pathLens(self):
        """View angles for valid events."""
        return self.losPathLen

    def np_save(self, mcintfactor, numEvPass):
        np.savez(
            str(self.config.simulation.spectrum.log_nu_tau_energy) + "output.npz",
            t=(self.times - self.too_source.eventtime)[self.event_mask].to_value("hr"),
            tf=(self.times - self.too_source.eventtime).to_value("hr"),
            nad=self.sourceNadRad[self.event_mask],
            nadf=self.sourceNadRad,
            mcint=mcintfactor,
            npass=numEvPass,
            betas=self.too_betas(),
        )

    def McIntegral(
        self,
        triggers,
        costhetaChEff,
        tauexitprob,
        threshold,
        spec_norm,
        spec_weights_sum,
        **kwargs,
    ):

        lenDec = kwargs["lenDec"]
        method = kwargs["method"]
        if "store" in kwargs.keys():
            store = kwargs["store"]
        else:
            store = None

        if method not in ["Optical", "Radio"]:
            raise ValueError("method must be Optical or Radio")

        # calculate the Cherenkov angle
        thetaChEff = np.arccos(costhetaChEff)
        tanthetaChEff = np.tan(thetaChEff)

        # cossepangle = self.costhetaTrSubV[self.event_mask]

        mcintfactor = (
            (self.pathLens() - lenDec) * (self.pathLens() - lenDec) * tanthetaChEff**2
        )

        # Branching ratio set to 1 to be consistent
        Bshr = 1
        mcnorm = np.pi * Bshr

        mcintfactor *= mcnorm

        # Geometry Factors
        mcintegralgeoonly = np.mean(mcintfactor)

        # Multiply by tau exit probability
        mcintfactor *= tauexitprob

        # Weighting by energy spectrum if other than monoenergetic spectrum
        mcintfactor /= spec_norm
        mcintfactor /= spec_weights_sum

        # PE threshold
        mcintfactor[triggers < threshold] = 0

        # Define a cut based on sun and moon position
        if self.sun_moon_cut and method == "Optical":
            sun_moon_cut_mask = self.too_source.sun_moon_cut(self.val_times())
            mcintfactor[~sun_moon_cut_mask] = 0

        mcintegral = np.sum(mcintfactor) / len(self.times)
        mcintegraluncert = np.sqrt(np.var(mcintfactor, ddof=1) / len(self.times))

        numEvPass = np.count_nonzero(mcintfactor)

        if store is not None:
            col_name = "tmcintopt" if method == "Optical" else "tmcintrad"
            store([col_name], [mcintfactor])

        return mcintegral, mcintegralgeoonly, numEvPass, mcintegraluncert

    def test_skymap_plot(self, mcint):
        from matplotlib import cm

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="mollweide")
        ax.set_facecolor("k")
        color = cm.coolwarm
        az = self.event_mask(self.alt_deg)
        alt = self.event_mask(self.alt_deg)
        for i in range(len(mcint)):
            # color = [cmap(mcint[i]) for i in mcint]
            ax.plot(
                np.radians(az[i]) - np.pi,
                np.radians(alt[i]),
                ".",
                color=color(mcint[i]),
            )
        ax.tick_params(axis="x", colors="white")
        ax.grid(True, color="white")
        ax.set_title(
            f"Source: {np.rad2deg(self.config.simulation.source_RA)}°, {np.rad2deg(self.config.simulation.source_DEC)}°(RA, DEC)\n Detector: {np.rad2deg(self.config.detector.detlat)}°N, {np.rad2deg(self.config.detector.detlong)}°W, {self.config.detector.altitude}km"
        )
        plt.savefig("Movement_over_sky_5.png")
        plt.show()

    def test_plot_mcint(self, mcint):
        times = np.sort(
            self.event_mask(self.times.gps - np.amin(self.times.gps)) / 3600
        )
        mcint = np.take_along_axis(
            mcint,
            np.argsort(
                self.event_mask(self.times.gps - np.amin(self.times.gps)) / 3600
            ),
            0,
        )
        plt.figure()
        plt.plot(times, mcint, ".")
        plt.grid(True)
        plt.yscale("log")
        plt.xlabel("Observation time in hours")
        plt.ylabel("MC Integral factor")
        plt.savefig("Mcint_vs_time.png")

    def test_plot_nadir_angle(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(
            self.times.gps, np.rad2deg(self.sourceNadRad), ".", label="not visible"
        )
        plt.plot(
            self.event_mask(self.times.gps),
            np.rad2deg(self.event_mask(self.sourceNadRad)),
            ".",
            label="visible",
        )
        plt.axhline(np.rad2deg(self.alphaHorizon), label="Horizon")
        plt.axhline(np.rad2deg(self.get_beta_angle(np.radians(42))), label="End of sim")
        plt.legend()
        plt.xlabel("GPS time in s")
        plt.ylabel("Nadir angle in deg")
        plt.show()


def show_plot(sim, plot):
    plotfs = tuple([geom_beta_tr_hist])
    inputs = tuple([0])
    outputs = ("beta_rad", "theta_rad", "path_len")
    decorators.nss_result_plot_from_file(sim, inputs, outputs, plotfs, plot)
