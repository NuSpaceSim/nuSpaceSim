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
from astropy import units as u
from astropy.constants import R_earth
from astropy.time import TimeDelta

from ...utils import decorators
from .local_plots import geom_beta_tr_hist
from .too import ToOEvent

__all__ = ["RegionGeom", "RegionGeomToO"]


class RegionGeom:
    """Region Geometry of trajectories."""

    def __init__(self, config):
        self.config = config
        self.earth_radius: np.float64 = R_earth.to(u.km).value
        self.earth_rad_2: np.float64 = self.earth_radius**2

        self.core_alt = (
            self.earth_radius + self.config.detector.initial_position.altitude
        )
        self.detection_mode = self.config.simulation.mode
        self.sun_moon_cut = self.config.detector.sun_moon.sun_moon_cuts

        self.detLat = config.detector.initial_position.latitude
        self.detLong = config.detector.initial_position.longitude

        alphaHorizon = np.pi / 2 - np.arccos(self.earth_radius / self.core_alt)
        alphaMin = alphaHorizon - config.simulation.angle_from_limb
        minChordLen = 2 * np.sqrt(
            self.earth_rad_2 - (self.core_alt * np.sin(alphaMin)) ** 2
        )

        # Line of Sight path length
        self.minLOSpathLen = self.core_alt * np.cos(alphaMin) - minChordLen / 2
        self.maxLOSpathLen = np.sqrt(self.core_alt**2 - self.earth_rad_2)

        self.sinOfMaxThetaTrSubV = np.sin(config.simulation.max_cherenkov_angle)

        self.maxPhiS = config.simulation.max_azimuth_angle * 0.5
        self.minPhiS = config.simulation.max_azimuth_angle * -0.5

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
            2 * self.earth_radius * self.core_alt
        )
        self.thetaS = np.arccos(costhetaS)

        self.costhetaNSubV = (self.core_alt**2 - self.earth_rad_2 - rvsqrd) / (
            2 * self.earth_radius * self.losPathLen
        )

        thetaNSubV = np.arccos(self.costhetaNSubV)

        self.costhetaTrSubN = np.cos(self.thetaTrSubV) * self.costhetaNSubV - np.sin(
            self.thetaTrSubV
        ) * np.sin(thetaNSubV) * np.cos(self.phiTrSubV)

        self.thetaTrSubN = np.arccos(self.costhetaTrSubN)

        self.betaTrSubN = np.degrees(0.5 * np.pi - self.thetaTrSubN)

        # Compute latitude and longitude of spot on the ground (based on transforming
        # from detector's East-North_Up (ENU) frame back to the Earth-Centered-Earth-Fixed (ECEF) frame

        rsinlatS = np.cos(self.detLat) * np.sin(self.thetaS) * np.sin(
            self.phiS
        ) + np.sin(self.detLat) * np.cos(self.thetaS)

        latS_rad = np.arcsin(rsinlatS)
        self.latS = np.degrees(latS_rad)

        rxS = (
            -np.sin(self.detLong) * np.sin(self.thetaS) * np.cos(self.phiS)
            - np.cos(self.detLong)
            * np.sin(self.detLat)
            * np.sin(self.thetaS)
            * np.sin(self.phiS)
            + np.cos(self.detLong) * np.cos(self.detLat) * np.cos(self.thetaS)
        )

        ryS = (
            np.cos(self.detLong) * np.sin(self.thetaS) * np.cos(self.phiS)
            - np.sin(self.detLong)
            * np.sin(self.detLat)
            * np.sin(self.thetaS)
            * np.sin(self.phiS)
            + np.sin(self.detLong) * np.cos(self.detLat) * np.cos(self.thetaS)
        )

        longS_rad = np.arctan2(ryS, rxS)
        self.longS = np.degrees(longS_rad) % 360.0  # Unit test possible

        # Compute xy-coordinates and elevation and azimuthal angles of unit vector of the
        # line-of-sight (los) vector between detector and spot on the ground (technically, the
        # unit vector parallel to the los with tail at the center of the Earth) in ENU frame of the
        # spot on the ground

        self.elevAngVSubN = (
            0.5 * np.pi - thetaNSubV
        )  # Elevation angle of the detector w.r.t. the spot on the ground; unit test possible

        xVSubN = -np.sin(longS_rad) * np.cos(self.detLat) * np.cos(
            self.detLong
        ) + np.cos(longS_rad) * np.cos(self.detLat) * np.sin(self.detLong)

        yVSubN = (
            -np.cos(longS_rad)
            * np.sin(latS_rad)
            * np.cos(self.detLat)
            * np.cos(self.detLong)
            - np.sin(longS_rad)
            * np.sin(latS_rad)
            * np.cos(self.detLat)
            * np.sin(self.detLong)
            + np.cos(latS_rad) * np.sin(self.detLat)
        )

        self.aziAngVSubN = np.arctan2(
            yVSubN, xVSubN
        )  # Azimuthal angle of the detector w.r.t. the spot on the ground

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

    def phis(self):
        return self.phiTrSubV[self.event_mask]

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

    def valid_longS(self):
        return self.longS[self.event_mask]

    def valid_latS(self):
        return self.latS[self.event_mask]

    def valid_longS_rad(self):
        return np.radians(self.valid_longS())

    def valid_latS_rad(self):
        return np.radians(self.valid_latS())

    def valid_elevAngVSubN(self):
        return self.elevAngVSubN[self.event_mask]

    def valid_aziAngVSubN(self):
        return self.aziAngVSubN[self.event_mask]

    def find_lat_long_along_traj(self, dist_along_traj):
        # Compute xyz-coordinates in ENU frame of los between detector and spot on the ground

        xPath_v = dist_along_traj * np.sin(self.thetas()) * np.cos(self.phis())

        yPath_v = dist_along_traj * np.sin(self.thetas()) * np.sin(
            self.phis()
        ) + self.earth_radius * np.cos(self.valid_elevAngVSubN())

        zPath_v = dist_along_traj * np.cos(self.thetas()) + self.earth_radius * np.sin(
            self.valid_elevAngVSubN()
        )

        # Compute xyz-coordinates in the spot's ENU frame (accomplished by transforming from
        # the los ENU frame to the spot's ENU frame)

        xPath_n = (
            -np.sin(self.valid_aziAngVSubN()) * xPath_v
            - np.cos(self.valid_aziAngVSubN())
            * np.sin(self.valid_elevAngVSubN())
            * yPath_v
            + np.cos(self.valid_aziAngVSubN())
            * np.cos(self.valid_elevAngVSubN())
            * zPath_v
        )

        yPath_n = (
            np.cos(self.valid_aziAngVSubN()) * xPath_v
            - np.sin(self.valid_aziAngVSubN())
            * np.sin(self.valid_elevAngVSubN())
            * yPath_v
            + np.sin(self.valid_aziAngVSubN())
            * np.cos(self.valid_elevAngVSubN())
            * zPath_v
        )

        zPath_n = (
            np.cos(self.valid_elevAngVSubN()) * yPath_v
            + np.sin(self.valid_elevAngVSubN()) * zPath_v
        )

        # Compute xyz-coordinates in the ECEF frame (accomplished by transforming from the spot's
        # ENU frame to the ECEF frame)

        xPath_ECEF = (
            -np.sin(self.valid_longS_rad()) * xPath_n
            - np.cos(self.valid_longS_rad()) * np.sin(self.valid_latS_rad()) * yPath_n
            + np.cos(self.valid_longS_rad()) * np.cos(self.valid_latS_rad()) * zPath_n
        )

        yPath_ECEF = (
            np.cos(self.valid_longS_rad()) * xPath_n
            - np.sin(self.valid_longS_rad()) * np.sin(self.valid_latS_rad()) * yPath_n
            + np.sin(self.valid_longS_rad()) * np.cos(self.valid_latS_rad()) * zPath_n
        )

        zPath_ECEF = (
            np.cos(self.valid_latS_rad()) * yPath_n
            + np.sin(self.valid_latS_rad()) * zPath_n
        )

        dist2EarthCenter = np.sqrt(xPath_ECEF**2 + yPath_ECEF**2 + zPath_ECEF**2)

        latPath = np.arcsin(zPath_ECEF / dist2EarthCenter)
        longPath = np.arctan2(yPath_ECEF, xPath_ECEF)

        return latPath, longPath

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

        # Number of Trajectories
        numTrajs = len(self.betaTrSubN)

        # Geometry Factors
        mcintfactor[cossepangle < costheta] = 0
        mcintegralgeoonly = np.sum(mcintfactor) * mcnorm / numTrajs

        Bshr = 0.826

        # Multiply by tau exit probability and branching ratio (currently ignores muon channel for tau decays; it's also hard coded in, so will need to be changed)
        mcintfactor *= Bshr * tauexitprob

        # Weighting by energy spectrum if other than monoenergetic spectrum
        mcintfactor /= spec_norm
        mcintfactor /= spec_weights_sum

        # PE threshold
        mcintfactor[triggers < threshold] = 0
        mcintegral = np.sum(mcintfactor) * mcnorm / numTrajs
        mcintegraluncert = np.sqrt(np.var(mcintfactor, ddof=1) / numTrajs) * mcnorm

        numEvPass = np.count_nonzero(mcintfactor)

        return mcintegral, mcintegralgeoonly, numEvPass, mcintegraluncert


class RegionGeomToO:
    def __init__(self, config):
        self.config = config
        self.earth_radius: np.float64 = R_earth.to(u.km).value
        self.earth_rad_2: np.float64 = self.earth_radius**2

        self.core_alt = (
            self.earth_radius + self.config.detector.initial_position.altitude
        )
        self.detection_mode = self.config.simulation.mode
        self.sun_moon_cut = self.config.detector.sun_moon.sun_moon_cuts

        self.detLat = config.detector.initial_position.latitude
        self.detLong = config.detector.initial_position.longitude

        self.sourceOBSTime = self.config.simulation.target.source_obst
        self.too_source = ToOEvent(self.config)

        self.alphaHorizon = 0.5 * np.pi - np.arccos(self.earth_radius / self.core_alt)

    @decorators.nss_result_plot(geom_beta_tr_hist)
    @decorators.nss_result_store("beta_rad", "theta_rad", "path_len", "times")
    def __call__(self, numtrajs, *args, **kwargs):
        """Throw numtrajs events and return valid betas."""
        self.throw(numtrajs)
        return self.beta_rad(), self.thetas(), self.pathLens(), self.val_times()

    def throw(self, times=None) -> None:
        """Throw N events with 1 * u random numbers for the Target detection mode"""

        # Calculate the local nadir angle of the source
        self.times = self.generate_times(times)
        local_coords = self.too_source.localcoords(self.times)
        self.sourceNadRad = 0.5 * np.pi + local_coords.alt.rad

        self.alt_deg = local_coords.alt.deg
        self.az_deg = local_coords.az.deg

        # Define a cut if the source is below the horizon
        self.horizon_mask = self.sourceNadRad < self.alphaHorizon

        # Calculate the earth emergence angle from the nadir angle
        self.sourcebeta = self.get_beta_angle(self.sourceNadRad[self.horizon_mask])

        # Define a cut if the source is below the horizon
        self.volume_mask = self.sourcebeta < np.min(
            [
                np.radians(42),
                self.get_beta_angle(
                    self.alphaHorizon - self.config.simulation.angle_from_limb
                ),
            ]
        )

        # Calculate the pathlength through the atmosphere
        self.losPathLen = self.get_path_length(
            self.sourcebeta[self.volume_mask], self.event_mask(self.sourceNadRad)
        )

    def generate_times(self, times) -> np.ndarray:
        """
        Function to generate times within the simulation time period
        """
        if isinstance(times, int):
            times = np.arange(times) / times

        if times is None:
            raise RuntimeError(
                "Provide a number of trajectories, or a 2D set of uniform random "
                "numbers in [0, 1]"
            )

        times *= self.sourceOBSTime  # in s
        times = TimeDelta(times, format="sec")
        times = self.too_source.eventtime + times
        return times

    def get_beta_angle(self, nadir_angle):
        return np.arccos(((self.core_alt) / self.earth_radius) * np.sin(nadir_angle))

    def get_path_length(self, beta, nadir_angle):
        return self.core_alt * np.cos(nadir_angle + beta) / np.cos(beta)

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

    def find_lat_long_along_traj(self, dist_along_traj):
        """Will have to work out the geometry for this."""
        return self.detLat * np.ones_like(dist_along_traj), self.detLong * np.ones_like(
            dist_along_traj
        )

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

    def mcintegral(
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

        mcintfactor_umsk = self.pathLens() - lenDec
        # mcintfactor_umsk = self.pathLens() # For testing purposes only

        mcintfactor = np.where(mcintfactor_umsk > 0.0, mcintfactor_umsk, 0.0)

        mcintfactor *= (self.pathLens() - lenDec) * tanthetaChEff**2
        # mcintfactor *= self.pathLens() * tanthetaChEff**2 # For testing purposes only

        # Geometry Factors

        mcintfactor *= np.pi

        mcintegralgeoonly = np.sum(mcintfactor) / len(self.times)

        Bshr = 0.826

        # Multiply by tau exit probability and branching ratio (currently ignores muon channel for tau decays; it's also hard coded in, so will need to be changed)

        mcintfactor *= Bshr * tauexitprob

        # Weighting by energy spectrum if other than monoenergetic spectrum
        mcintfactor /= spec_norm
        mcintfactor /= spec_weights_sum

        # PE threshold
        mcintfactor[triggers < threshold] = 0

        # Define a cut based on sun and moon position
        if self.sun_moon_cut and method == "Optical":
            sun_moon_cut_mask = self.too_source.sun_moon_cut(self.val_times())
            mcintfactor[~sun_moon_cut_mask] = 0

            # self.test_plot_sunmooncut(self.too_source.sun_moon_cut)

        mcintegral = np.sum(mcintfactor) / len(self.times)
        mcintegraluncert = np.sqrt(np.var(mcintfactor, ddof=1) / len(self.times))

        numEvPass = np.count_nonzero(mcintfactor)

        if store is not None:
            col_name = "tmcintopt" if method == "Optical" else "tmcintrad"
            store([col_name], [mcintfactor])

        return mcintegral, mcintegralgeoonly, numEvPass, mcintegraluncert


def show_plot(sim_results, simclass, plot):
    plotfs = tuple([geom_beta_tr_hist])
    inputs = tuple([0])
    outputs = ("beta_rad", "theta_rad", "path_len")
    decorators.nss_result_plot_from_file(
        sim_results, simclass, inputs, outputs, plotfs, plot
    )
