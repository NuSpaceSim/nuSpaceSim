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

import astropy.coordinates
import astropy.time
import numpy as np
from astropy import units as u
from astropy.constants import R_earth

from ...config import Detector, NssConfig, Simulation
from ...utils import decorators
from .local_plots import geom_beta_tr_hist
from .too import ToOEvent

# from astropy.time import TimeDelta


# table_max_beta_e = 0.733038 # radians -- 42 degrees (upper limit on beta_E in nupyprop tables)
table_max_beta_e_deg = 42  # degrees
table_max_beta_e = np.radians(table_max_beta_e_deg)
table_eta_max = table_max_beta_e + np.pi / 2

__all__ = ["RegionGeomMonteCarlo", "RegionGeomTargetApprox"]


class RegionGeomMonteCarlo:
    """Region Geometry of trajectories."""

    def __init__(self, config: NssConfig):
        self.config = config
        self.obs_type = self.config.simulation.mode
        self.num_time_bins = self.config.simulation.num_time_bins
        self.num_events_per_time_bin = (
            self.config.simulation.integ_method.num_events_per_time_bin
        )

        self.earth_radius: np.float64 = R_earth.to(u.km).value
        self.earth_rad_2: np.float64 = self.earth_radius**2

        self.initial_det_height = self.config.detector.initial_position.altitude
        self.det_height = (
            self.initial_det_height
        )  # Will need to be updated for moving platforms

        self.core_alt = self.earth_radius + self.det_height

        self.sun_moon_cut = self.config.detector.sun_moon.sun_moon_cuts

        self.alphaHorizon = np.pi / 2 - np.arccos(self.earth_radius / self.core_alt)

        self.initial_detLat = self.config.detector.initial_position.latitude
        self.initial_detLong = self.config.detector.initial_position.longitude

        self.detLat = (
            self.initial_detLat
        )  # Will need to be updated for moving platforms
        self.detLong = self.initial_detLong

        self.nadir_span = self.config.detector.field_of_view.nadir_span
        self.azimuth_span = self.config.detector.field_of_view.azimuth_span

        if isinstance(self.config.detector.pointing, Detector.LimbPoint):
            self.nadir_center = (
                self.config.detector.pointing.nadir_center_wrt_limb + self.alphaHorizon
            )
            self.azimuth_center = self.config.detector.pointing.azimuth_center

        elif isinstance(self.config.detector.pointing, Detector.DetRefPoint):
            self.nadir_center = (
                self.config.detector.pointing.tilt_angle_center_wrt_horiz + np.pi / 2
            )
            self.azimuth_center = self.config.detector.pointing.azimuth_center_det_ref

        elif isinstance(self.config.detector.pointing, Detector.SourceTracking):
            self.det_center_RA = self.config.detector.pointing.tracked_source_RA
            self.det_center_dec = self.config.detector.pointing.tracked_source_DEC
            self.obs_date_time = self.config.detector.pointing.obs_date_and_time
            self.obs_date_time_format = (
                self.config.detector.pointing.obs_date_and_time_format
            )

            self.time_of_obs = astropy.time.Time(
                self.obs_date_time, format=self.obs_date_time_format, scale="utc"
            )

            self.det_center_coords = astropy.coordinates.SkyCoord(
                ra=self.det_center_RA * u.rad,
                dec=self.det_center_dec * u.rad,
                frame="icrs",
            )

            self.det_earth_coords = astropy.coordinates.EarthLocation(
                lat=self.detLat * u.rad,
                lon=self.detLong * u.rad,
                height=self.det_height * 1000 * u.m,
            )

            self.det_frame = astropy.coordinates.AltAz(
                obstime=self.time_of_obs, location=self.det_earth_coords
            )

            self.det_center_coords.transform_to(self.det_frame)

            self.nadir_center = np.pi / 2 + self.det_center_coords.alt.rad
            self.azimuth_center = self.det_center_coords.az.rad

        else:
            RuntimeError(
                f"Unrecognized Pointing Type: {type(self.config.detector.pointing)}!"
            )

        # Define region of acceptance for events

        self.det_alpha_min = self.nadir_center - self.nadir_span / 2

        table_alpha_min = np.arcsin(
            np.sin(table_eta_max) * self.earth_radius / self.core_alt
        )

        if self.det_alpha_min < table_alpha_min:
            self.acc_reg_alpha_min = table_alpha_min
        else:
            self.acc_reg_alpha_min = self.det_alpha_min

        self.det_alpha_max = self.nadir_center + self.nadir_span / 2

        if self.det_alpha_max > self.alphaHorizon:
            self.acc_reg_alpha_max = self.alphaHorizon

        else:
            self.acc_reg_alpha_max = self.det_alpha_max

        self.sin_of_acc_reg_alpha_min = np.sin(self.acc_reg_alpha_min)
        self.sin_of_acc_reg_alpha_max = np.sin(self.acc_reg_alpha_max)

        self.det_phi_min = self.azimuth_center - self.azimuth_span / 2
        self.det_phi_max = self.azimuth_center + self.azimuth_span / 2

        self.acc_reg_min_phi_s = self.det_phi_min
        self.acc_reg_max_phi_s = self.det_phi_max

        normPhiS = np.reciprocal(self.acc_reg_max_phi_s - self.acc_reg_min_phi_s)

        if self.obs_type == "Diffuse":
            # if self.det_alpha_min < table_alpha_min:
            #    self.alphaMin = table_alpha_min
            # else:
            #    self.alphaMin = self.det_alpha_min

            maxChordLen = 2 * np.sqrt(
                self.earth_rad_2 - (self.core_alt * np.sin(self.acc_reg_alpha_min)) ** 2
            )

            # Minimum line of Sight path length

            self.minLOSpathLen = (
                self.core_alt * np.cos(self.acc_reg_alpha_min) - maxChordLen / 2
            )

            if self.det_alpha_max > self.alphaHorizon:
                self.maxLOSpathLen = np.sqrt(self.core_alt**2 - self.earth_rad_2)

            else:
                minChordLen = 2 * np.sqrt(
                    self.earth_rad_2
                    - (self.core_alt * np.sin(self.acc_reg_alpha_max)) ** 2
                )
                self.maxLOSpathLen = (
                    self.core_alt * np.cos(self.acc_reg_alpha_max) - minChordLen / 2
                )

            self.sinOfMaxThetaTrSubV = np.sin(config.simulation.max_cherenkov_angle)

            normThetaTrSubV = 2 / self.sinOfMaxThetaTrSubV**2
            normPhiTrSubV = np.reciprocal(2 * np.pi)

            bracketForNormThetaS = (
                (self.core_alt**2 - self.earth_rad_2) * self.maxLOSpathLen
                - (1.0 / 3.0) * self.maxLOSpathLen**3
                - (self.core_alt**2 - self.earth_rad_2) * self.minLOSpathLen
                + (1.0 / 3.0) * self.minLOSpathLen**3
            )

            normThetaS = 2.0 * self.core_alt * self.earth_rad_2 / bracketForNormThetaS

            pdfnorm = normThetaTrSubV * normPhiTrSubV * normPhiS * normThetaS
            self.mcnorm = self.earth_rad_2 / pdfnorm

        elif self.obs_type == "Target":
            eta_at_min = np.pi - np.arcsin(
                (self.core_alt / self.earth_radius) * self.sin_of_acc_reg_alpha_min
            )
            self.acc_reg_min_theta_s = np.pi - (self.acc_reg_alpha_min + eta_at_min)
            self.acc_reg_max_cos_theta_s = np.cos(self.acc_reg_min_theta_s)

            eta_at_max = np.pi - np.arcsin(
                (self.core_alt / self.earth_radius) * self.sin_of_acc_reg_alpha_max
            )
            self.acc_reg_max_theta_s = np.pi - (self.acc_reg_alpha_max + eta_at_max)
            self.acc_reg_min_cos_theta_s = np.cos(self.acc_reg_max_theta_s)

            normThetaS = 2 * np.reciprocal(
                self.acc_reg_max_cos_theta_s**2 - self.acc_reg_min_cos_theta_s**2
            )

            pdfnorm = normPhiS * normThetaS
            self.mcnorm = self.earth_rad_2 / pdfnorm

            self.sourceOBSTime = self.config.simulation.target.source_obst
            self.too_source = ToOEvent(self.config)

        else:
            RuntimeError(f"Unrecognized Observation Type: {type(self.obs_type)}!")

    def throw(self, u=None):
        self.times = self.generate_times(self.num_time_bins)

        local_coords = self.too_source.localcoords(
            self.times
        )  # Will need updating for moving platforms

        """Throw N events with 4 * u random numbers"""

        if self.obs_type == "Diffuse":
            if isinstance(u, int):
                # fix to make closed in [0,1]

                u = np.random.rand(4, u)

                # Alternative method (slower?)
                # u1 = np.random.rand(self.time_bins, u)
                # u2 = np.random.rand(self.time_bins, u)
                # u3 = np.random.rand(self.time_bins, u)
                # u4 = np.random.rand(self.time_bins, u)

            if u is None:
                raise RuntimeError(
                    "Provide a number of trajectories, or a 2D set of uniform random "
                    "numbers in [0, 1]"
                )

            if u.shape[0] != 4:
                raise RuntimeError("u random numbers must be of shape (4, N)")

            u1, u2, u3, u4 = u

            self.numTrajs = u1.size

            u1 = np.broadcast_to(u1, (self.num_time_bins, u1.size)).copy()
            u2 = np.broadcast_to(u2, (self.num_time_bins, u2.size)).copy()
            u3 = np.broadcast_to(u3, (self.num_time_bins, u3.size)).copy()
            u4 = np.broadcast_to(u4, (self.num_time_bins, u4.size)).copy()

            self.thetaTrSubV = np.arcsin(self.sinOfMaxThetaTrSubV * np.sqrt(u1))
            self.costhetaTrSubV = np.cos(self.thetaTrSubV)
            self.phiTrSubV = 2.0 * np.pi * u2
            self.phiS = (
                self.acc_reg_max_phi_s - self.acc_reg_min_phi_s
            ) * u3 + self.acc_reg_min_phi_s

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

            self.thetaNSubV = np.arccos(self.costhetaNSubV)

            self.costhetaTrSubN = np.cos(
                self.thetaTrSubV
            ) * self.costhetaNSubV - np.sin(self.thetaTrSubV) * np.sin(
                self.thetaNSubV
            ) * np.cos(
                self.phiTrSubV
            )

            self.thetaTrSubN = np.arccos(self.costhetaTrSubN)

            self.betaTrSubN = np.degrees(np.pi / 2 - self.thetaTrSubN)

            self.event_mask = np.logical_and(
                self.costhetaTrSubN >= 0, self.betaTrSubN < table_max_beta_e_deg
            )
        elif self.obs_type == "Target":
            if isinstance(u, int):
                # fix to make closed in [0,1]

                u = np.random.rand(2, u)

                # Alternative method (slower?)
                # u1 = np.random.rand(self.time_bins, u)
                # u2 = np.random.rand(self.time_bins, u)

            if u is None:
                raise RuntimeError(
                    "Provide a number of trajectories, or a 2D set of uniform random "
                    "numbers in [0, 1]"
                )

            if u.shape[0] != 2:
                raise RuntimeError("u random numbers must be of shape (2, N)")

            u1, u2 = u

            self.numTrajs = u1.size

            u1 = np.broadcast_to(u1, (self.num_time_bins, u1.size)).copy()
            u2 = np.broadcast_to(u2, (self.num_time_bins, u2.size)).copy()

            self.phiS = (
                self.acc_reg_max_phi_s - self.acc_reg_min_phi_s
            ) * u1 + self.acc_reg_min_phi_s
            self.costhetaS = np.sqrt(
                u2 * (self.acc_reg_max_cos_theta_s**2 - self.acc_reg_min_cos_theta_s**2)
                + self.acc_reg_min_cos_theta_s**2
            )
            self.thetaS = np.arccos(self.costhetaS)
            self.sinthetaS = np.sin(self.thetaS)

            self.losPathLen = np.sqrt(
                2 * self.earth_radius * (1 - self.costhetaS) * self.core_alt
                + self.det_altitude**2
            )

            # self.times = self.generate_times(self.time_bins)

            # local_coords = self.too_source.localcoords(self.times) # Will need updating for moving platforms

            self.sourceNadRad = np.pi / 2 + local_coords.alt.rad
            self.sourceAziRad = local_coords.az.rad

            # Define a cut around the region where the source is observable

            nadir_range_min = (
                self.det_alpha_min - self.config.simulation.max_cherenkov_angle
            )
            nadir_range_max = (
                self.det_alpha_max + self.config.simulation.max_cherenkov_angle
            )

            src_in_nadir_range = np.logical_and(
                self.sourceNadRad >= nadir_range_min,
                self.sourceNadRad <= nadir_range_max,
            )

            azi_range_min = (
                self.det_phi_min - self.config.simulation.max_cherenkov_angle
            )
            azi_range_max = (
                self.det_phi_max + self.config.simulation.max_cherenkov_angle
            )

            src_in_azi_range = np.logical_and(
                self.sourceAziRad >= azi_range_min, self.sourceAziRad <= azi_range_max
            )

            self.src_is_observable_mask = np.logical_and(
                src_in_nadir_range, src_in_azi_range
            )

            # Define a cut if the source is below the horizon
            # self.horizon_mask = self.sourceNadRad < self.alphaHorizon

            eta_at_src = np.where(
                self.sourceNadRad < self.alphaHorizon,
                (
                    np.pi
                    - np.arcsin(
                        (self.core_alt / self.earth_radius) * np.sin(self.sourceNadRad)
                    )
                ),
                (np.pi / 2),
            )

            self.theta_src = np.pi - (self.sourceNadRad + eta_at_src)
            self.sin_theta_src = np.sin(self.theta_src)
            self.cos_theta_src = np.cos(self.theta_src)

            self.src_los_path_len = np.where(
                self.sourceNadRad < self.alphaHorizon,
                (
                    np.sqrt(
                        2 * self.earth_radius * (1 - self.cos_theta_src) * self.core_alt
                        + self.det_altitude**2
                    )
                ),
                (self.core_alt * self.sin_theta_src),
            )

            self.costhetaTrSubV = np.where(
                self.sourceNadRad < self.alphaHorizon,
                (1 / (self.src_los_path_len * self.losPathLen))
                * (
                    self.earth_rad_2
                    * self.sin_theta_src
                    * self.sinthetaS
                    * np.cos(self.sourceAziRad - self.phiS)
                    + (self.core_alt - self.earth_radius * self.cos_theta_src)
                    * (self.core_alt - self.earth_radius * self.costhetaS)
                ),
                (1 / self.losPathLen)
                * (
                    self.earth_radius
                    * self.sinthetaS
                    * np.sin(self.sourceNadRad)
                    * np.cos(self.sourceAziRad - self.phiS)
                    + (self.core_alt - self.earth_radius * self.costhetaS)
                    * np.cos(self.sourceNadRad)
                ),
            )

            self.costhetaTrSubN = np.where(
                self.sourceNadRad < self.alphaHorizon,
                (1 / self.src_los_path_len)
                * (
                    (self.core_alt - self.earth_radius * self.cos_theta_src)
                    * self.costhetaS
                    - self.earth_radius
                    * self.sin_theta_src
                    * self.sinthetaS
                    * np.cos(self.sourceAziRad - self.phiS)
                ),
                -1
                * self.sinthetaS
                * np.sin(self.sourceNadRad)
                * np.cos(self.sourceAziRad - self.phiS)
                + np.cos(self.sourceNadRad) * self.costhetaS,
            )

            self.thetaTrSubV = np.arccos(self.costhetaTrSubV)

            self.thetaTrSubN = np.arccos(self.costhetaTrSubN)

            self.betaTrSubN = np.degrees(0.5 * np.pi - self.thetaTrSubN)

            self.sinbetaTrSubN = np.sin(0.5 * np.pi - self.thetaTrSubN)

            event_traj_mask = np.logical_and(
                self.costhetaTrSubN >= 0, self.betaTrSubN < table_max_beta_e_deg
            )
            self.event_mask = np.logical_and(
                event_traj_mask, self.src_is_observable_mask
            )

        else:
            RuntimeError(f"Unrecognized Observation Type: {type(self.obs_type)}!")

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
            np.pi / 2 - self.thetaNSubV
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

        # self.event_mask = np.logical_and(self.costhetaTrSubN >= 0, self.betaTrSubN < table_max_beta_e_deg)

    def generate_times(self, times) -> np.ndarray:
        """
        Function to generate times within the simulation time period
        """
        if isinstance(times, int):
            # times_arr = np.arange(times) / times
            times_arr_nb = np.arange(times) / times
            times_arr = np.broadcast_to(
                times_arr_nb[:, np.newaxis], (times, self.num_events_per_time_bin)
            ).copy()

        if times is None:
            raise RuntimeError("Provide a number of time bins")

        times_arr *= self.sourceOBSTime  # in s
        times_arr = astropy.time.TimeDelta(times_arr, format="sec")
        times_arr = self.too_source.eventtime + times_arr
        return times_arr

    def valid_times(self):
        if self.obs_type == "Diffuse":
            val_times = self.times
        elif self.obs_type == "Target":
            val_times = self.times[self.src_is_observable_mask]
        else:
            RuntimeError(f"Unrecognized Observation Type: {type(self.obs_type)}!")
        return val_times

        # return self.times[self.src_is_observable_mask]

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

    def valid_costhetaS(self):
        return self.costhetaS[self.event_mask]

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
    @decorators.nss_result_store("beta_rad", "theta_rad", "path_len", "times")
    # def __call__(self, numtrajs, *args, **kwargs):
    def __call__(self, *args, **kwargs):
        """Throw numtrajs events and return valid betas."""
        # self.throw(numtrajs)
        self.throw(self.num_events_per_time_bin)
        return self.beta_rad(), self.thetas(), self.pathLens(), self.val_times()

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

        # if self.obs_type == "Diffuse":
        # u1 = np.broadcast_to(u1, (self.num_time_bins,u1.size)).copy()
        # u2 = np.broadcast_to(u2, (self.num_time_bins,u2.size)).copy()
        # u3 = np.broadcast_to(u3, (self.num_time_bins,u3.size)).copy()
        # u4 = np.broadcast_to(u4, (self.num_time_bins,u4.size)).copy(

        cossepangle = self.costhetaTrSubV[self.event_mask]

        if self.obs_type == "Diffuse":
            mcintfactor = (
                self.valid_costhetaTrSubN()
                / self.valid_costhetaNSubV()
                / self.valid_costhetaTrSubV()
            )
        elif self.obs_type == "Target":
            mcintfactor = self.valid_costhetaTrSubN() / self.valid_costhetaS()
        else:
            RuntimeError(f"Unrecognized Observation Type: {type(self.obs_type)}!")

        mcnorm = self.mcnorm

        # Number of Trajectories
        # numTrajs = len(self.betaTrSubN)

        # Geometry Factors
        mcintfactor[cossepangle < costheta] = 0
        mcintegralgeoonly = np.sum(mcintfactor, axis=1) * mcnorm / self.numTrajs

        Bshr = 0.826

        # Multiply by tau exit probability and branching ratio (currently ignores muon channel for tau decays; it's also hard coded in, so will need to be changed)
        mcintfactor *= Bshr * tauexitprob

        # Weighting by energy spectrum if other than monoenergetic spectrum
        mcintfactor /= spec_norm
        mcintfactor /= spec_weights_sum

        # PE threshold
        mcintfactor[triggers < threshold] = 0
        mcintegral = np.sum(mcintfactor, axis=1) * mcnorm / self.numTrajs
        mcintegraluncert = (
            np.sqrt(np.var(mcintfactor, axis=1, ddof=1) / self.numTrajs) * mcnorm
        )

        numEvPass = np.count_nonzero(mcintfactor, axis=1)

        # elif self.obs_type == "Target":
        # else:
        #    RuntimeError(f"Unrecognized Observation Type: {type(self.obs_type)}!")

        return mcintegral, mcintegralgeoonly, numEvPass, mcintegraluncert


class RegionGeomTargetApprox:
    def __init__(self, config: NssConfig):
        self.config = config
        self.num_time_bins = self.config.simulation.num_time_bins

        self.earth_radius: np.float64 = R_earth.to(u.km).value
        self.earth_rad_2: np.float64 = self.earth_radius**2

        self.initial_detLat = self.config.detector.initial_position.latitude
        self.initial_detLong = self.config.detector.initial_position.longitude

        self.detLat = (
            self.initial_detLat
        )  # Will need to be updated for moving platforms
        self.detLong = self.initial_detLong

        self.initial_det_height = self.config.detector.initial_position.altitude
        self.det_altitude = self.initial_det_altitude

        self.core_alt = self.earth_radius + self.det_altitude
        self.integ_method = self.config.simulation.integ_method
        # self.detection_mode = self.config.simulation.mode
        self.sun_moon_cut = self.config.detector.sun_moon.sun_moon_cuts

        self.alphaHorizon = np.pi / 2 - np.arccos(self.earth_radius / self.core_alt)

        self.nadir_span = self.config.detector.field_of_view.nadir_span
        self.azimuth_span = self.config.detector.field_of_view.azimuth_span

        if isinstance(self.config.detector.pointing, Detector.LimbPoint):
            self.nadir_center = (
                self.config.detector.pointing.nadir_center_wrt_limb + self.alphaHorizon
            )
            self.azimuth_center = self.config.detector.pointing.azimuth_center

        elif isinstance(self.config.detector.pointing, Detector.DetRefPoint):
            self.nadir_center = (
                self.config.detector.pointing.tilt_angle_center_wrt_horiz + np.pi / 2
            )
            self.azimuth_center = self.config.detector.pointing.azimuth_center_det_ref

        elif isinstance(self.config.detector.pointing, Detector.SourceTracking):
            self.det_center_RA = self.config.detector.pointing.tracked_source_RA
            self.det_center_dec = self.config.detector.pointing.tracked_source_DEC
            self.obs_date_time = self.config.detector.pointing.obs_date_and_time
            self.obs_date_time_format = (
                self.config.detector.pointing.obs_date_and_time_format
            )

            self.time_of_obs = astropy.time.Time(
                self.obs_date_time, format=self.obs_date_time_format, scale="utc"
            )

            self.det_center_coords = astropy.coordinates.SkyCoord(
                ra=self.det_center_RA * u.rad,
                dec=self.det_center_dec * u.rad,
                frame="icrs",
            )

            self.det_earth_coords = astropy.coordinates.EarthLocation(
                lat=self.detLat * u.rad,
                lon=self.detLong * u.rad,
                height=self.det_height * 1000 * u.m,
            )

            self.det_frame = astropy.coordinates.AltAz(
                obstime=self.time_of_obs, location=self.det_earth_coords
            )

            self.det_center_coords.transform_to(self.det_frame)

            self.nadir_center = np.pi / 2 + self.det_center_coords.alt.rad
            self.azimuth_center = self.det_center_coords.az.rad

        else:
            RuntimeError(
                f"Unrecognized Pointing Type: {type(self.config.detector.pointing)}!"
            )

        # Define region of acceptance for events

        self.det_alpha_min = self.nadir_center - self.nadir_span / 2

        table_alpha_min = np.arcsin(
            np.sin(table_eta_max) * self.earth_radius / self.core_alt
        )

        self.acc_reg_alpha_min = np.where(
            self.det_alpha_min < table_alpha_min, table_alpha_min, self.det_alpha_min
        )

        # if self.det_alpha_min < table_alpha_min:
        #    self.acc_reg_alpha_min = table_alpha_min
        # else:
        #    self.acc_reg_alpha_min = self.det_alpha_min

        self.det_alpha_max = self.nadir_center + self.nadir_span / 2

        self.acc_reg_alpha_max = np.where(
            self.det_alpha_max > self.alphaHorizon,
            self.alphaHorizon,
            self.det_alpha_max,
        )

        # if self.det_alpha_max > self.alphaHorizon:
        #    self.acc_reg_alpha_max = self.alphaHorizon
        # else:
        #    self.acc_reg_alpha_max = self.det_alpha_max

        self.sin_of_acc_reg_alpha_min = np.sin(self.acc_reg_alpha_min)
        self.sin_of_acc_reg_alpha_max = np.sin(self.acc_reg_alpha_max)

        self.det_phi_min = self.azimuth_center - self.azimuth_span / 2
        self.det_phi_max = self.azimuth_center + self.azimuth_span / 2

        self.acc_reg_min_phi_s = self.det_phi_min
        self.acc_reg_max_phi_s = self.det_phi_max

        # normPhiS = np.reciprocal(self.acc_reg_max_phi_s - self.acc_reg_min_phi_s)

        # eta_at_min = np.pi - np.arcsin((self.core_alt / self.earth_radius) * self.sin_of_acc_reg_alpha_min)
        # self.acc_reg_min_theta_s = pi - (self.acc_reg_alpha_min + eta_at_min)
        # self.acc_reg_max_cos_theta_s = np.cos(self.acc_reg_min_theta_s)

        # eta_at_max = np.pi - np.arcsin((self.core_alt / self.earth_radius) * self.sin_of_acc_reg_alpha_max)
        # self.acc_reg_max_theta_s = pi - (self.acc_reg_alpha_max + eta_at_max)
        # self.acc_reg_min_cos_theta_s = np.cos(self.acc_reg_max_theta_s)

        # normThetaS = 2 * np.reciprocal(self.acc_reg_max_cos_theta_s**2 - self.acc_reg_min_cos_theta_s**2)

        # pdfnorm = normPhiS * normThetaS
        # self.mcnorm = self.earth_rad_2 / pdfnorm

        self.sourceOBSTime = self.config.simulation.target.source_obst
        self.too_source = ToOEvent(self.config)

    @decorators.nss_result_plot(geom_beta_tr_hist)
    @decorators.nss_result_store("beta_rad", "theta_rad", "path_len", "times")
    # def __call__(self, numtrajs, *args, **kwargs):
    def __call__(self, *args, **kwargs):
        """Throw numtrajs events and return valid betas."""
        # self.throw(numtrajs)
        self.throw(self.num_time_bins)
        return self.beta_rad(), self.thetas(), self.pathLens(), self.val_times()

    def throw(self, times=None) -> None:
        """Throw N events with 1 * u random numbers for the Target detection mode"""

        if isinstance(u, int):
            self.times = self.generate_times(u)
            local_coords = self.too_source.localcoords(
                self.times
            )  # Will need updating for moving platforms
            self.sourceNadRad = np.pi / 2 + local_coords.alt.rad

            self.alt_deg = local_coords.alt.deg
            self.az_deg = local_coords.az.deg

            # Create a mask for those times for which the source is above the maximum nadir angle

            # self.horizon_mask = self.sourceNadRad < self.alphaHorizon
            self.horizon_mask = self.sourceNadRad < self.acc_reg_alpha_max

            # Calculate the earth emergence angle from the nadir angle
            self.sourcebeta = self.get_beta_angle(self.sourceNadRad[self.horizon_mask])

            # Define a cut if the source is below the horizon
            self.volume_mask = self.sourcebeta < np.min(
                [
                    table_max_beta_e,
                    # self.get_beta_angle(
                    #    self.alphaHorizon - self.config.simulation.angle_from_limb
                    # )
                    self.get_beta_angle(self.acc_reg_alpha_min),
                ]
            )

            # Calculate the pathlength through the atmosphere
            self.losPathLen = self.get_path_length(
                self.sourcebeta[self.volume_mask], self.event_mask(self.sourceNadRad)
            )

            # Calculate the pathlength through the atmosphere
            self.losPathLen = self.get_path_length(
                self.sourcebeta[self.volume_mask], self.event_mask(self.sourceNadRad)
            )

        if u is None:
            raise RuntimeError("Provide a number of time bins")

    def generate_times(self, times) -> np.ndarray:
        """
        Function to generate times within the simulation time period
        """
        if isinstance(times, int):
            times_arr = np.arange(times) / times

        if times is None:
            raise RuntimeError("Provide a number of time bins")

        times_arr *= self.sourceOBSTime  # in s
        times_arr = astropy.time.TimeDelta(times_arr, format="sec")
        times_arr = self.too_source.eventtime + times_arr
        return times_arr

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
        return np.zeros_like(self.beta_rad())

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


# class RegionGeomTargetMonteCarlo:
#    def __init__(self, config):
#        self.config = config
#        self.earth_radius: np.float64 = R_earth.to(u.km).value
#        self.core_alt = (
#            self.earth_radius + self.config.detector.initial_position.altitude
#        )
#        self.detection_mode = self.config.simulation.mode
#        self.sun_moon_cut = self.config.detector.sun_moon.sun_moon_cuts
#
#        self.detLat = config.detector.initial_position.latitude
#        self.detLong = config.detector.initial_position.longitude
#
#        self.sourceOBSTime = self.config.simulation.target.source_obst
#        self.too_source = ToOEvent(self.config)
#
#        self.alphaHorizon = 0.5 * np.pi - np.arccos(self.earth_radius / self.core_alt)


def show_plot(sim_results, simclass, plot):
    plotfs = tuple([geom_beta_tr_hist])
    inputs = tuple([0])
    outputs = ("beta_rad", "theta_rad", "path_len")
    decorators.nss_result_plot_from_file(
        sim_results, simclass, inputs, outputs, plotfs, plot
    )
