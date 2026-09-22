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
import astropy.units as u
import numpy as np
from scipy.interpolate import CubicSpline


class ToOEvent:
    def __init__(self, config):
        self.config = config
        self.sun_alt_cut = self.config.detector.sun_moon.sun_alt_cut
        self.moon_alt_cut = self.config.detector.sun_moon.moon_alt_cut
        self.MoonMinPhaseAngleCut = (
            self.config.detector.sun_moon.moon_min_phase_angle_cut
        )

        # Detector definitions -- will need to be updated for moving platforms
        self.detlat = self.config.detector.initial_position.latitude
        self.detlong = self.config.detector.initial_position.longitude
        self.detalt = self.config.detector.initial_position.altitude
        # Target(ToO) definitions
        self.sourceRA = self.config.simulation.target.source_RA
        self.sourceDEC = self.config.simulation.target.source_DEC
        self.sourceDATE = self.config.simulation.target.source_date
        self.sourceDateFormat = self.config.simulation.target.source_date_format
        self.sourceOBSTime = self.config.simulation.target.source_obst
        self.ephemeris_step = self.config.simulation.target.ephemeris_step

        self.eventtime = astropy.time.Time(
            self.sourceDATE, format=self.sourceDateFormat, scale="utc"
        )  # note make scale variable

        self.eventcoords = astropy.coordinates.SkyCoord(
            ra=self.sourceRA * u.rad,
            dec=self.sourceDEC * u.rad,
            frame="icrs",
        )  # note make frame variable

        # Note: these are geodetic coordinates
        self.detcoords = astropy.coordinates.EarthLocation(
            lat=self.detlat * u.rad,
            lon=self.detlong * u.rad,
            height=self.detalt * 1000 * u.m,
        )

    def detframe(self, time):
        return astropy.coordinates.AltAz(obstime=time, location=self.detcoords)

    def _ephemeris_grid(self, time):
        """Coarse time grid spanning ``time`` for interpolated sky positions.

        Returns ``(grid_time, grid_x, x)`` with ``x`` the query abscissae in
        seconds, or ``None`` when exact evaluation is requested or is no more
        work than the grid itself.
        """
        step = self.ephemeris_step
        if step <= 0 or time.isscalar:
            return None
        x = time.utc.unix
        x0, x1 = x.min(), x.max()
        # Pad one step each side so every query is interior to the spline.
        n_grid = int(np.ceil((x1 - x0) / step)) + 3
        if n_grid >= x.size:
            return None
        grid_x = (x0 - step) + step * np.arange(n_grid)
        grid_time = astropy.time.Time(grid_x, format="unix", scale="utc")
        return grid_time, grid_x, x

    def localcoords(self, time):
        grid = self._ephemeris_grid(time)
        if grid is None:
            return self.eventcoords.transform_to(self.detframe(time))
        grid_time, grid_x, x = grid
        exact = self.eventcoords.transform_to(self.detframe(grid_time))
        # Interpolate the unit vector, not (alt, az): azimuth wraps at 2pi.
        cos_alt = np.cos(exact.alt.rad)
        vec = np.stack(
            [
                cos_alt * np.cos(exact.az.rad),
                cos_alt * np.sin(exact.az.rad),
                np.sin(exact.alt.rad),
            ]
        )
        vx, vy, vz = CubicSpline(grid_x, vec, axis=1)(x)
        alt = np.arctan2(vz, np.hypot(vx, vy))
        az = np.arctan2(vy, vx) % (2.0 * np.pi)
        return astropy.coordinates.AltAz(
            alt=alt * u.rad, az=az * u.rad, obstime=time, location=self.detcoords
        )

    def get_sun(self, time):
        sun_coord = astropy.coordinates.get_body("sun", time)
        return sun_coord.transform_to(self.detframe(time))

    def get_moon(self, time):
        moon_coord = astropy.coordinates.get_body("moon", time)
        return moon_coord.transform_to(self.detframe(time))

    @staticmethod
    def phase_angle_from_bodies(sun, moon):
        """
        Moon phase angle in rad from geocentric sun and moon coordinates
        0 -> full moon
        pi -> new moon
        """
        elongation = sun.separation(moon)
        return np.arctan2(
            sun.distance * np.sin(elongation),
            moon.distance - sun.distance * np.cos(elongation),
        )

    @classmethod
    def moon_phase_angle(cls, time: astropy.time.Time) -> float:
        """
        Returns the moon phase angle in rad
        0 -> full moon
        pi -> new moon
        """
        sun = astropy.coordinates.get_body("sun", time)
        moon = astropy.coordinates.get_body("moon", time)
        return cls.phase_angle_from_bodies(sun, moon)

    def _sun_moon_state(self, time):
        """Exact (sun altitude, moon altitude, moon phase angle) in rad."""
        # One ephemeris lookup per body and one AltAz frame serve the altitude
        # cuts and the phase angle; the lookups dominate ToO-mode runtime.
        sun = astropy.coordinates.get_body("sun", time)
        moon = astropy.coordinates.get_body("moon", time)
        detframe = self.detframe(time)
        return np.stack(
            [
                sun.transform_to(detframe).alt.rad,
                moon.transform_to(detframe).alt.rad,
                self.phase_angle_from_bodies(sun, moon).value,
            ]
        )

    def sun_moon_state(self, time):
        """(sun altitude, moon altitude, moon phase angle) in rad at ``time``.

        Evaluated on the ephemeris grid and cubic-interpolated when that is
        cheaper than evaluating every requested time.
        """
        grid = self._ephemeris_grid(time)
        if grid is None:
            return self._sun_moon_state(time)
        grid_time, grid_x, x = grid
        return CubicSpline(grid_x, self._sun_moon_state(grid_time), axis=1)(x)

    def sun_moon_cut(self, time: astropy.time.Time) -> bool:
        """
        Function to calculate the time during which sun and moon allow observation
        True -> observation possible
        False -> no observation posible
        """
        sun_alt, moon_alt, moon_phase = self.sun_moon_state(time)
        moon_cut = np.logical_or(
            moon_phase > self.MoonMinPhaseAngleCut, moon_alt < self.moon_alt_cut
        )
        return np.logical_and(sun_alt < self.sun_alt_cut, moon_cut)
