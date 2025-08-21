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


class ToOEvent:
    def __init__(self, config):
        self.config = config
        self.sun_alt_cut = self.config.detector.sun_moon.sun_alt_cut
        self.moon_alt_cut = self.config.detector.sun_moon.moon_alt_cut
        self.MoonMinPhaseAngleCut = (
            self.config.detector.sun_moon.moon_min_phase_angle_cut
        )

        # Detector definitions
        self.detlat = self.config.detector.initial_position.latitude
        self.detlong = self.config.detector.initial_position.longitude
        self.detalt = self.config.detector.initial_position.altitude
        # Target(ToO) definitions
        self.sourceRA = self.config.simulation.target.source_RA
        self.sourceDEC = self.config.simulation.target.source_DEC
        self.sourceDATE = self.config.simulation.target.source_date
        self.sourceDateFormat = self.config.simulation.target.source_date_format
        self.sourceOBSTime = self.config.simulation.target.source_obst

        self.eventtime = astropy.time.Time(
            self.sourceDATE, format=self.sourceDateFormat, scale="utc"
        )  # note make scale variable

        self.eventcoords = astropy.coordinates.SkyCoord(
            ra=self.sourceRA * u.rad,
            dec=self.sourceDEC * u.rad,
            frame="icrs",
        )  # note make frame variable

        # Note: these are geodetic coordinates
        self.detcords = astropy.coordinates.EarthLocation(
            lat=self.detlat * u.rad,
            lon=self.detlong * u.rad,
            height=self.detalt * 1000 * u.m,
        )

    def localcoords(self, time):
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return self.eventcoords.transform_to(detframe)

    def get_sun(self, time):
        sun_coord = astropy.coordinates.get_body("sun", time)
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return sun_coord.transform_to(detframe)

    def get_moon(self, time):
        moon_coord = astropy.coordinates.get_body("moon", time)
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return moon_coord.transform_to(detframe)

    @staticmethod
    def moon_phase_angle(time: astropy.time.Time) -> float:
        """
        Returns the moon phase angle in rad
        0 -> full moon
        pi -> new moon
        """
        sun = astropy.coordinates.get_body("sun", time)
        moon = astropy.coordinates.get_body("moon", time)
        elongation = sun.separation(moon)
        return np.arctan2(
            sun.distance * np.sin(elongation),
            moon.distance - sun.distance * np.cos(elongation),
        )

    def sun_moon_cut(self, time: astropy.time.Time) -> bool:
        """
        Function to calculate the time during which sun and moon allow observation
        True -> observation possible
        False -> no observation posible
        """
        sun_alt = self.get_sun(time).alt.rad < self.sun_alt_cut
        moon_alt = self.get_moon(time).alt.rad < self.moon_alt_cut
        moon_phase = self.moon_phase_angle(time).value > self.MoonMinPhaseAngleCut
        moon_cut = np.logical_or(moon_phase, moon_alt)

        return np.logical_and(sun_alt, moon_cut)
