#!/usr/bin/python

import numpy as np

import astropy.coordinates
import astropy.time
import astropy.units


class tooevent:
    def __init__(self, config):
        self.config = config
        self.sun_alt_cut = self.config.detector.sun_alt_cut
        self.moon_alt_cut = self.config.detector.moon_alt_cut
        self.MoonMinPhaseAngleCut = self.config.detector.MoonMinPhaseAngleCut

        # Detector definitions
        self.detlat = self.config.detector.ra_start
        self.detlong = self.config.detector.dec_start
        self.detalt = self.config.detector.altitude

        # ToO definitions
        self.sourceRA = self.config.simulation.source_RA
        self.sourceDEC = self.config.simulation.source_DEC
        self.sourceDATE = self.config.simulation.source_date
        self.sourceDateFormat = self.config.simulation.source_date_format
        self.sourceOBSTime = self.config.simulation.source_obst

        self.eventtime = astropy.time.Time(
            self.sourceDATE, format=self.sourceDateFormat, scale="utc"
        )

        self.eventcoords = astropy.coordinates.SkyCoord(
            ra=self.sourceRA * astropy.units.degree,
            dec=self.sourceDEC * astropy.units.degree,
            frame="icrs",
        )

        self.detcords = astropy.coordinates.EarthLocation(
            lat=self.detlat, lon=self.detlong, height=self.detalt
        )

        self.moon_alt_cutoff = 0
        self.sun_alt_cutoff = -13.5
        self.moon_phase_cut = 150

    def localcoords(self, time):
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return self.eventcoords.transform_to(detframe)

    def get_sun(self, time):
        sun_coord = astropy.coordinates.get_sun(time)
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return sun_coord.transform_to(detframe)

    def get_moon(self, time):
        moon_coord = astropy.coordinates.get_moon(time)
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return moon_coord.transform_to(detframe)

    @staticmethod
    def moon_phase_angle(time):
        sun = astropy.coordinates.get_sun(time)
        moon = astropy.coordinates.get_moon(time)
        elongation = sun.separation(moon)
        return np.arctan2(
            sun.distance * np.sin(elongation),
            moon.distance - sun.distance * np.cos(elongation),
        )

    def sun_moon_cut(self, time):
        sun_alt = self.get_sun(time).alt.degree < self.sun_alt_cut
        moon_alt = self.get_moon(time).alt.degree < self.moon_alt_cut
        moon_phase = (
            np.rad2deg(self.moon_phase_angle(time).value) > self.MoonMinPhaseAngleCut
        )
        moon_cut = np.logical_or(moon_phase, moon_alt)
        return np.logical_and(sun_alt, moon_cut)
