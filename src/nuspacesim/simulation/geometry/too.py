import astropy.coordinates
import astropy.time
import astropy.units as u
import numpy as np


class ToOEvent:
    def __init__(self, config):
        self.config = config
        self.sun_alt_cut = self.config.detector.sun_alt_cut
        self.moon_alt_cut = self.config.detector.moon_alt_cut
        self.MoonMinPhaseAngleCut = self.config.detector.MoonMinPhaseAngleCut

        # Detector definitions
        self.detlat = self.config.detector.detlat
        self.detlong = self.config.detector.detlong
        self.detalt = self.config.detector.altitude
        # ToO definitions
        self.sourceRA = self.config.simulation.source_RA
        self.sourceDEC = self.config.simulation.source_DEC
        self.sourceDATE = self.config.simulation.source_date
        self.sourceDateFormat = self.config.simulation.source_date_format
        self.sourceOBSTime = self.config.simulation.source_obst

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
        sun_coord = astropy.coordinates.get_sun(time)
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return sun_coord.transform_to(detframe)

    def get_moon(self, time):
        moon_coord = astropy.coordinates.get_moon(time)
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return moon_coord.transform_to(detframe)

    @staticmethod
    def moon_phase_angle(time: astropy.time.Time) -> float:
        """
        Returns the moon phase angle in rad
        0 -> full moon
        pi -> new moon
        """
        sun = astropy.coordinates.get_sun(time)
        moon = astropy.coordinates.get_moon(time)
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
        sun_alt = self.get_sun(time).alt.degree < self.sun_alt_cut
        moon_alt = self.get_moon(time).alt.degree < self.moon_alt_cut
        moon_phase = self.moon_phase_angle(time).value > self.MoonMinPhaseAngleCut
        moon_cut = np.logical_or(moon_phase, moon_alt)
        return np.logical_and(sun_alt, moon_cut)