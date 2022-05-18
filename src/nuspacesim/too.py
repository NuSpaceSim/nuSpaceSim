#!/usr/bin/python

import numpy as np

import astropy.coordinates
import astropy.time
import astropy.units as u


class tooevent:
    def __init__(
        self,
        RA,
        DEC,
        eventday,
        eventtime,
        detlat,
        detlong,
        detalt,
        obstime,
        units="deg",
    ):

        if units == "rad":
            self.RA = np.rad2deg(RA)
            self.DEC = np.rad2deg(DEC)
            self.detlat = np.rad2deg(detlat)
            self.detlong = np.rad2deg(detlong)

        self.eventtime = astropy.time.Time(
            eventday, format="mjd"
        ) + astropy.time.TimeDelta(eventtime, format="sec")

        self.eventcoords = astropy.coordinates.SkyCoord(
            ra=RA * u.degree, dec=DEC * u.degree, frame="icrs"
        )

        self.detcords = astropy.coordinates.EarthLocation(
            lat=detlat, lon=detlong, height=detalt
        )

        self.obstime = obstime

    def localcoords(self, time):
        self.detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)

        return self.eventcoords.transform_to(self.detframe)

    def event_time(self):
        return self.eventtime


# WANAKALAT = -35.20666735 * u.deg
# WANAKALONG = -69.315833 * u.deg
# WANAKAHEIGHT = 30000 * u.m

# RA = 0
# DEC = 0
# day = 60035.5  # 04/01/2023
# time = 3600

# event = tooevent(RA, DEC, day, time, WANAKALAT, WANAKALONG, WANAKAHEIGHT)
# time = event.eventtime

# print(event.localcoords(time))
