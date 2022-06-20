#!/usr/bin/python

import numpy as np

import astropy.coordinates
import astropy.time
import astropy.units


class tooevent:
    def __init__(
        self,
        RA,
        DEC,
        eventday,
        detlat,
        detlong,
        detalt,
        obstime,
        units="rad",
    ):

        if units == "rad":
            self.RA = np.rad2deg(RA)
            self.DEC = np.rad2deg(DEC)
            self.detlat = np.rad2deg(detlat)
            self.detlong = np.rad2deg(detlong)

        self.eventtime = astropy.time.Time(
            eventday[0], format=eventday[1], scale="utc"
        )

        self.eventcoords = astropy.coordinates.SkyCoord(
            ra=RA * astropy.units.degree,
            dec=DEC * astropy.units.degree,
            frame="icrs"
        )

        self.detcords = astropy.coordinates.EarthLocation(
            lat=detlat,
            lon=detlong,
            height=detalt
        )

        self.obstime = obstime

    def localcoords(self, time):
        detframe = astropy.coordinates.AltAz(obstime=time, location=self.detcords)
        return self.eventcoords.transform_to(detframe)
