import numpy as np


class Geo:
    def __init__(self, config):
        self.config = config
        radE = config.constants.earth_radius
        detalt = config.detector.altitude
        detra = config.detector.ra_start
        detdec = config.detector.dec_start
        delAlpha = config.simulation.ang_from_limb
        maxsepangle = config.simulation.theta_ch_max
        delAziAng = config.simulation.max_azimuth_angle

        self.earth_radius = radE
        self.earth_rad_2 = self.earth_radius * self.earth_radius

        self.detAlt = detalt
        self.radEplusDetAlt = self.earth_radius + self.detAlt
        self.radEplusDetAltSqrd = self.radEplusDetAlt * self.radEplusDetAlt

        self.detRA = np.radians(detra)
        self.detDec = np.radians(detdec)

        alphaHorizon = 0.5 * np.pi - np.arccos(self.earth_radius / self.radEplusDetAlt)
        alphaMin = alphaHorizon - delAlpha
        minChordLen = 2.0 * np.sqrt(
            self.earth_rad_2
            - self.radEplusDetAltSqrd * np.sin(alphaMin) * np.sin(alphaMin)
        )

        self.minLOSpathLen = self.radEplusDetAlt * np.cos(alphaMin) - 0.5 * minChordLen

        self.maxThetaS = np.arccos(self.earth_radius / self.radEplusDetAlt)
        self.minThetaS = np.arccos(
            (self.radEplusDetAltSqrd + self.earth_rad_2 - self.minLOSpathLen ** 2)
            * 1.0
            / (2.0 * self.radEplusDetAlt * self.earth_radius)
        )

        self.maxLOSpathLen = np.sqrt(self.radEplusDetAltSqrd - self.earth_rad_2)
        self.maxLOSpathLenCubed = self.maxLOSpathLen ** 3

        self.maxThetaTrSubV = maxsepangle
        self.sinOfMaxThetaTrSubV = np.sin(self.maxThetaTrSubV)

        self.maxPhiS = 0.5 * delAziAng
        self.minPhiS = -0.5 * delAziAng

        self.normThetaTrSubV = 2.0 / (
            self.sinOfMaxThetaTrSubV * self.sinOfMaxThetaTrSubV
        )
        self.normPhiTrSubV = 1.0 / (2.0 * np.pi)
        self.normPhiS = 1.0 / (self.maxPhiS - self.minPhiS)

        bracketForNormThetaS = (
            (self.radEplusDetAltSqrd - self.earth_rad_2) * self.maxLOSpathLen
            - (1.0 / 3.0) * self.maxLOSpathLen ** 3
            - (self.radEplusDetAltSqrd - self.earth_rad_2) * self.minLOSpathLen
            + (1.0 / 3.0) * self.minLOSpathLen ** 3
        )

        normThetaS = 2.0 * self.radEplusDetAlt * self.earth_rad_2 / bracketForNormThetaS

        pdfnorm = self.normThetaTrSubV * self.normPhiTrSubV * self.normPhiS * normThetaS
        self.mcnorm = self.earth_rad_2 / pdfnorm

    def throw(self, u):
        """Throw N events with 4 * u random numbers"""

        # u1, u2, u3, u4 = u
        u1 = u[:, 0]
        u2 = u[:, 1]
        u3 = u[:, 2]
        u4 = u[:, 3]
        N = u.shape[0]

        self.thetaTrSubV = np.arcsin(self.sinOfMaxThetaTrSubV * np.sqrt(u1))
        costhetaTrSubV = np.cos(self.thetaTrSubV)
        phiTrSubV = 2.0 * np.pi * u2
        phiS = (self.maxPhiS - self.minPhiS) * u3 + self.minPhiS

        # Generate theta_s (the colatitude on the surface of the Earth in the
        # detector nadir perspective)

        b = (
            3.0 * (self.radEplusDetAltSqrd - self.earth_rad_2) * self.maxLOSpathLen
            - self.maxLOSpathLen ** 3
            - 3.0 * (self.radEplusDetAltSqrd - self.earth_rad_2) * self.minLOSpathLen
            + self.minLOSpathLen ** 3
        )
        q = -1.0 * (self.radEplusDetAltSqrd - self.earth_rad_2)
        r = (
            -1.5 * (self.radEplusDetAltSqrd - self.earth_rad_2) * self.maxLOSpathLen
            + 0.5 * self.maxLOSpathLenCubed
            + 0.5 * b * u4
        )

        psi = np.arccos(r / np.sqrt(-1.0 * q * q * q))
        v1 = 2.0 * np.sqrt(-1.0 * q) * np.cos(psi / 3.0)
        v2 = 2.0 * np.sqrt(-1.0 * q) * np.cos((psi + 2.0 * np.pi) / 3.0)
        v3 = 2.0 * np.sqrt(-1.0 * q) * np.cos((psi + 4.0 * np.pi) / 3.0)

        dscr = q * q * q + r * r

        d_mask = dscr <= 0
        v1_msk = (
            d_mask & (v1 > 0) & (v1 >= self.minLOSpathLen) & (v1 <= self.maxLOSpathLen)
        )
        v2_msk = (
            d_mask & (v2 > 0) & (v2 >= self.minLOSpathLen) & (v2 <= self.maxLOSpathLen)
        )
        v3_msk = (
            d_mask & (v3 > 0) & (v3 >= self.minLOSpathLen) & (v3 <= self.maxLOSpathLen)
        )

        losPathLen = np.zeros_like(v1)
        losPathLen[v1_msk] = v1[v1_msk]
        losPathLen[v2_msk] = v2[v2_msk]
        losPathLen[v3_msk] = v3[v3_msk]

        losPathLen[~d_mask] = np.sum(
            np.cbrt(r[~d_mask] + np.multiply.outer([1, -1], np.sqrt(dscr[~d_mask]))),
            axis=0,
        )

        rvsqrd = losPathLen * losPathLen
        costhetaS = (self.radEplusDetAltSqrd + self.earth_rad_2 - rvsqrd) / (
            2.0 * self.earth_radius * self.radEplusDetAlt
        )
        thetaS = np.arccos(costhetaS)

        costhetaNSubV = (self.radEplusDetAltSqrd - self.earth_rad_2 - rvsqrd) / (
            2.0 * self.earth_radius * losPathLen
        )

        thetaNSubV = np.arccos(costhetaNSubV)

        self.costhetaTrSubN = np.cos(self.thetaTrSubV) * costhetaNSubV - np.sin(
            self.thetaTrSubV
        ) * np.sin(thetaNSubV) * np.cos(phiTrSubV)

        thetaTrSubN = np.arccos(self.costhetaTrSubN)

        # if (rcosthetaTrSubN < 0.0)
        #   geoKeep = False; // Trajectories going into the ground

        self.betaTrSubN = np.degrees(0.5 * np.pi - thetaTrSubN)
        rsindecS = np.sin(self.config.detector.dec_start) * costhetaS - np.cos(
            self.config.detector.dec_start
        ) * np.sin(thetaS) * np.cos(phiS)
        self.rdecS = np.degrees(np.arcsin(rsindecS))

        self.rxS = (
            np.sin(self.config.detector.dec_start)
            * np.cos(self.config.detector.ra_start)
            * np.sin(thetaS)
            * np.cos(phiS)
            - np.sin(self.config.detector.ra_start) * np.sin(thetaS) * np.sin(phiS)
            + np.cos(self.config.detector.dec_start)
            * np.cos(self.config.detector.ra_start)
            * np.cos(thetaS)
        )

        self.ryS = (
            np.sin(self.config.detector.dec_start)
            * np.sin(self.config.detector.ra_start)
            * np.sin(thetaS)
            * np.cos(phiS)
            + np.cos(self.config.detector.ra_start) * np.sin(thetaS) * np.sin(phiS)
            + np.cos(self.config.detector.dec_start)
            * np.sin(self.config.detector.ra_start)
            * np.cos(thetaS)
        )

        self.rraS = np.degrees(np.arctan2(self.ryS, self.rxS))
        self.rraS[self.rraS < 0.0] += 360.0

        self.event_mask = self.costhetaTrSubN >= 0

        good = costhetaTrSubV >= np.cos(np.radians(1.5))
        self.geom_factor = np.sum(
            self.costhetaTrSubN / (costhetaNSubV * costhetaTrSubV),
            where=(self.event_mask & good),
        )
        self.geom_factor *= np.reciprocal(N) * self.mcnorm
