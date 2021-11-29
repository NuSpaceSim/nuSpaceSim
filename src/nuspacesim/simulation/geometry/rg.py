import numpy as np


def geo_dmc(u, config):

    radE = config.constants.earth_radius
    detalt = config.detector.altitude
    detra = config.detector.ra_start
    detdec = config.detector.dec_start
    delAlpha = config.simulation.ang_from_limb
    maxsepangle = config.simulation.theta_ch_max
    delAziAng = config.simulation.max_azimuth_angle

    earthRadius = radE
    earthRadiusSqrd = earthRadius * earthRadius

    detAlt = detalt
    radEplusDetAlt = earthRadius + detAlt
    radEplusDetAltSqrd = radEplusDetAlt * radEplusDetAlt

    detRA = np.radians(detra)
    detDec = np.radians(detdec)

    alphaHorizon = 0.5 * np.pi - np.arccos(earthRadius / radEplusDetAlt)
    alphaMin = alphaHorizon - delAlpha
    minChordLen = 2.0 * np.sqrt(
        earthRadiusSqrd - radEplusDetAltSqrd * np.sin(alphaMin) * np.sin(alphaMin)
    )

    minLOSpathLen = radEplusDetAlt * np.cos(alphaMin) - 0.5 * minChordLen
    # minLOSpathLenSqrd = minLOSpathLen * minLOSpathLen
    minLOSpathLenCubed = minLOSpathLen * minLOSpathLen * minLOSpathLen

    # maxThetaS = np.arccos(earthRadius / radEplusDetAlt)
    # minThetaS = np.arccos(
    #     (radEplusDetAltSqrd + earthRadiusSqrd - minLOSpathLenSqrd)
    #     * 1.0
    #     / (2.0 * radEplusDetAlt * earthRadius)
    # )
    # cosOfMaxThetaS = np.cos(maxThetaS)
    # cosOfMinThetaS = np.cos(minThetaS)

    maxLOSpathLen = np.sqrt(radEplusDetAltSqrd - earthRadiusSqrd)
    # maxLOSpathLenSqrd = maxLOSpathLen * maxLOSpathLen
    maxLOSpathLenCubed = maxLOSpathLen * maxLOSpathLen * maxLOSpathLen

    maxThetaTrSubV = maxsepangle
    sinOfMaxThetaTrSubV = np.sin(maxThetaTrSubV)

    maxPhiS = 0.5 * delAziAng
    minPhiS = -0.5 * delAziAng

    normThetaTrSubV = 2.0 / (sinOfMaxThetaTrSubV * sinOfMaxThetaTrSubV)
    normPhiTrSubV = 1.0 / (2.0 * np.pi)
    normPhiS = 1.0 / (maxPhiS - minPhiS)

    bracketForNormThetaS = (
        (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen
        - (1.0 / 3.0) * maxLOSpathLenCubed
        - (radEplusDetAltSqrd - earthRadiusSqrd) * minLOSpathLen
        + (1.0 / 3.0) * minLOSpathLenCubed
    )

    normThetaS = 2.0 * radEplusDetAlt * earthRadiusSqrd / bracketForNormThetaS

    pdfnorm = normThetaTrSubV * normPhiTrSubV * normPhiS * normThetaS
    mcnorm = earthRadiusSqrd / pdfnorm

    # u1, u2, u3, u4 = u
    u1 = u[:, 0]
    u2 = u[:, 1]
    u3 = u[:, 2]
    u4 = u[:, 3]
    N = u.shape[0]

    rthetaTrSubV = np.arcsin(sinOfMaxThetaTrSubV * np.sqrt(u1))
    rcosthetaTrSubV = np.cos(rthetaTrSubV)
    rphiTrSubV = 2.0 * np.pi * u2
    rphiS = (maxPhiS - minPhiS) * u3 + minPhiS

    # Generate theta_s (the colatitude on the surface of the Earth in the
    # detector nadir perspective)

    b = (
        3.0 * (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen
        - maxLOSpathLenCubed
        - 3.0 * (radEplusDetAltSqrd - earthRadiusSqrd) * minLOSpathLen
        + minLOSpathLenCubed
    )
    q = -1.0 * (radEplusDetAltSqrd - earthRadiusSqrd)
    r = (
        -1.5 * (radEplusDetAltSqrd - earthRadiusSqrd) * maxLOSpathLen
        + 0.5 * maxLOSpathLenCubed
        + 0.5 * b * u4
    )

    psi = np.arccos(r / np.sqrt(-1.0 * q * q * q))
    v1 = 2.0 * np.sqrt(-1.0 * q) * np.cos(psi / 3.0)
    v2 = 2.0 * np.sqrt(-1.0 * q) * np.cos((psi + 2.0 * np.pi) / 3.0)
    v3 = 2.0 * np.sqrt(-1.0 * q) * np.cos((psi + 4.0 * np.pi) / 3.0)

    discriminant = q * q * q + r * r

    d_mask = discriminant <= 0
    v1_msk = d_mask & (v1 > 0) & (v1 >= minLOSpathLen) & (v1 <= maxLOSpathLen)
    v2_msk = d_mask & (v2 > 0) & (v2 >= minLOSpathLen) & (v2 <= maxLOSpathLen)
    v3_msk = d_mask & (v3 > 0) & (v3 >= minLOSpathLen) & (v3 <= maxLOSpathLen)

    rlosPathLen = np.zeros_like(v1)
    rlosPathLen[v1_msk] = v1[v1_msk]
    rlosPathLen[v2_msk] = v2[v2_msk]
    rlosPathLen[v3_msk] = v3[v3_msk]

    rlosPathLen[~d_mask] = np.sum(
        np.cbrt(
            r[~d_mask] + np.multiply.outer([1, -1], np.sqrt(discriminant[~d_mask]))
        ),
        axis=0,
    )

    rvsqrd = rlosPathLen * rlosPathLen
    rcosthetaS = (radEplusDetAltSqrd + earthRadiusSqrd - rvsqrd) / (
        2.0 * earthRadius * radEplusDetAlt
    )
    rthetaS = np.arccos(rcosthetaS)

    rcosthetaNSubV = (radEplusDetAltSqrd - earthRadiusSqrd - rvsqrd) / (
        2.0 * earthRadius * rlosPathLen
    )

    rthetaNSubV = np.arccos(rcosthetaNSubV)

    rcosthetaTrSubN = np.cos(rthetaTrSubV) * rcosthetaNSubV - np.sin(
        rthetaTrSubV
    ) * np.sin(rthetaNSubV) * np.cos(rphiTrSubV)

    rthetaTrSubN = np.arccos(rcosthetaTrSubN)

    # if (rcosthetaTrSubN < 0.0)
    #   geoKeep = False; // Trajectories going into the ground

    rbetaTrSubN = np.degrees(0.5 * np.pi - rthetaTrSubN)

    rsindecS = np.sin(detDec) * rcosthetaS - np.cos(detDec) * np.sin(rthetaS) * np.cos(
        rphiS
    )
    rdecS = np.degrees(np.arcsin(rsindecS))

    rxS = (
        np.sin(detDec) * np.cos(detRA) * np.sin(rthetaS) * np.cos(rphiS)
        - np.sin(detRA) * np.sin(rthetaS) * np.sin(rphiS)
        + np.cos(detDec) * np.cos(detRA) * np.cos(rthetaS)
    )

    ryS = (
        np.sin(detDec) * np.sin(detRA) * np.sin(rthetaS) * np.cos(rphiS)
        + np.cos(detRA) * np.sin(rthetaS) * np.sin(rphiS)
        + np.cos(detDec) * np.sin(detRA) * np.cos(rthetaS)
    )

    rraS = np.degrees(np.arctan2(ryS, rxS))
    rraS[rraS < 0.0] += 360.0

    good_event = rcosthetaTrSubN >= 0

    good = rcosthetaTrSubV >= np.cos(np.radians(1.5))
    geom_factor = np.sum(
        rcosthetaTrSubN / (rcosthetaNSubV * rcosthetaTrSubV), where=(good_event & good)
    )
    geom_factor *= np.reciprocal(N) * mcnorm

    return (
        geom_factor,
        good_event,
        rthetaS,
        rphiS,
        rraS,
        rdecS,
        rthetaTrSubV,
        rcosthetaTrSubV,
        rphiTrSubV,
        rthetaTrSubN,
        rcosthetaTrSubN,
        rbetaTrSubN,
        rlosPathLen,
        rthetaNSubV,
        rcosthetaNSubV,
    )
