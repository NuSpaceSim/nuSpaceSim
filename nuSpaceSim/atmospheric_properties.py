"""
Atmospheric properties for use in simulation and geometry models.
"""

from nuSpaceSim.EAScherGen.zsteps import zsteps as cppzsteps
import numpy as np
from nuSpaceSim import constants as c

OzZeta = lambda: np.array(
    [5.35, 10.2, 14.75, 19.15, 23.55, 28.1, 32.8, 37.7, 42.85, 48.25, 100.0]
)

OzDepth = lambda: np.array(
    [15.0, 9.0, 10.0, 31.0, 71.0, 87.2, 57.0, 29.4, 10.9, 3.2, 1.3]
)

OzDsum = lambda: np.array(
    [310.0, 301.0, 291.0, 260.0, 189.0, 101.8, 44.8, 15.4, 4.5, 1.3, 0.1]
)


def theta_view(betaE, zmax=c.low_earth_orbit, earth_radius=c.earth_radius):
    """
    Compute theta view from initial betas
    """

    ThetProp = np.radians(betaE)
    ThetView = earth_radius / (earth_radius + zmax)
    ThetView *= np.cos(ThetProp)
    ThetView = np.arcsin(ThetView)
    return ThetView


def grammage(z):
    """
    # c     Calculate Grammage from altitudes (z) in km.
    """

    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25

    X = np.empty_like(z)
    X[mask1] = np.power(((z[mask1] - 44.34) / -11.861), (1 / 0.19))
    X[mask2] = np.exp(np.divide(z[mask2] - 45.5, -6.34))
    X[mask3] = np.exp(np.subtract(13.841, np.sqrt(28.920 + 3.344 * z[mask3])))

    rho = np.empty_like(z)
    rho[mask1] = (
        (-1.0e-5)
        * (1 / 0.19)
        / (-11.861)
        * ((z[mask1] - 44.34) / -11.861) ** ((1.0 / 0.19) - 1.0)
    )
    rho[mask2] = np.multiply(-1e-5 * np.reciprocal(-6.34), X[mask2])
    rho[mask3] = np.multiply(
        np.divide(0.5e-5 * 3.344, np.sqrt(28.920 + 3.344 * z[mask3])), X[mask3]
    )
    return X, rho


def ozone_losses(z, OzZeta=OzZeta(), OzDepth=OzDepth(), OzDsum=OzDsum()):
    """
    Calculate ozone losses from altitudes (z) in km.
    """

    mask1 = z < 5.35
    mask2 = z >= 100
    mask3 = np.logical_and(~mask1, ~mask2)

    TotZon = np.empty_like(z)
    TotZon[mask1] = (310) + (((5.35) - z[mask1]) / (5.35)) * (15)
    TotZon[mask2] = 0.1

    idxs = np.searchsorted(OzZeta, z[mask3])

    TotZon[mask3] = (
        OzDsum[idxs]
        + ((OzZeta[idxs] - z[mask3]) / (OzZeta[idxs] - OzZeta[idxs - 1]))
        * OzDepth[idxs]
    )

    return TotZon


def theta_prop(
    z, zmax=c.low_earth_orbit, sin_theta_view=None, earth_radius=c.earth_radius
):
    """
    theta propagation.
    """

    if sin_theta_view is None:
        sin_theta_view = np.sin(theta_view(z, zmax, earth_radius))

    tp = (earth_radius + zmax) / (earth_radius + z)
    return np.arccos(sin_theta_view * tp)


def zsteps(
    z,
    zmax=c.low_earth_orbit,
    zMaxZ=c.zMaxZ,
    sinThetView=None,
    earth_radius=c.earth_radius,
    dL=0.1,
    pi=np.pi,
):
    """
    Compute all mid-bin z steps and corresponding delz values
    """

    if sinThetView is None:
        sinThetView = np.sin(theta_view(z, zmax, earth_radius))

    return cppzsteps(z, sinThetView, earth_radius, zMaxZ, zmax, dL, pi)


def slant_depth(
    alt,
    zmax=c.low_earth_orbit,
    zMaxZ=c.zMaxZ,
    sinThetView=None,
    earth_radius=c.earth_radius,
    dL=0.1,
):
    """
    Determine Rayleigh and Ozone slant depth.
    """

    if sinThetView is None:
        sinThetView = np.sin(theta_view(alt, zmax, earth_radius))

    zsave, delzs = zsteps(alt, zmax, zMaxZ, sinThetView, earth_radius, dL)
    gramz, rhos = grammage(zsave)

    delgram_vals = rhos * dL * (1e5)
    gramsum = np.cumsum(delgram_vals)
    delgram = np.cumsum(delgram_vals[::-1])[::-1]

    TotZons = ozone_losses(np.insert(zsave, 0, alt))
    ZonZ_vals = (TotZons[:-1] - TotZons[1:]) / delzs * dL
    ZonZ = np.cumsum(ZonZ_vals[::-1])[::-1]

    ThetPrpA = theta_prop(zsave, zmax, sinThetView)

    return zsave, delgram, gramsum, gramz, ZonZ, ThetPrpA
