"""
Atmospheric properties for use in simulation and geometry models.
"""

from nuSpaceSim.EAScherGen.zsteps import zsteps as cppzsteps
import numpy as np
from nuSpaceSim import constants as c


def theta_view(betaE, detector_height, earth_radius=c.earth_radius):
    """
    Compute theta view from initial betas
    """

    ThetProp = np.radians(betaE)
    ThetView = earth_radius / (earth_radius + detector_height)
    ThetView *= np.cos(ThetProp)
    ThetView = np.arcsin(ThetView)
    return ThetView


def zsteps(
    z,
    detector_height=c.low_earth_orbit,
    atmosphere_end=c.atmosphere_end,
    sinThetView=None,
    earth_radius=c.earth_radius,
    dL=0.1,
    pi=np.pi,
):
    """
    Compute all mid-bin z steps and corresponding delz values
    """

    if sinThetView is None:
        sinThetView = np.sin(theta_view(z, detector_height, earth_radius))

    return cppzsteps(
        z, sinThetView, earth_radius, atmosphere_end, detector_height, dL, pi
    )


# FROM EAScherGen/subCphotAng.f line 409
# c   set z to measure rho at center of bin
# c 20-Mar-18 need to think about this ...
#        z=Zold+delz/2.
#        Zold=z

#        zsave(iz)=z

# c   add in the other 1/2 to record top of bin
#        Zold=z+delz/2.

# c     Calculate Grammage
#        if (z.lt.11.) then
#         a1=44.34
#         a2=-11.861
#         a3=1./0.19
#         a4=a3-1.
#         X=((z-a1)/a2)**a3
#          rho=-1.e-5*a3/a2*((z-a1)/a2)**a4
#        else if (z.lt.25.) then
#         a1=45.5
#         a2=-6.34
#         X=exp((z-a1)/a2)
#         rho=-1.e-5/a2*exp((z-a1)/a2)
#        else
#         a1=13.841
#         a2=28.920
#         a3=3.344
#         X=exp(a1-sqrt(a2+a3*z))
#         rho=0.5e-5*a3/sqrt(a2+a3*z)*
#      +  exp(a1-sqrt(a2+a3*z))
#        endif

#        delgram(iz)=rho*dL*1.e5
#        GramTsum=GramTsum+delgram(iz)
#        gramsum(iz)=GramTsum
#        gramz(iz)=X


def grammage(z, lower_bound=11, upper_bound=25):
    """
    # c     Calculate Grammage from altitudes (z) in km.
    """

    # Compute differently based on region bounds
    mask1 = z < lower_bound
    mask2 = np.logical_and(z >= lower_bound, z < upper_bound)
    mask3 = z >= upper_bound

    # grammage
    X = np.empty_like(z)
    X[mask1] = np.power(((z[mask1] - 44.34) / -11.861), (1 / 0.19))
    X[mask2] = np.exp(np.divide(z[mask2] - 45.5, -6.34))
    X[mask3] = np.exp(np.subtract(13.841, np.sqrt(28.920 + 3.344 * z[mask3])))

    # grammage derivative
    rho = np.empty_like(z)
    rho[mask1] = np.power(
        (-1.0e-5) * (1 / 0.19) / (-11.861) * ((z[mask1] - 44.34) / -11.861),
        (1.0 / 0.19) - 1.0,
    )
    rho[mask2] = np.multiply(-1e-5 * np.reciprocal(-6.34), X[mask2])
    rho[mask3] = np.multiply(
        np.divide(0.5e-5 * 3.344, np.sqrt(28.920 + 3.344 * z[mask3])), X[mask3]
    )

    return X, rho


def slant_depth(
    decay_altitude,
    detector_height=c.low_earth_orbit,
    atmosphere_end=c.atmosphere_end,
    sinThetView=None,
    earth_radius=c.earth_radius,
    dL=0.1,
):
    """
    slant depth from grammage: scalar decay_altitude inputs only.

    Given a starting decay_altitude, detector_height, and atmosphere_end, compute the
    grammage at each altitude step, cumulative integral of the derivative of grammage
    at each step, and the step points were computed.
    """

    if sinThetView is None:
        sinThetView = np.sin(theta_view(decay_altitude, detector_height, earth_radius))

    zs, _ = zsteps(
        decay_altitude, detector_height, atmosphere_end, sinThetView, earth_radius, dL
    )

    gramz, rhos = grammage(zs)

    delgram_vals = rhos * dL * (1e5)
    gramsum = np.cumsum(delgram_vals)
    # delgram = np.cumsum(delgram_vals[::-1])[::-1]

    return (
        gramz,
        gramsum,
        zs,
    )  # delgram
