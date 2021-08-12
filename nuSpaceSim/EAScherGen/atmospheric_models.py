import numpy as np
import scipy.integrate
from nuSpaceSim import constants as c
from typing import Iterable

__all__ = ["rho", "slant_depth", "slant_depth_integrand", "slant_depth_steps"]

def rho(z):
    """
    Density parameterized from altitude (z) values

    Computation from equation (2) in https://arxiv.org/pdf/2011.09869.pdf
    """
    z = z if isinstance(z, Iterable) else np.array([z])

    rval = np.empty_like(z)

    mask = z <= 100

    b, c = param_b_c(z[mask])
    rval[mask] = (b / c) * np.exp(-z / c)

    b, c = param_b_c(z[~mask])
    rval[~mask] = b / c

    return rval


def slant_depth_integrand(
    z,
    theta_tr: float,
    earth_radius: float = c.earth_radius,
):
    """
    Integrand for computing slant_depth from input altitude z.
    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    """
    return rho(z) * (
        (z + earth_radius) / np.sqrt(earth_radius ** 2) * np.cos(theta_tr) ** 2
        + z ** 2
        + 2 * z * earth_radius
    )


def slant_depth(
    z_lo: float,
    z_hi: float,
    theta_tr: float,
    earth_radius: float = c.earth_radius,
    integrand_f=None,
):
    """
    Slant-depth integral

    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    using gaussian quadriture, and along a full length using the cumulative_trapezoid
    rule.

          z_hi
         /
    X =  |  integrand_f(z, theta_tr, earth_radius) dz
         /
         z_lo

    Params
    ======

        z_lo: (float) starting altitude for slant depth track.

        z_hi: (float) stopping altitude for slant depth track.

        theta_tr: (float) trajectory angle of track to observer.

        earth_radius: (float) radius of a spherical earth.
        Default from nuSpaceSim.constants

        integrand_f: (real valued function) the integrand for slant_depth. If None,
        Default of `slant_depth_integrand()` is used.

    Returns
    =======

        x_sd: (float) X [slant depth].

        err: (float) numerical error.

    """

    if integrand_f is None:
        integrand_f = lambda x: slant_depth_integrand(x, theta_tr, earth_radius)

    x_sd, err = scipy.integrate.quad(integrand_f, z_lo, z_hi)

    return x_sd, err


def slant_depth_steps(
    z_lo: float,
    z_hi: float,
    theta_tr: float,
    earth_radius: float = c.earth_radius,
    dz: float = 0.01,
    integrand_f=None,
):
    """
    Slant-depth integral approximated along path.

    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    along a full length using the cumulative_trapezoid rule.

          z_hi
         /
    X =  |  integrand_f(z, theta_tr, earth_radius) dz
         /
         z_lo

    Params
    ======

        z_lo: (float) starting altitude for slant depth track.

        z_hi: (float) stopping altitude for slant depth track.

        theta_tr: (float) trajectory angle of track to observer.

        earth_radius: (float) radius of a spherical earth.
        Default from nuSpaceSim.constants

        dz: (float) static step size for sampling points in range [z_lo, z_hi]

        integrand_f: (real valued function) the integrand for slant_depth. If None,
        Default of `slant_depth_integrand()` is used.

    Returns
    =======

        xs: (float) slant depth at each altitude along track.

        zs: (float) altitudes at which slant_depth was evaluated.

    """

    if integrand_f is None:
        integrand_f = lambda x: slant_depth_integrand(x, theta_tr, earth_radius)

    zs = np.arange(z_lo, z_hi, dz)
    xs = scipy.integrate.cumulative_trapezoid(integrand_f(zs), zs)

    return xs, zs


def param_b_c(z: float):
    """rho parameterization table from https://arxiv.org/pdf/2011.09869.pdf"""

    bins = np.array(
        [
            # 0.0,
            4.0,
            10.0,
            40.0,
            100.0,
        ]
    )
    b = np.array(
        [
            1222.6562,
            1144.9069,
            1305.5948,
            540.1778,
            1.0,
        ]
    )
    c = np.array(
        [
            994186.38,
            878153.55,
            636143.04,
            772170.16,
            1e9,
        ]
    )

    idxs = np.searchsorted(bins, z)

    return b[idxs], c[idxs]


if __name__ == "__main__":
    print(*slant_depth_steps(0, 10, 10, dz=0.001), sep="\n")
    print(*slant_depth(0, 10, 10), sep="\n")
