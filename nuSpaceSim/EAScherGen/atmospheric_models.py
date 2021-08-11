import numpy as np
import scipy.integrate
from nuSpaceSim import constants as c
from typing import Iterable


def rho(z):
    """
    rho parameterized from z values

    Computation from equation (2) in https://arxiv.org/pdf/2011.09869.pdf
    """
    z = z if isinstance(z, Iterable) else np.array([z])

    rval = np.empty_like(z)

    b, c = param_b_c(z[z <= 100])
    rval[z <= 100] = (b / c) * np.exp(-z / c)

    b, c = param_b_c(z[z > 100])
    rval[z > 100] = b / c

    return rval


def slant_depth(
    z_lo: float,
    z_hi: float,
    theta_tr: float,
    earth_radius: float = c.earth_radius,
    dz = 0.01,
):
    """
    Slant-depth integral

    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    using gaussian quadriture, and along a full length using the cumulative_trapezoid
    rule.
    """
    f = lambda z: rho(z) * (
        (z + earth_radius) / np.sqrt(earth_radius ** 2) * np.cos(theta_tr) ** 2
        + z ** 2
        + 2 * z * earth_radius
    )

    res, err = scipy.integrate.quad(f, z_lo, z_hi)

    zs = np.arange(z_lo, z_hi, dz)
    cum_trapz = scipy.integrate.cumulative_trapezoid(f(zs), zs)

    return (zs, cum_trapz, res, err)


def param_b_c(z: float):

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
    print(*slant_depth(0, 10, 10), sep='\n')
