# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Atmospheric model calculations for upward going showers.

Slant depth and related computations are implementations of equations in
https://arxiv.org/pdf/2011.09869.pdf by Cummings et. al.

author: Alexander Reustle
date: 2021 August 12
"""

import numpy as np
import scipy.integrate

from typing import Callable, Union, Tuple
from numpy.typing import ArrayLike

from ... import constants as const

__all__ = ["rho", "slant_depth", "slant_depth_integrand", "slant_depth_steps"]


def rho(z: Union[float, ArrayLike]) -> ArrayLike:
    """
    Density (g/cm^3) parameterized from altitude (z) values

    Computation from equation (2) in https://arxiv.org/pdf/2011.09869.pdf
    """

    z = np.array([z]) if isinstance(z, float) else z
    b, c = param_b_c(z)
    p = b / c

    mask = z <= 100
    p[mask] *= np.exp(-1e5 * z[mask] / c[mask])

    return p


def slant_depth_integrand(
    z: float,
    theta_tr: Union[float, ArrayLike],
    earth_radius: float = const.earth_radius,
) -> ArrayLike:
    """
    Integrand for computing slant_depth from input altitude z.
    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    """

    theta_tr = np.array([theta_tr]) if isinstance(theta_tr, float) else theta_tr

    i = earth_radius ** 2 * np.cos(theta_tr) ** 2
    j = (z) ** 2
    k = 2 * z * earth_radius

    ijk = i[:, None] + j + k

    return 1e5 * rho(z) * ((z + earth_radius) / np.sqrt(ijk))


def slant_depth(
    z_lo: float,
    z_hi: float,
    theta_tr: Union[float, ArrayLike],
    earth_radius: float = const.earth_radius,
    integrand_f: Callable[..., ArrayLike]=None,
):
    """
    Slant-depth in g/cm^2 from equation (3) in https://arxiv.org/pdf/2011.09869.pdf

    Parameters
    ----------
    z_lo : float
        Starting altitude for slant depth track.
    z_hi : float
        Stopping altitude for slant depth track.
    theta_tr: float, array_like
        Trajectory angle in radians between the track and earth zenith.
    earth_radius: float
        Radius of a spherical earth. Default from nuspacesim.constants
    integrand_f: callable
        The integrand for slant_depth. If None, defaults to `slant_depth_integrand()`.

    Returns
    -------
    x_sd: ndarray
        slant_depth g/cm^2
    err: (float) numerical error.

    """

    if integrand_f is None:
        integrand_f = slant_depth_integrand

    f = lambda x: integrand_f(x, theta_tr=theta_tr, earth_radius=earth_radius)

    return scipy.integrate.quad_vec(f, z_lo, z_hi)


def slant_depth_steps(
    z_lo: float,
    z_hi: float,
    theta_tr: Union[float, ArrayLike],
    dz: float = 0.01,
    earth_radius: float = const.earth_radius,
    integrand_f: Callable[..., ArrayLike] = None,
) -> Tuple:
    r"""Slant-depth integral approximated along path.

    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    along a full length using the cumulative_trapezoid rule.

    Parameters
    ----------
    z_lo: float
        starting altitude for slant depth track.
    z_hi: float
        stopping altitude for slant depth track.
    theta_tr: float
        trajectory angle of track to observer.
    earth_radius: float
        radius of a spherical earth. Default from nuspacesim.constants
    dz: float
        static step size for sampling points in range [z_lo, z_hi]
    integrand_f: real valued function
        the integrand for slant_depth. If None, Default of `slant_depth_integrand()` is used.

    Returns
    -------
        xs: float
            slant depth at each altitude along track.
        zs: float
            altitudes at which slant_depth was evaluated.

    """

    if integrand_f is None:
        integrand_f = slant_depth_integrand

    f = lambda x: integrand_f(x, theta_tr, earth_radius)

    zs = np.arange(z_lo, z_hi, dz)
    xs = scipy.integrate.cumulative_trapezoid(f(zs), zs)

    return xs, zs


def param_b_c(
    z: Union[float, ArrayLike]
) -> Tuple[ArrayLike, ArrayLike]:
    """rho parameterization table from https://arxiv.org/pdf/2011.09869.pdf"""

    bins = np.array([4.0, 10.0, 40.0, 100.0])
    b = np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0])
    c = np.array([994186.38, 878153.55, 636143.04, 772170.16, 1e9])

    idxs = np.searchsorted(bins, z)
    return b[idxs], c[idxs]


if __name__ == "__main__":

    # Density
    # kms = np.arange(0, 88, 2)
    # ps = rho(kms)
    # print("Density (g/cm^3)", *[f"{a}\t {b:.4e}" for a, b in zip(kms, ps)], sep="\n")

    # slant-depth from gaussian quadriture
    X = slant_depth(0, 100, np.pi / 4)
    print(f"Slant Depth: {X[0][0]}, numerical error: {X[1]}", sep="\n")

    # slant-depth from trapezoidal rule
    Y = slant_depth_steps(0, 100, np.pi / 4)
    print(f"Slant Depth steps: {Y[0][:, -1]}", sep="\n")

    theta_tr = np.linspace(-np.pi / 2, np.pi / 2, 100)
    # Y = slant_depth_steps(1, 100, theta_tr)
    # print(f"Slant Depth steps (-pi/2 to +pi/2): {Y[0]}", sep="\n")

    # plot over multiple starting heights.
    sds = [slant_depth(z_lo, 100, theta_tr)[0] for z_lo in (0, 1, 2, 5, 10)]
    tds = [
        slant_depth_steps(z_lo, 100, theta_tr)[0][:, -1] for z_lo in (0, 1, 2, 5, 10)
    ]
    labs = [f"z: [{z_lo}, 100] km" for z_lo in (0, 1, 2, 5, 10)]

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 18})

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, squeeze=True)
    coloridx = np.linspace(0, 1, len(sds))
    for i, s, t, l in zip(coloridx, sds, tds, labs):
        ax1.plot(theta_tr, s, alpha=0.5, label=l, color=plt.cm.jet(i))
        ax1.plot(theta_tr, t, ":", label=l + "trap", color=plt.cm.jet(i))
    ax1.set_ylabel(r"slant depth ($\frac{g}{cm^2}$)")
    ax1.grid()
    ax1.legend()
    ax1.set_ylim([0, 4e4])
    for i, s, t, l in zip(coloridx, sds, tds, labs):
        ax2.semilogy(theta_tr, s, alpha=0.5, label=l, color=plt.cm.jet(i))
        ax2.semilogy(theta_tr, t, ":", label=l + "trap", color=plt.cm.jet(i))
    ax2.set_ylabel(r"slant depth (log $\frac{g}{cm^2}$)")
    ax2.grid()
    ax2.legend()
    ax2.set_ylim([7e1, 4e4])
    ax2.set_xlabel(r"$\theta_{tr}$ (radians)")
    fig.suptitle(
        r"Slant Depth over $\theta_{tr}\in\left(\frac{-\pi}{2}, \frac{\pi}{2}\right)$"
    )
    plt.show()
