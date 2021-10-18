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
import quadpy as qp

from numpy.polynomial import Polynomial

from ... import constants as const

__all__ = ["rho", "slant_depth", "slant_depth_integrand", "slant_depth_steps"]


def rho(z):
    """
    Density (g/cm^3) parameterized from altitude (z) values

    Computation from equation (2) in https://arxiv.org/pdf/2011.09869.pdf
    """

    z = np.asarray(z)
    bins = np.array([4.0, 10.0, 40.0, 100.0])
    b_arr = np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0])
    c_arr = np.array([994186.38, 878153.55, 636143.04, 772170.16, 1e9])

    idxs = np.searchsorted(bins, z)
    b, c = np.asarray(b_arr[idxs]), np.asarray(c_arr[idxs])
    p = np.asarray(b / c)

    mask = z <= 100
    p[mask] *= np.exp(-1e5 * z[mask] / c[mask])
    return p


_polyrho = Polynomial(
    [
        -1.00867666e-07,
        2.39812768e-06,
        9.91786255e-05,
        -3.14065045e-04,
        -6.30927456e-04,
        1.70053229e-03,
        2.61087236e-03,
        -5.69630760e-03,
        -2.12098836e-03,
        5.68074214e-03,
        6.54893281e-04,
        -1.98622752e-03,
    ],
    domain=[0.0, 100.0],
)


def polyrho(z):
    """
    Density (g/cm^3) parameterized from altitude (z) values

    Computation is an (11) degree polynomial fit to equation (2)
    in https://arxiv.org/pdf/2011.09869.pdf
    Fit performed using numpy.Polynomial.fit
    """

    p = np.where(z < 100, _polyrho(z), np.reciprocal(1e9))
    return p


def slant_depth_integrand(z, theta_tr, rho=polyrho, earth_radius=const.earth_radius):
    """
    Integrand for computing slant_depth from input altitude z.
    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    """

    theta_tr = np.asarray(theta_tr)

    i = earth_radius ** 2 * np.cos(theta_tr) ** 2
    j = z ** 2
    k = 2 * z * earth_radius

    ijk = i + j + k + 1e-12

    return 1e5 * rho(z) * ((z + earth_radius) / np.sqrt(ijk))


def slant_depth(
    z_lo,
    z_hi,
    theta_tr,
    earth_radius=const.earth_radius,
    func=slant_depth_integrand,
    epsabs=1e-1,
    epsrel=1e-1,
    *args,
    **kwargs
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
    func: callable
        The integrand for slant_depth. If None, defaults to `slant_depth_integrand()`.

    Returns
    -------
    x_sd: ndarray
        slant_depth g/cm^2
    err: (float) numerical error.

    """

    global nnn
    nnn = 0

    theta_tr = np.asarray(theta_tr)

    def f(x):
        y = np.multiply.outer(z_hi - z_lo, x).T + z_lo
        return (func(y, theta_tr=theta_tr, earth_radius=earth_radius) * (z_hi - z_lo)).T

    return qp.quad(f, 0.0, 1.0, epsabs=epsabs, epsrel=epsrel)


def slant_depth_steps(
    z_lo,
    z_hi,
    theta_tr,
    dz=0.01,
    earth_radius=const.earth_radius,
    func=slant_depth_integrand,
):
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
    func: real valued function
        the integrand for slant_depth. If None, Default of `slant_depth_integrand()` is used.

    Returns
    -------
        xs: float
            slant depth at each altitude along track.
        zs: float
            altitudes at which slant_depth was evaluated.

    """

    def f(x):
        return func(x, theta_tr, earth_radius)

    zs = np.arange(z_lo, z_hi, dz)
    xs = scipy.integrate.cumulative_trapezoid(f(zs), zs)

    return xs, zs


def param_b_c(z):
    """rho parameterization table from https://arxiv.org/pdf/2011.09869.pdf"""

    bins = np.array([4.0, 10.0, 40.0, 100.0])
    b = np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0])
    c = np.array([994186.38, 878153.55, 636143.04, 772170.16, 1e9])

    idxs = np.searchsorted(bins, z)
    return np.asarray(b[idxs]), np.asarray(c[idxs])
