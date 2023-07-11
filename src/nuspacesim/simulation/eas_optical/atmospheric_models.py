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

# from functools import cache

import cubepy as cp
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline

from ... import constants as const
from ...utils.ode import dormand_prince_rk54

# from scipy.special import erf, hyp2f1

__all__ = [
    # "cummings_atmospheric_density",
    "shibata_grammage",
    "slant_depth_us",
    # "slant_depth_integrand",
    # "slant_depth",
    # "slant_depth_trig_approx",
    # "slant_depth_trig_behind_ahead",
]


def us_std_atm_density(z, earth_radius=const.earth_radius):
    H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852, np.inf])
    Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
    T_b = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    # fmt: off
    P_b = 1.01325e5 * np.array(
        [1.0, 2.233611e-1, 5.403295e-2, 8.5666784e-3, 1.0945601e-3, 6.6063531e-4,
         3.9046834e-5, 3.68501e-6, ])
    # fmt: on

    Rstar = 8.31432e3
    M0 = 28.9644
    gmr = 34.163195

    z = np.asarray(z)
    h = z * earth_radius / (z + earth_radius)

    pressure = np.zeros_like(z)
    temperature = np.zeros_like(z)

    for i in range(len(H_b) - 1):
        m = (h >= H_b[i]) & (h < H_b[i + 1])
        deltah = h[m] - H_b[i]
        temperature[m] = T_b[i] + Lm_b[i] * deltah
        # mask = Lm_b[i] == 0
        if Lm_b[i] == 0:
            pressure[m] = P_b[i] * np.exp(-(gmr / T_b[i]) * deltah)
        else:
            pressure[m] = P_b[i] * (T_b[i] / temperature[m]) ** (gmr / Lm_b[i])

    density = (pressure * M0) / (Rstar * temperature)  # kg/m^3
    return 1e-3 * density  # g/cm^3


def slant_depth_us(z_lo, z_hi, theta_tr, eps=1e-4, Re=const.earth_radius, **kwargs):
    H_ = np.array([0, 11, 20, 32, 47, 51, 71, 84.852, 1000])
    Lm = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
    T_ = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    # fmt: off
    P_ = 1.01325e5 * np.array(
        [1.0, 2.233611e-1, 5.403295e-2, 8.5666784e-3, 1.0945601e-3, 6.6063531e-4,
         3.9046834e-5, 3.68501e-6, ])
    # fmt: on

    def h2z(h):
        return h * Re / (Re - h)

    def z2h(z):
        return z * Re / (Re + z)

    Z_ = h2z(H_)

    X = np.zeros_like(z_lo)

    for b in range(len(H_) - 1):
        alt_lo = np.clip(z_lo, Z_[b], Z_[b + 1])
        alt_hi = np.clip(z_hi, Z_[b], Z_[b + 1])

        m = alt_lo != alt_hi
        if not np.any(m):
            continue

        Rstar = 8.31432e3
        M0 = 28.9644
        gmr = 34.163195

        def pressure_Lm0(dh):
            return P_[b] * np.exp(-(gmr / T_[b]) * dh)

        def pressure_Lm1(dh):
            temp = T_[b] + Lm[b] * dh
            return P_[b] * (T_[b] / temp) ** (gmr / Lm[b])

        pressure = pressure_Lm0 if Lm[b] == 0 else pressure_Lm1

        def df(z, t):
            dh = z2h(z) - H_[b]
            temp = T_[b] + Lm[b] * dh
            rho = 1e-3 * pressure(dh) * M0 / (Rstar * temp)  # kg/m^3 -> g/cm^3

            return (
                1e5
                * rho
                * (Re + z)
                / np.sqrt(z**2 + 2 * Re * z + Re**2 * np.cos(t) ** 2 + eps)
            )

        X[m] += cp.integrate(
            df,
            np.asarray(alt_lo)[m],
            np.asarray(alt_hi)[m],
            args=(np.asarray(theta_tr)[m],),
            rtol=1e-8,
            atol=1e-5,
            itermax=32,
        )[0]

    return X


def slant_depth_riemann(z_lo, z_hi, theta_tr, nstep=1e4, Re=const.earth_radius):
    zs = np.linspace(z_lo, z_hi, int(nstep))
    dstep = (z_hi - z_lo) / int(nstep)
    rho = us_std_atm_density(zs)
    X = (
        rho
        * (zs + Re)
        / np.sqrt(
            zs**2 + 2 * Re * zs + Re**2 * np.cos(theta_tr)[None, ...] ** 2 + 1e-4
        )
    )
    return 1e5 * np.sum(X, axis=0) * dstep


def slant_depth_integrand(z, theta_tr, rho=us_std_atm_density, Re=const.earth_radius):
    """
    Integrand for computing slant_depth from input altitude z.
    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    """

    theta_tr = np.asarray(theta_tr)
    L_R = np.sqrt(z**2 + 2 * z * Re + Re**2 * np.cos(theta_tr) ** 2)
    return 1e5 * rho(z) * ((z + Re) / (L_R + 1e-4))


def slant_depth_inverse_func(
    z, theta_tr, rho=us_std_atm_density, Re=const.earth_radius
):
    theta_tr = np.asarray(theta_tr)
    L_R = np.sqrt(z**2 + 2 * z * Re + Re**2 * np.cos(theta_tr) ** 2)
    return 1e-5 * L_R / (rho(z) * (z + Re) + 1e-4)


def inv_optical_depth_integrand(_, L, costhet, Re=const.earth_radius):
    H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852, np.inf])
    Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
    T_b = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    # fmt: off
    # Inverse Pressure, proportional to volume
    V_b = 1.01325e5 * np.array(
        [1.0, 2.233611e+1, 5.403295e+2, 8.5666784e+3, 1.0945601e+3, 6.6063531e+4,
         3.9046834e+5, 3.68501e+6, ])
    # fmt: on
    # M0Rstar = 28.9644 / 8.31432e3
    RstarM0 = 8.31432e3 / 28.9644
    gmr = 34.163195
    z = -Re + np.sqrt(L**2 + Re**2 + 2 * L * Re * costhet)
    h = z * Re / (z + Re)
    a = np.zeros_like(h)
    for i in range(len(H_b) - 1):
        m = (h >= H_b[i]) & (h < H_b[i + 1])
        dh = h[m] - H_b[i]
        if Lm_b[i] == 0:
            a[m] = V_b[i] * T_b[i] * np.exp((gmr / T_b[i]) * dh)
        else:
            q = gmr / Lm_b[i]
            a[m] = V_b[i] * T_b[i] ** (-q) * np.power(T_b[i] + Lm_b[i] * dh, q - 1)

    return 1e-2 * RstarM0 * a


# def path_length_from_slant_depth(L0, L1, theta_tr, Re=const.earth_radius):
#
#     cos_theta = np.cos(theta_tr)
#
#     ts, ys = dormand_prince_rk54(optical_depth_integrand,


def shibata_grammage(z):
    z = np.asarray(z)
    # conditional cutoffs
    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25

    # 1 / 0.19
    r1 = 1 / 0.19

    # X_v vertical atmospheric depth g / cm^2
    X_v = np.empty_like(z)
    X_v[mask1] = np.power(((z[mask1] - 44.34) / -11.861), r1)
    X_v[mask2] = np.exp((z[mask2] - 45.5) / -6.34)
    X_v[mask3] = np.exp(13.841 - np.sqrt(28.920 + 3.344 * z[mask3]))

    return X_v


def index_of_refraction_air(altitude, vert_depth_f=shibata_grammage):
    r"""Index of refraction in air (Nair)

    Index of refraction of air as a function of altitude in km. Computed from vertical
    atmospheric depth X_v (g/cm^2) via the shibata_grammage

    Hillas 1475 eqn (2)
    """
    X_v = vert_depth_f(altitude)
    temperature = 204.0 + 0.091 * X_v
    n = 1.0 + 0.000296 * (X_v / 1032.9414) * (273.2 / temperature)
    return n


def rad_len_atm_depth(x, L0recip=0.02727768685):
    """
    T is X / L0, units of radiation length scaled g/cm^2
    default L0 = 36.66
    """
    return x * L0recip


def cherenkov_angle(AirN):
    thetaC = np.arccos(np.asarray(1.0 / AirN))
    return thetaC


def cherenkov_threshold(AirN):
    np.asarray(AirN)
    eCthres = np.full_like(AirN, 1e6)
    mask = (AirN != 0) & (AirN != 1)
    eCthres[mask] = 0.511 / np.sqrt(1.0 - (1.0 / AirN[mask] ** 2))
    return eCthres


def cherenkov_photons_created(wavelength, thetaC):
    # 1e9 * 2 * np.pi * (1.0 / 137.04) * np.sin(thetaC) ** 2 * (1.0 / wavelength**2)
    coeff = 45849279.8247196912  # 1e9 * 2 * pi * alpha
    return coeff * np.sin(thetaC) ** 2 * (1.0 / wavelength**2)


def sokolsky_rayleigh_scatter(wavelength, X):
    return np.exp(np.multiply(-X / 2974.0, (400.0 / wavelength) ** 4))


_spline_ozone = BSpline(
    # fmt: off
    np.array([
        3.47776272e-26, 3.47776272e-26, 3.47776272e-26, 3.47776272e-26,
        1.67769146e1, 2.42250765e1, 3.29917411e1, 4.08815284e1, 4.88486345e1,
        5.10758252e1, 5.69937723e1, 6.34112652e1, 6.58453130e1, 9.02965953e1,
        9.02965953e1, 9.02965953e1, 9.02965953e1
    ]),
    np.array([
        2.80373832, 3.41049110, -4.22593929, 2.57297359e1, 6.33106297,
        1.72934400, 6.66859235e-2, 8.66981139e-3, 3.18321395e-2, 2.45436302e-2,
        2.63104170e-2, 2.51036483e-2, 2.51207729e-2, 0.0, 0.0, 0.0, 0.0
    ]),
    # fmt: on
    3,
    extrapolate=False,
)

_spline_ozone_antideriv = BSpline(
    # fmt: off
    np.array([
        3.47776272e-26, 3.47776272e-26, 3.47776272e-26, 3.47776272e-26,
        3.47776272e-26, 1.67769146e01, 2.42250765e01, 3.29917411e01,
        4.08815284e01, 4.88486345e01, 5.10758252e01, 5.69937723e01,
        6.34112652e01, 6.58453130e01, 9.02965953e01, 9.02965953e01,
        9.02965953e01, 9.02965953e01, 9.02965953e01
    ]),
    np.array([
        0.0, 11.759519588846869, 32.41437153886416, -2.4409022011352945,
        260.5268300289521, 311.2888495897276, 322.8973948796908,
        323.2975442838028, 323.34637642598335, 323.4816365862455,
        323.72229160561886, 323.94134439572065, 324.1100743635356,
        324.26363314097864, 324.26363314097864, 324.26363314097864,
        324.26363314097864, 324.26363314097864, 324.26363314097864
    ]),
    # fmt: on
    4,
)


def ozone_losses(wavelength, z, detector_altitude, theta_tr):
    term1 = -2.3457647210565447 + 3.386767160062623 * 1.0 / np.sqrt(np.cos(theta_tr))

    def ozone_depth_approx(x):
        s = 0.8509489496334921 * _spline_ozone_antideriv(x)
        return term1 * (5.090655199334124 + s)

    d1 = ozone_depth_approx(detector_altitude)
    d0 = ozone_depth_approx(z)
    depth = d1 - d0

    ozone_kappa = 10 ** (110.5 - 44.21 * np.log10(wavelength))

    return np.exp(-1e-3 * depth * ozone_kappa)


_spline_aod = BSpline(
    # fmt: off
    np.array([
        0., 0., 0., 0., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
        28., 29., 30., 31., 32., 33., 40., 50., 60., 70., 80., 100., 100.,
        100., 100.
    ]),
    np.array([
        2.5e-01, 1.44032618e-01, 8.79347648e-02, 6.34311861e-02,
        5.44663855e-02, 4.87032719e-02, 4.47205268e-02, 4.24146208e-02,
        3.76209899e-02, 3.51014195e-02, 3.19733321e-02, 2.90052521e-02,
        2.60056594e-02, 2.29721102e-02, 2.01058999e-02, 1.66042901e-02,
        1.54769396e-02, 1.14879516e-02, 1.05712540e-02, 6.22703239e-03,
        6.52061645e-03, 3.69050182e-03, 2.71737625e-03, 3.43999316e-03,
        1.52265111e-03, 2.46940239e-03, 5.99739313e-04, 1.13164035e-03,
        8.73699276e-04, 1.37356254e-03, -3.67949454e-4, 9.82352716e-05,
        -2.49916324e-5, 5.51770388e-05, -3.37867722e-5, 1.03674479e-05,
        -2.77699498e-6, 7.40531994e-07, -4.93687996e-7, 2.46843998e-7, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]),
    # fmt: on
    3,
    extrapolate=False,
)
""" BSpline approximation. Aerosol Depth at 55 microns."""


def aerosol_optical_depth(z):
    """Aerosol Optical Depth at 55 microns."""
    return _spline_aod(np.asarray(z))


aP = Polynomial(
    np.array(
        [-1.2971, 0.22046e-01, -0.19505e-04, 0.94394e-08, -0.21938e-11, 0.19390e-15]
    )
)
"""Degree 5 polynomial fit of 1/beta_aerosol vs wavelength."""


def aBetaF(wavelength):
    return (1.0 / aP(wavelength)) / 0.158


# def propagation_angle(beta_tr, z, Re=6378.1):
#     return np.arccos((Re / (Re + z)) * np.cos(beta_tr))


def elterman_mie_aerosol_scatter(wavelength, z, theta):
    """
    z values above 30 (km altitude) return OD filled to 0, this should then
    return aTrans = 1, but in the future masking may be used to further
    optimze for performance by avoiding this computation.
    """

    # theta = np.sin(thetaProp)
    aTrans = np.exp(-np.multiply(aerosol_optical_depth(z) / theta, aBetaF(wavelength)))
    return aTrans


# def cummings_atmospheric_density(z):
#     """
#     Density (g/cm^3) parameterized from altitude (z) values
#
#     Computation from equation (2) in https://arxiv.org/pdf/2011.09869.pdf
#     """
#
#     z = np.asarray(z)
#     bins = np.array([4.0, 10.0, 40.0, 100.0])
#     b_arr = np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0])
#     c_arr = np.array([994186.38, 878153.55, 636143.04, 772170.16, 1e9])
#
#     idxs = np.searchsorted(bins, z)
#     b, c = np.asarray(b_arr[idxs]), np.asarray(c_arr[idxs])
#     p = np.asarray(b / c)
#
#     mask = z <= 100
#     p[mask] *= np.exp(-1e5 * z[mask] / c[mask])
#     return p
#
#
# # fmt: off
# _polyrho = Polynomial(
#     [-1.00867666e-07, 2.39812768e-06, 9.91786255e-05, -3.14065045e-04, -6.30927456e-04,
#      1.70053229e-03, 2.61087236e-03, -5.69630760e-03, -2.12098836e-03, 5.68074214e-03,
#      6.54893281e-04, -1.98622752e-03, ],
#     domain=[0.0, 100.0],
# )
# # fmt: on
#
#
# def polyrho(z):
#     """
#     Density (g/cm^3) parameterized from altitude (z) values
#
#     Computation is an (11) degree polynomial fit to equation (2)
#     in https://arxiv.org/pdf/2011.09869.pdf
#     Fit performed using numpy.Polynomial.fit
#     """
#
#     p = np.where(z < 100, _polyrho(z), 1e-9)
#     return p
#
#
# def deriv_polyrho(z, m=1):
#     p = np.where(z < 100, _polyrho.deriv(m)(z), 1e-9)
#     return p

# def slant_depth_analytic(z_lo, z_hi, theta_tr, Re=6371.8):
#     H = np.array([0, 11, 20, 32, 47, 51, 71, 84.852, 1e3])
#     Z = Re * H / (Re - H)
#     Lm = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
#     T = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
#     # fmt: off
#     P = 1.01325e5 * np.array([1.0, 2.233611e-1, 5.403295e-2, 8.5666784e-3, 1.0945601e-3, 6.6063531e-4, 3.9046834e-5, 3.68501e-6])
#     # fmt: on
#     Rstar = 8.31432e3
#     M0 = 28.9644
#     M_R = M0 / Rstar
#     gmr = 34.163195
#     q = gmr / Lm[0]
#     r = -q - 1
#     cos_t = np.cos(theta_tr)
#     sin_t = np.sin(theta_tr)
#     # Layers with constant temperature (lack-rate Lm == 0): { 1, 4 }
#     # Layers with changing temperature (lack-rate Lm != 0): { 0, 2, 3, 5, 6 }
#     X = 0
#     for b in range(1):
#         alt_lo = np.clip(z_lo, Z[b], Z[b + 1])
#         alt_hi = np.clip(z_hi, Z[b], Z[b + 1])
#         dz = alt_hi - alt_lo
#         h_lo = (Re * alt_lo) / (Re + alt_lo)
#         h_hi = (Re * alt_hi) / (Re + alt_hi)
#         Tp = T[b] + Lm[b] * (h_lo - H[0])
#         gamma = Lm[b] * (h_hi - h_lo) / Tp
#
#         R = Re + alt_lo
#         alpha1 = dz / (R * (1 + sin_t))
#
#         d_R = dz / R
#         R_d = R / dz
#         zeta = (R / dz) * (1 - sin_t)
#
#         # def X_1a():
#         #     alpha0 = dz / (R * zeta)
#         #
#         #     print(f"gamma: {gamma} alpha1: {alpha1} alpha0: {alpha0}")
#         #
#         #     def f(k, m, n):
#         #         Q = mp.binomial(r, k) * mp.binomial(-0.5, m) * mp.binomial(-0.5, n)
#         #         Q *= gamma**k * alpha1**m * alpha0**n
#         #         X = Q * (dz / (2 + k + m + n) + R / (1 + k + m + n))
#         #         return X
#         #
#         #     X = mp.nsum(f, [0, mp.inf], [0, mp.inf], [0, mp.inf], tol=1e-4, maxterms=20)
#         #     X *= 1e2 * M_R * P[b] * T[b] ** q * Tp**r * d_R / cos_t
#         #     return X
#         #
#         # def X_1b():
#         #     print(f"gamma: {gamma} alpha: {alpha1} zeta: {zeta} d_R: {d_R}")
#         #
#         #     def f(k, m, n):
#         #         Q = mp.binomial(r, k) * mp.binomial(-0.5, m) * mp.binomial(-0.5, n)
#         #         Q *= 2 * gamma**k * alpha1**m * zeta**n
#         #         X = Q * (dz / (2 * (k + m - n) + 3) + R / (1 + 2 * (k + m - n)))
#         #         # print(k, m, n, ":", X)
#         #         return X
#         #
#         #     X = mp.nsum(f, [0, mp.inf], [0, mp.inf], [0, mp.inf], tol=1e-4, maxterms=20)
#         #     X *= 1e2 * M_R * P[b] * T[b] ** q * Tp**r * d_R * np.sqrt(1 / (1 + sin_t))
#         #     return X
#         #
#         # def X_1c():
#         #     def f(k, m):
#         #         p0 = dz + R - R * sin_t
#         #         p1 = R * sin_t - R
#         #         p2 = k + m + 1
#         #         Q = mp.binomial(r, k) * mp.binomial(-0.5, m)
#         #         Q *= gamma**k * alpha1**m
#         #         Q /= p2 + 0.5
#         #         Q *= (p0 / dz) ** 0.5
#         #         Q *= (R * 2 * sin_t * p2 + R) * (dz / p1) ** (-k - m) * mp.hyp2f1(
#         #             0.5, -k - m, 1.5, p0 / (R * (1 - sin_t))
#         #         ) + dz
#         #         # print(k, m, ":", Q)
#         #         return Q
#         #
#         #     X = mp.nsum(
#         #         f, [0, mp.inf], [0, mp.inf], tol=1e-4, maxterms=20, verbose=True
#         #     )
#         #     X *= (
#         #         1e2 * M_R * P[b] * T[b] ** q * Tp**r * np.sqrt(alpha1)
#         #     )  # * np.sqrt(d_R)
#         #     return X
#         #
#         # def X_1d():  # IBP then Appell Hypergeom the remaining integral.
#         #     a = R / dz * (1 - sin_t)
#         #     # b = R / dz * (1 + sin_t)
#         #     c = (dz**2 + 2 * R * dz + R**2 * cos_t**2) ** 0.5
#         #     y = gamma
#         #     g = 2 * R / dz * sin_t
#         #
#         #     X1 = (1 + y) ** r * (c + (1 - y) / y * R * cos_t)
#         #     # _x1 = 2 / 3 * r * dz * (1 + a) ** 1.5
#         #     # _x1 *= (1 - y * (1 + a)) ** r / (y + y * a - 1) * g**0.5
#         #     # _x1 *= mp.appellf1(
#         #     #     1.5, -0.5, 1 - r, 2.5, (1 + a) / (-g), y * (1 + a) / (y * a - 1)
#         #     # )
#         #     # X1 += _x1
#         #
#         #     X0 = R / y * cos_t
#         #     # _x0 = 2 / 3 * r * dz * a**1.5 * (1 - y * a) ** r / (y * a - 1) * g**0.5
#         #     # _x0 *= mp.appellf1(
#         #     #     1.5,
#         #     #     -0.5,
#         #     #     1 - r,
#         #     #     2.5,
#         #     #     (sin_t - 1) / (2 * sin_t),
#         #     #     (y * a) / (y * a - 1),
#         #     # )
#         #     # X0 += _x0
#         #
#         #     X = 1e2 * M_R * P[b] * T[b] ** q * Tp**r * (X1 - X0)
#         #
#         #     return X
#
#         def X_1e():
#             _a = R_d**2 * cos_t**2
#             _b = 2 * R_d
#             _c = 1
#             Del = 4 * _a * _c - _b**2
#
#             @cache
#             def fR(t: int, _q: int = 1):
#                 return (_a + _b * t + _c * t**2) ** (_q / 2)
#
#             def psi(t):
#                 if Del == 0:
#                     return mp.log(2 * t + _b)
#                 if 2 * t + _b > mp.sqrt(-Del):
#                     return mp.log(2 * fR(t) + 2 * t + _b)
#                 if 2 * t + _b > mp.sqrt(-Del):
#                     return mp.log(2 * fR(t) - 2 * t - _b)
#
#             def I0(t):
#                 return (2 * _c * t + _b) * fR(t) / 4 + Del / 8 * psi(t)
#
#             def I1(t):
#                 return (
#                     fR(t, 3) / 3
#                     - (2 * _c * t + _b) * _b / 8 * fR(t)
#                     - _b * Del / 16 * psi(t)
#                 )
#
#             @cache
#             def Im(m: int, t: int):
#                 if m == 0:
#                     return I0(t)
#                 if m == 1:
#                     return I1(t)
#                 m2 = m + 2
#                 return (
#                     t ** (m - 1) * fR(t, 3) / m2
#                     - (2 * m + 1) * _b / (2 * m2) * Im(m - 1, t)
#                     - (m - 1) * _a / m2 * Im(m - 2, t)
#                 )
#
#             def f(m, t):
#                 Q = mp.binomial(r - 1, m) * gamma**m
#                 Q *= Im(m, 1) - Im(m, 0)
#                 print(m, Q)
#                 return Q
#
#             X0 = dz * fR(0) + (1 - gamma) / gamma * R * cos_t
#             X0 -= (
#                 r
#                 * dz
#                 * mp.nsum(
#                     lambda m: f(m, 0), [0, mp.inf], tol=1e-4, maxterms=20, verbose=True
#                 )
#             )
#
#             X1 = (1 + gamma) ** r * (dz * fR(1) + (1 - gamma) / gamma * R * cos_t)
#             X1 -= (
#                 r
#                 * dz
#                 * mp.nsum(
#                     lambda m: f(m, 1), [0, mp.inf], tol=1e-4, maxterms=20, verbose=True
#                 )
#             )
#             X = 1e2 * M_R * P[b] * T[b] ** q * Tp**r * (X1 - X0)
#             return X
#
#         #
#         X = X_1e()
#         print(X)
#         # print(theta_tr, np.arcsin(1 - d_R))
#         # X = X_1a() if theta_tr < np.arcsin(1 - d_R) else X_1b()
#
#     return X
#
#
# def slant_depth_inverse_func(z, theta_tr, Re=const.earth_radius):
#     H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852, np.inf])
#     Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
#     T_b = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
#     # fmt: off
#     P_b = 1.01325e5 * np.array(
#         [1.0, 2.233611e-1, 5.403295e-2, 8.5666784e-3, 1.0945601e-3, 6.6063531e-4,
#          3.9046834e-5, 3.68501e-6, ])
#     # fmt: on
#     M0Rstar = 28.9644 / 8.31432e3
#     gmr = 34.163195
#
#     z = np.asarray(z)
#     theta_tr = np.asarray(theta_tr)
#     a = np.sqrt(z**2 + 2 * z * Re + Re**2 * np.cos(theta_tr) ** 2)
#     h = z * Re / (z + Re)
#
#     pressure = np.zeros_like(z)
#     temperature = np.zeros_like(z)
#     for i in range(len(H_b) - 1):
#         m = (h >= H_b[i]) & (h < H_b[i + 1])
#         deltah = h[m] - H_b[i]
#         temperature[m] = T_b[i] + Lm_b[i] * deltah
#         # mask = Lm_b[i] == 0
#         if Lm_b[i] == 0:
#             pressure[m] = P_b[i] * np.exp(-(gmr / T_b[i]) * deltah)
#         else:
#             pressure[m] = P_b[i] * (T_b[i] / temperature[m]) ** (gmr / Lm_b[i])
#
#     # density = (pressure * M0Rstar) / temperature  # kg/m^3
#     # return 1e-3 * density  # g/cm^3
#     # return 1e-5 * a / (rho(z) * (z + Re) + 1e-9)
#     return 1e-2 * a * temperature / (pressure * M0Rstar)
#
#
# def us_std_atm_density_deriv(
#     z, rho=us_std_atm_density, earth_radius=const.earth_radius
# ):
#     H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852, np.inf])
#     Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
#     T_b = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
#     gmr = 34.163195
#
#     z = np.asarray(z)
#     h = z * earth_radius / (z + earth_radius)
#     alpha = np.zeros_like(z)
#     temperature = np.zeros_like(z)
#
#     for i in range(len(H_b) - 1):
#         m = (h >= H_b[i]) & (h < H_b[i + 1])
#         deltah = h[m] - H_b[i]
#         temperature[m] = T_b[i] + Lm_b[i] * deltah
#         alpha[m] = gmr if Lm_b[i] == 0 else Lm_b[i] * ((gmr / Lm_b[i]) + 1)
#
#     density = rho(z, earth_radius)
#     return -density * earth_radius**2 / (z + earth_radius) ** 2
#
#
# def slant_depth_inverse_jac(
#     y,
#     theta_tr,
#     rho=us_std_atm_density,
#     rhoprime=us_std_atm_density_deriv,
#     Re=const.earth_radius,
# ):
#     theta_tr = np.asarray(theta_tr)
#
#     fy = rho(y, Re)
#     fpy = rhoprime(y, rho, Re)
#
#     val = Re**2 * fy * np.sin(theta_tr) ** 2
#     val -= (
#         0.5
#         * (Re + y)
#         * fpy
#         * (Re**2 * np.cos(2 * theta_tr) + Re**2 + 4 * Re * y + 2 * y**2)
#     )
#     val /= (
#         fy**2
#         * (Re + y) ** 2
#         * np.sqrt(Re**2 * np.cos(theta_tr) ** 2 + y**2 + 2 * y * Re + 1e-4)
#     )
#     return 1e-5 * val
