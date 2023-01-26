# The Clear BSD License
#
# Copyright (c) 2023 Alexander Reustle and the NuSpaceSim Team
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
Upware Going Extensive Air Shower calculations and properties.

author: Alexander Reustle
date: 2023 January 23
"""

import numpy as np
from scipy.optimize import newton
from .atmospheric_models import us_std_atm_density


def viewing_angle(beta_tr, Zdet, Re=6378.1):
    return np.arcsin((Re / (Re + Zdet)) * np.cos(beta_tr))


def propagation_angle(beta_tr, z, Re=6378.1):
    return np.arccos((Re / (Re + z)) * np.cos(beta_tr))


# def length_along_prop_axis(z_start, z_stop, beta_tr, Re=6378.1):
#     L1 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_stop + z_stop ** 2
#     L2 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_start + z_start ** 2
#     L = np.sqrt(L1) - np.sqrt(L2)
#     return L


def path_length_tau_atm(z, beta_tr, Re=6378.1, xp=np):
    """
    From https://arxiv.org/pdf/1902.11287.pdf Eqn (11)
    """
    Resinb = Re * xp.sin(beta_tr)
    return xp.sqrt(Resinb ** 2 + (Re + z) ** 2 - Re ** 2) - Resinb


# def altitude_along_prop_axis(L, z_start, beta_tr, Re=6378.1):
#     r1 = Re ** 2
#     r2 = 2 * Re * z_start
#     r3 = z_start ** 2
#     return -Re + np.sqrt(
#         L ** 2 + 2 * L * np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3) + r1 + r2 + r3
#     )


def altitude_along_path_length(s, beta_tr, Re=6378.1, xp=np):
    """Derived by solving for z in path_length_tau_atm."""
    # (s + B)**2 = B**2 + (Re + z)**2 - Re**2
    # (s + B)**2 - B**2 + Re**2 = (Re + z)**2
    # s**2 + 2sB + B**2 - B**2 + Re**2 = (Re + z)**2
    # s**2 + 2sB + Re**2 = (Re + z)**2
    # Re + z = sqrt(s**2 + 2sB + Re**2)
    # z = sqrt(s**2 + 2sB + Re**2) - Re
    return xp.sqrt(s ** 2 + 2.0 * s * Re * xp.sin(beta_tr) + Re ** 2) - Re


# def gain_in_altitude_along_prop_axis(L, z_start, beta_tr, Re=6378.1):
#     return altitude_along_prop_axis(L, z_start, beta_tr, Re) - z_start


def distance_to_detector(beta_tr, z, z_det, earth_radius=6378.1):
    """Unsure how I derived this."""
    theta_view = viewing_angle(beta_tr, z_det, earth_radius)
    theta_prop = propagation_angle(beta_tr, z, earth_radius)
    ang_e = np.pi / 2 - theta_view - theta_prop
    return np.sin(ang_e) / np.sin(theta_view) * (z + earth_radius)


def index_of_refraction_air(X_v):
    r"""Index of refraction in air (Nair)

    Index of refraction as a function of vertical atmospheric depth x_v (g/cm^2)

    Hillas 1475 eqn (2)
    """
    temperature = 204.0 + 0.091 * X_v
    n = 1.0 + 0.000296 * (X_v / 1032.9414) * (273.2 / temperature)
    return n


def rad_len_atm_depth(x, L0=36.66):
    """
    T is X / L0, units of radiation length scaled g/cm^2
    """
    T = x / L0
    return T


def shower_age(T, param_beta=np.log(10 ** 8 / (0.710 / 8.36))):
    r"""Shower age (s) as a function of atmospheric depth in mass units (g/cm^2)


    Hillas 1475 eqn (1)

    s = 3 * T / (T + 2 * beta)
    """
    return 3.0 * T / (T + 2.0 * param_beta)


def greisen_particle_count(T, s, param_beta=np.log(10 ** 8 / (0.710 / 8.36))):
    r"""Particle count as a function of radiation length from atmospheric depth

    Hillas 1461 eqn (6)

    N_e(T) where y is beta in EASCherGen
    """
    N_e = (0.31 / np.sqrt(param_beta)) * np.exp(T * (1.0 - 1.5 * np.log(s)))
    N_e[N_e < 0] = 0.0
    return N_e


def shower_age_of_greisen_particle_count(target_count, x0=2):

    # for target_count = 2, shower_age = 1.899901462640018

    param_beta = np.log(10 ** 8 / (0.710 / 8.36))

    def rns(s):
        return (
            0.31
            * np.exp((2 * param_beta * s * (1.5 * np.log(s) - 1)) / (s - 3))
            / np.sqrt(param_beta)
            - target_count
        )

    return newton(rns, x0)


def slant_depth_trig_approx(z_lo, z_hi, theta_tr, z_max=100.0):

    rho = us_std_atm_density
    r0 = rho(0)
    ptan = 92.64363150999402 * np.tan(theta_tr) + 101.4463720303218

    def approx_slant_depth(z):
        return ptan * (8.398443922535177 + r0 - 6340.6095008383245 * rho(z))

    fmax = approx_slant_depth(z_max)
    sd_hi = np.where(z_hi >= z_max, fmax, approx_slant_depth(z_hi))
    sd_lo = np.where(z_lo >= z_max, fmax, approx_slant_depth(z_lo))

    return sd_hi - sd_lo


def slant_depth_trig_behind_ahead(z_lo, z, z_hi, theta_tr, z_max=100.0):

    rho = us_std_atm_density
    r0 = rho(0)
    ptan = 92.64363150999402 * np.tan(theta_tr) + 101.4463720303218

    def approx_slant_depth(z):
        return ptan * (8.398443922535177 + r0 - 6340.6095008383245 * rho(z))

    fmax = approx_slant_depth(z_max)
    sd_hi = np.where(z_hi >= z_max, fmax, approx_slant_depth(z_hi))
    sd_mid = np.where(z >= z_max, fmax, approx_slant_depth(z))
    sd_lo = np.where(z_lo >= z_max, fmax, approx_slant_depth(z_lo))

    return sd_mid - sd_lo, sd_hi - sd_mid


def altitude_at_shower_age(s, alt_dec, beta_tr, z_max=65.0, **kwargs):
    """Altitude as a function of shower age, decay altitude and emergence angle."""

    alt_dec = np.asarray(alt_dec)
    beta_tr = np.asarray(beta_tr)

    theta_tr = 0.5 * np.pi - beta_tr
    param_beta = np.log(10 ** 8 / (0.710 / 8.36))

    # Check that shower age is within bounds
    ss = shower_age(
        rad_len_atm_depth(slant_depth_trig_approx(alt_dec, z_max, theta_tr))
    )
    mask = ss < s

    X_s = -1.222e19 * param_beta * s / ((10.0 / 6.0) * 1e17 * s - 5e17)

    def ff(z):
        X = slant_depth_trig_approx(alt_dec[~mask], z, theta_tr[~mask])
        return X - X_s

    altitude = np.full_like(alt_dec, z_max)
    altitude[~mask] = newton(ff, alt_dec[~mask], **kwargs)

    return altitude


def track_length(s, E):
    r"""Differential Track Length in radiation lengths.

    Hillas 1461 eqn (8) variable T(E) =
    (Total track length of all charged particles with kinetic energy > E)
    /
    (Total vertical component of tracks of all charged particles).

    The total track length of particles of energy > E in vertical thickness interval dx
    of the shower is N_e*T(E)*dx.

    """

    E0 = np.where(s >= 0.4, 44.0 - 17.0 * (s - 1.46), 26.0)
    "Kinetic Energy charged primary of shower particles (MeV)"

    return ((0.89 * E0 - 1.2) / (E0 + E)) ** s * (1.0 + 1.0e-4 * s * E) ** -2


def hillas_dndu(energy, theta, s):

    e2hill = 1150.0 + 454.0 * np.log(s)
    mask = e2hill > 0
    vhill = energy[mask] / e2hill[mask]
    whill = 2.0 * (1.0 - np.cos(theta[mask])) * ((energy[mask] / 21.0) ** 2)
    w_ave = (
        0.0054 * energy[mask] * (1.0 + vhill) / (1.0 + 13.0 * vhill + 8.3 * vhill ** 2)
    )
    uhill = whill / w_ave

    zhill = np.sqrt(uhill)
    a2hill = np.where(zhill < 0.59, 0.478, 0.380)
    sv2 = 0.777 * np.exp(-(zhill - 0.59) / a2hill)
    rval = np.zeros_like(e2hill)
    rval[mask] = sv2
    return rval
