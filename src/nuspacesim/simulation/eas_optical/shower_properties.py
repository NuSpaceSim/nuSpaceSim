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
Upward Going Extensive Air Shower calculations and properties.

author: Alexander Reustle
date: 2023 January 23
"""

import numpy as np
from scipy.optimize import newton

from . import atmospheric_models as atm

# from .atmospheric_models import (
#     cherenkov_photons_created,
#     ozone_losses,
#     elterman_mie_aerosol_scatter,
#     sokolsky_rayleigh_scatter,
#     slant_depth_trig_approx,
#     # slant_depth_trig_behind_ahead,
#     rad_len_atm_depth
# )

__all__ = [
    "propagation_angle",
    "path_length_tau_atm",
    "altitude_along_path_length",
    "shower_age",
    "greisen_particle_count",
    "shower_age_of_greisen_particle_count",
    "gaisser_hillas_particle_count",
]


def propagation_angle(beta_tr, z, Re=6378.1):
    return np.arccos((Re / (Re + z)) * np.cos(beta_tr))


def path_length_tau_atm(z, beta_tr, Re=6378.1, xp=np):
    """
    From https://arxiv.org/pdf/1902.11287.pdf Eqn (11)
    """
    Resinb = Re * xp.sin(beta_tr)
    return xp.sqrt(Resinb**2 + (Re + z) ** 2 - Re**2) - Resinb


def altitude_along_path_length(path_len, beta_tr, Re=6378.1, xp=np):
    """Derived by solving for z in path_length_tau_atm."""
    return xp.sqrt(path_len**2 + 2.0 * path_len * Re * xp.sin(beta_tr) + Re**2) - Re


def shower_age(T):
    r"""Shower age (s) as a function of atmospheric depth in mass units (g/cm^2)


    Hillas 1475 eqn (1)

    s = 3 * T / (T + 2 * beta)
    where from EASCherGen beta = ln(10 ** 8 / (0.710 / 8.36))
    so 2 * beta = 41.773258959991503347439824715462431074518643532553348404286170740...
    """
    return 3.0 * T / (T + 41.77325895999150334743982471)


def greisen_particle_count(T, s):
    r"""Particle count as a function of radiation length from atmospheric depth

    Hillas 1461 eqn (6)

    N_e(T) where y is beta in EASCherGen, thus
    (0.31 / sqrt (10^8 / (0.710 / 8.36)))
    = 0.0678308895484773316048795658058110209448440898800928880798622962...
    """
    # , param_beta=np.log(10 ** 8 / (0.710 / 8.36))
    # N_e = (0.31 / np.sqrt(param_beta)) * np.exp(T * (1.0 - 1.5 * np.log(s)))
    # N_e[N_e < 0] = 0.0
    N_e = 0.067830889548477331 * np.exp(T * (1.0 - 1.5 * np.log(s)))
    N_e[N_e < 0] = 0.0
    return N_e


def greisen_particle_count_shower_age(slant_depth):
    r"""greisen_particle_count and shower_age in (g/cm^2)."""
    T = atm.rad_len_atm_depth(slant_depth)
    s = shower_age(T)
    RN = greisen_particle_count(T, s)
    return RN, s


def shower_age_of_greisen_particle_count(target_count, x0=2):
    # for target_count = 2, shower_age = 1.899901462640018
    # param_beta = np.log(10 ** 8 / (0.710 / 8.36))

    def rns(s):
        return (
            0.067830889548477331
            * np.exp(
                (41.77325895999150334743982471 * s * (1.5 * np.log(s) - 1)) / (s - 3.0)
            )
            - target_count
        )

    return newton(rns, x0)


def gaisser_hillas_particle_count(X, Nmax, X0, Xmax, invlam):
    # return ((X - X0) / (Xmax - X0)) ** xmax * np.exp((Xmax - X) * invlam)
    xmax = (Xmax - X0) * invlam
    x = (X - X0) * invlam
    return Nmax * (x / xmax) ** xmax * np.exp(xmax - x)


def altitude_at_shower_age(s, alt_dec, beta_tr, z_max=65.0, **kwargs):
    """Altitude as a function of shower age, decay altitude and emergence angle."""

    alt_dec = np.asarray(alt_dec)
    beta_tr = np.asarray(beta_tr)

    theta_tr = 0.5 * np.pi - beta_tr
    param_beta = np.log(10**8 / (0.710 / 8.36))

    # Check that shower age is within bounds
    ss = shower_age(
        atm.rad_len_atm_depth(atm.slant_depth_trig_approx(alt_dec, z_max, theta_tr))
    )
    mask = ss < s

    X_s = -1.222e19 * param_beta * s / ((10.0 / 6.0) * 1e17 * s - 5e17)

    def f(z):
        X = atm.slant_depth_trig_approx(alt_dec[~mask], z, theta_tr[~mask])
        return X - X_s

    altitude = np.full_like(alt_dec, z_max)
    altitude[~mask] = newton(f, alt_dec[~mask], **kwargs)

    return altitude


def cherenkov_yield(w, detector_altitude, z, beta_tr, thetaC, X_ahead):
    """Differential proportion of photons generated and not scattered."""
    return (
        atm.cherenkov_photons_created(w, thetaC)
        * atm.ozone_losses(w, z, detector_altitude, 0.5 * np.pi - beta_tr)
        * atm.elterman_mie_aerosol_scatter(w, z, propagation_angle(beta_tr, z))
        * atm.sokolsky_rayleigh_scatter(w, X_ahead)
    )


def differential_track_length(E, s):
    return E * s


def track_length(E, s):
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

    return ((0.89 * E0 - 1.2) / (E0 + E)) ** s * (1.0 + 1e-4 * s * E) ** -2


def hillas_dndu(energy, costheta, shower_age):
    """
    Hillas 1461 ss 2.4: angular distribution of charged particles.

    Internal Variables:
    w: Eqn (9). Convenience variable describing angles (theta) from shower axis.

    w_ave: Eqn (12). Expected value of w.

    dndu: Eqn (13).
    """

    w = 2.0 * (1.0 - costheta) * ((1.0 / 21.0) * energy) ** 2
    v = energy * np.reciprocal(1150.0 + 454.0 * np.log(shower_age))
    w_ave = 0.0054 * energy * (1.0 + v) / (1.0 + 13.0 * v + 8.3 * v**2)
    z_hill = np.sqrt(w / w_ave)
    lam2 = np.where(z_hill < 0.59, 0.478, 0.380)
    dndu = 0.777 * np.exp(-(z_hill - 0.59) / lam2)
    return dndu


def cherenkov_cone_particle_count_integrand(logenergy, costheta, shower_age):
    energy = 10.0**logenergy
    dTdE = differential_track_length(energy, shower_age)
    dndu = hillas_dndu(energy, costheta, shower_age)
    return dTdE * dndu
