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

from ...utils.ode import dormand_prince_rk54
from . import atmospheric_models as atm

# from scipy.integrate import solve_ivp


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
    "altitude_along_path_length",
    "shower_age",
    "greisen_particle_count",
    "shower_age_of_greisen_particle_count",
    "gaisser_hillas_particle_count",
]


def propagation_angle(beta_tr, z, Re=6378.1):
    return np.arccos((Re / (Re + z)) * np.cos(beta_tr))


def path_length_tau_atm_theta(z, theta_tr, Re=6378.1):
    """
    From https://arxiv.org/pdf/1902.11287.pdf Eqn (11)
    """
    Recost = Re * np.cos(theta_tr)
    return np.sqrt(Recost**2 + (Re + z) ** 2 - Re**2) - Recost


def path_length_tau_atm_beta(z, beta_tr, Re=6378.1):
    """
    From https://arxiv.org/pdf/1902.11287.pdf Eqn (11)
    """
    Resinb = Re * np.sin(beta_tr)
    return np.sqrt(Resinb**2 + ((Re + z) ** 2 - Re**2)) - Resinb


def altitude_along_path_length(plen, beta_tr, Re=6378.1):
    """Simple application of cosine rule."""
    return np.sqrt(plen**2 + 2.0 * plen * Re * np.sin(beta_tr) + Re**2) - Re


def greisen_beta(
    shower_energy,
    critical_energy=0.0849282296650717703349282296650717703349282296650717703349282296,
):
    r"""Greisen beta (y in hillas 1461) parameter from Energy of Primary and
    Critical energy (in GeV) is
    0.710 / 8.36 == 0.0849282296650717703349282296650717703349282296650717703349282296
    """
    return np.log(shower_energy / critical_energy)


def greisen_beta_opt(shower_energy):
    r"""Greisen beta (or y in hillas 1461) parameter from Energy of Primary
    beta = ln(Eprim / Ecrit)
         = ln(Eprim) - ln(Ecrit)
         = ln(Eprim) - ln(Ecrit)
         = ln(Eprim) + (-ln(Ecrit))
    Where ln(Ecrit) == -2.465948736043386201575980720256301876450509857246490393876
    """
    beta = np.log(shower_energy) + 2.46594873604338620157598072025630187645050985724649
    beta[beta < 0] = 0  # beta < 0 should not be possible. Is this Necessary?
    return beta


def shower_age(T, greisen_param):
    r"""Shower age (s) as a function of atmospheric depth in mass units (g/cm^2)
    Hillas 1461 eqn (7)
    s = 3 / (1 + 2 * (greisen_param/T))
    where from EASCherGen greisen_param = ln(10 ** 8 / (0.710 / 8.36))
    so 2 * greisen_param = 41.773258959991503347439824715462431074518643532553348404286170740...
    """
    return 3.0 / (1.0 + 2.0 * (greisen_param / T))


def greisen_particle_count(T, s, greisen_param):
    r"""Particle count as a function of radiation length from atmospheric depth
    Hillas 1461 eqn (6)
    N_e(T) where y is greisen_param in EASCherGen, thus (0.31 / sqrt(greisen_param))
    """
    N_e = (0.31 / np.sqrt(greisen_param)) * np.exp(T * (1.0 - 1.5 * np.log(s)))
    N_e[N_e < 0] = 0.0  # Also shouldn't be possible
    return N_e


def greisen_particle_count_shower_age(slant_depth, shower_energy):
    r"""greisen_particle_count and shower_age in (g/cm^2)."""
    T = atm.rad_len_atm_depth(slant_depth)
    greisen_param = greisen_beta_opt(shower_energy)
    s = shower_age(T, greisen_param)
    RN = greisen_particle_count(T, s, greisen_param)
    return RN, s


def shower_age_of_greisen_particle_count(target_count, greisen_param, s0=2):
    def rns(s):
        T = 2.0 * greisen_param / (3.0 / s - 1)
        count = 0.31 / np.sqrt(greisen_param) * np.exp(T * (1 - 1.5 * np.log(s)))
        return count - target_count

    return newton(rns, s0)


def gaisser_hillas_particle_count(X, Nmax, X0, Xmax, invlam):
    # return ((X - X0) / (Xmax - X0)) ** xmax * np.exp((Xmax - X) * invlam)
    xmax = (Xmax - X0) * invlam
    x = (X - X0) * invlam
    return Nmax * (x / xmax) ** xmax * np.exp(xmax - x)


def slant_depth_range_from_shower_age(
    shower_age_begin,
    shower_age_end,
    decay_altitude,
    theta_tr,
    greisen_param,
    z_max=65.0,
):
    X_pmax = atm.slant_depth_us(
        decay_altitude,
        np.full_like(decay_altitude, z_max),
        theta_tr,
        rtol=1e-5,
        atol=1e-8,
        itermax=32,
    )

    X_low = (73.32 * greisen_param) / ((3.0 / shower_age_begin) - 1)
    X_low_bound = np.where(X_low > X_pmax, X_pmax, X_low)
    X_hi = (73.32 * greisen_param) / ((3.0 / shower_age_end) - 1)
    X_hi_bound = np.where(X_hi > X_pmax, X_pmax, X_hi)

    return X_low_bound, X_hi_bound


def altitude_range_shower_visible(X_low, X_hi, decay_altitude, theta_tr):
    """Altitude as a function of shower age, decay altitude and zenith angle."""

    z_range = dormand_prince_rk54(
        lambda _, y, theta: atm.slant_depth_inverse_func(y, theta),
        np.stack((np.zeros_like(X_hi), X_hi)),
        decay_altitude,
        theta_tr,
        t_eval=[X_low, X_hi],
        rtol=1e-8,
        atol=1e-6,
    )

    return z_range


def cherenkov_yield(w, detector_altitude, z, beta_tr, thetaC, X_ahead):
    """Differential proportion of photons generated and not scattered."""
    return (
        atm.cherenkov_photons_created(w, thetaC)
        * atm.ozone_losses(w, z, detector_altitude, 0.5 * np.pi - beta_tr)
        * atm.elterman_mie_aerosol_scatter(w, z, propagation_angle(beta_tr, z))
        * atm.sokolsky_rayleigh_scatter(w, X_ahead)
    )


def track_length(E, s):
    r"""Track Length in radiation lengths.

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


def differential_track_length(E, s):
    # -(100000000 s ((0.89 E_0 - 1.2)/(E_0 + x))^s (2 E_0 + (s + 2) x + 10000))/((E_0 + x) (s x + 10000)^3)
    # return E * s
    E0 = np.where(s >= 0.4, 44.0 - 17.0 * (s - 1.46), 26.0)
    "Kinetic Energy charged primary of shower particles (MeV)"

    return -(
        1e8 * s * ((0.89 * E0 - 1.2) / (E0 + E)) ** s * (2.0 * E0 + (s + 2.0) * E + 1e4)
    ) / ((E0 + E) * (s * E + 1e-4) ** 3)


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
    # dTdE = differential_track_length(energy, shower_age)
    dndu = hillas_dndu(energy, costheta, shower_age)
    # return dTdE * dndu
    return dndu
