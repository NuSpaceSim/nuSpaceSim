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

from .atmospheric_models import refractive_index


def propagation_angle(beta_tr, z, Re=6371.0):
    return np.arccos((Re / (Re + z)) * np.cos(beta_tr))


propagation_theta = propagation_angle


def rad_len_atm_depth(x, L0recip=0.02727768685):
    """
    T is X / L0, units of radiation length scaled g/cm^2
    default L0 = 36.66
    """
    return x * L0recip


def shower_age(T):
    r"""Shower age (s) as a function of atmospheric depth in mass units (g/cm^2)


    Hillas 1475 eqn (1)

    s = 3 * T / (T + 2 * beta)
    where from EASCherGen beta = ln(10 ** 8 / (0.710 / 8.36))
    so 2 * beta = 41.773258959991503347439824715462431074518643532553348404286170740...
    """
    return 3.0 * T / (T + 41.77325895999150334743982471)


def greisen_particle_count(t, s, greisen_beta, mask, *args, dtype=np.float32, **kwargs):
    r"""Particle count as a function of radiation length from atmospheric depth

    Hillas 1461 eqn (6)

    N_e(T) where y is beta in EASCherGen, thus
    (0.31 / sqrt (10^8 / (0.710 / 8.36)))
    = 0.0678308895484773316048795658058110209448440898800928880798622962...
    """

    # , param_beta=np.log(10 ** 8 / (0.710 / 8.36))
    # N_e = (0.31 / np.sqrt(param_beta)) * np.exp(T * (1.0 - 1.5 * np.log(s)))
    # N_e[N_e < 0] = 0.0
    alpha = dtype(0.31) / np.sqrt(greisen_beta, dtype=dtype)

    RN = alpha * np.exp(
        t * (dtype(1) - dtype(1.5) * np.log(s, dtype=dtype)), dtype=dtype
    )
    RN[RN < 0] = 0.0
    return RN, mask


def gaisser_hillas_particle_count_exp_form(
    gramsum, X0, Xmax, Nmax, gh_lam, *args, dtype=np.float32, **kwargs
):
    # Parametric Form Parameters
    x = (gramsum - X0) / gh_lam
    m = (Xmax - X0) / gh_lam
    return Nmax * np.exp(m * (np.log(x) - np.log(m)) - (x - m))


def particle_count_parameterized_gaisser_hillas(
    gramsum, Eshow, *args, mask, dtype=np.float32, **kwargs
):
    """
    Shower particle count from Gaisser Hillas formula with static parameters.
    """

    # Nuclear Collision length in Air from PDG.
    # From https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/air_dry_1_atm.html
    X0 = 61.3
    gramsum_mask = gramsum > X0
    mask &= gramsum_mask
    Xmask = gramsum[gramsum_mask]
    Xm = 739.0
    # 65.12 g/cm^2 value obtained by evaluating lambda at Xmax for 1000 upward pion 10^17 eV EAS at 5 deg Earth-emergence angle and starting at sea level
    gh_lam = 65.12
    Nmax = 0.045 * (1.0 + 0.0217 * np.log(Eshow / 1.0e5)) * Eshow / 0.074
    XmaxOff = 58.0 * np.log10(Eshow / 1.0e8)
    Xmax = Xm + XmaxOff

    particle_count = gaisser_hillas_particle_count_exp_form(
        Xmask, X0, Xmax, Nmax, gh_lam
    )

    return particle_count, mask


def particle_count_fluctuated_gaisser_hillas(
    CONEX_table, gramsum, Eshow, mask, *args, dtype=np.float32, **kwargs
):
    """
    Shower particle count from Gaisser Hillas formula with fluctuated parameters.
    """
    # Gaisser Hillas Values from CONEX File
    idx = np.random.randint(low=0, high=CONEX_table.shape[0])
    Nm, Xm, X0, p1, p2, p3 = CONEX_table[idx]

    # Masking Gramsum
    gramsum_mask = gramsum > (0.0 if X0 < 0.0 else X0)
    mask &= gramsum_mask
    Xmask = gramsum[gramsum_mask]

    # JFK : put in form from Tom Gaisser's book, pg: 238 - 239
    Nmax100 = 6.99e7
    NmaxE = 0.045 * (1.0 + 0.0217 * np.log(Eshow / 1.0e5)) * Eshow / 0.074
    Nmax = Nm * NmaxE / Nmax100

    # the following from Gaisser leads to ~80 g/cm^2/decade elongation rate
    # DOI:10.1103/RevModPhys.83.907, Letessier-Selvon & Stanev
    #  gives in Egn 3 ~ 85 g/cm^2/decade is for EM showers
    #    Xmax = 36. * np.log(Eshow/0.074)
    # HiRes Measurement:  R. U. Abbasi et al 2005 ApJ 622 910 : gives ~ 56 g/cm^2, Auger ~60 g/cm^2
    # DOI:10.1103/RevModPhys.83.907, Letessier-Selvon & Stanev gives in Egn 7 ~ 62 g/cm^2/decade
    #   use a single 58 g/cm^2 per decade energy addition/subtraction,
    #   assume using only 100 PeV energy file for this to be correct
    XmaxOff = 58.0 * np.log10(Eshow / 1.0e8)
    Xmax = Xm + XmaxOff

    gh_lam = p1 + p2 * Xmask + p3 * Xmask * Xmask
    gh_lam[gh_lam > 100.0] = 100.0
    gh_lam[gh_lam < 1.0e-5] = 1.0e-5

    particle_count = gaisser_hillas_particle_count_exp_form(
        Xmask, X0, Xmax, Nmax, gh_lam
    )

    return particle_count, mask


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


def greisen_particle_count_v2(X, Eprim, Ecrit=0.08492822966507177):
    r"""Particle count as a function of radiation length from atmospheric depth

    Hillas 1461 eqn (6)

    N_e(T) where y is beta in EASCherGen, thus
    (0.31 / sqrt (10^8 / (0.710 / 8.36)))
    = 0.0678308895484773316048795658058110209448440898800928880798622962...
    """
    T = rad_len_atm_depth(X)
    y = np.atleast_1d(np.log(Eprim / Ecrit))
    s = 3.0 / (1.0 + 2.0 * y / T)
    N_e = (0.31 / np.sqrt(y)) * np.exp(T * (1.0 - 1.5 * np.log(s)))
    return np.squeeze(N_e)


def estimate_slant_depth_linear(Eprim):
    """
    Returns the approximate slant depth X (g/cm^2) where the
    Greisen particle count drops to 1.

    Fit derived from:
      E=1e8  -> X=2644.78
      E=1e12 -> X=3882.56
    """
    slope = 309.4446  # (3882.56 - 2644.78) / (12 - 8)
    return slope * np.log10(Eprim) + 169.227


def slant_depth_of_greisen_particle_count(
    target_count, Eprim, Ecrit=0.08492822966507177, X0=None
):
    # for target_count = 1 & Eprim=1e8,  X = 2644.7846080608188
    # for target_count = 1 & Eprim=1e12, X = 3882.563748098721

    X0 = X0 if X0 is not None else estimate_slant_depth_linear(Eprim)

    def f(X):
        return greisen_particle_count_v2(X, Eprim, Ecrit) - target_count

    return newton(f, X0)


def shower_age_at_depth(X_to_node, gb):
    """Atmospheric depth parameter t and shower age s.

    Parameters
    ----------
    X_to_node : ndarray, shape (..., n_nodes)
        Cumulative slant depth to each node (g/cm²).
    gb : ndarray, shape (n_showers,)
        Greisen beta = log(Eshow / ecrit).

    Returns
    -------
    t : ndarray, shape (..., n_nodes)
    s : ndarray, shape (..., n_nodes)
    """
    t = X_to_node / 36.66
    s = 3.0 * t / (t + 2.0 * np.atleast_1d(gb)[..., None])
    return t, s


def greisen_particles(t, s, gb):
    """Greisen particle count at each node.

    Parameters
    ----------
    t, s : ndarray, shape (..., n_nodes)
    gb : ndarray, shape (n_showers,)

    Returns
    -------
    RN : ndarray, shape (..., n_nodes)
    """
    alpha = 0.31 / np.sqrt(np.atleast_1d(gb))
    safe_s = np.where(s > 0, s, 1.0)
    RN = alpha[..., None] * np.exp(t * (1.0 - 1.5 * np.log(safe_s)))
    return np.maximum(RN, 0.0)


def hillas_e2(s):
    """Hillas E-squared parameter e2hill from shower age."""
    safe_s = np.where(s > 0, s, 1.0)
    return np.where(s > 0, 1150.0 + 454.0 * np.log(safe_s), 0.0)


def shower_state_at_nodes(z_nodes, X_to_node, gb):
    """Per-node shower fields from the propagation grid.

    Returns
    -------
    AirN : refractive index of air at each node
    s : shower age
    RN : Greisen particle count
    e2hill : Hillas E² parameter
    """
    AirN = refractive_index(z_nodes)
    t, s = shower_age_at_depth(X_to_node, gb)
    RN = greisen_particles(t, s, gb)
    e2hill = hillas_e2(s)
    return AirN, s, RN, e2hill
