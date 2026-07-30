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

# Re-exported for backward compatibility: the US Standard Atmosphere density
# now has a single implementation in propagation (validated against the former
# local copy to <2e-7 relative; the only differences were rounding of the
# base-pressure table and a searchsorted wraparound for z < 0).
from .propagation import lexpr, us_std_atm_density

__all__ = [
    "us_std_atm_density",
    "ozone_losses",
    "ozone_rate",
    "ozone_column",
    "aerosol_column",
    "refractive_index",
    "OZONE_SEGMENT_BOUNDS",
    "OZONE_SEGMENT_RATES",
    "AEROSOL_SEGMENT_BOUNDS",
    "AEROSOL_EXTINCTION",
]

_F32 = np.float32

# -- Ozone profile tables (atm-cm). Piecewise-linear cumulative column TotZon
#    over the OZ_ZETA altitude breakpoints; heritage EASCherGen values. --------
OZ_ZETA = np.array(
    [5.35, 10.2, 14.75, 19.15, 23.55, 28.1, 32.8, 37.7, 42.85, 48.25, 100.0],
    dtype=_F32,
)
OZ_DEPTH = np.array(
    [15.0, 9.0, 10.0, 31.0, 71.0, 87.2, 57.0, 29.4, 10.9, 3.2, 1.3], dtype=_F32
)
OZ_DSUM = np.array(
    [310.0, 301.0, 291.0, 260.0, 189.0, 101.8, 44.8, 15.4, 4.5, 1.3, 0.1],
    dtype=_F32,
)

# -- Aerosol vertical optical depth at 55 km visibility, per km altitude band
#    (aOD55). Heritage EASCherGen values. ------------------------------------
# fmt: off
AOD55 = np.array(
    [
        0.250, 0.136, 0.086, 0.065, 0.055, 0.049, 0.045, 0.042, 0.038, 0.035,
        0.032, 0.029, 0.026, 0.023, 0.020, 0.017, 0.015, 0.012, 0.010, 0.007,
        0.006, 0.004, 0.003, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001,
    ],
    dtype=_F32,
)
# fmt: on


def ozone_losses(z):
    """Cumulative ozone column TotZon (atm-cm) above altitude z (km)."""
    msk1 = z < 5.35
    TotZon = np.empty_like(z)
    TotZon[msk1] = 310 + ((5.35 - z[msk1]) / 5.35) * 15
    msk2 = z >= 100
    TotZon[msk2] = 0.1

    msk3 = ~msk1 & ~msk2
    idxs = np.searchsorted(OZ_ZETA, z[msk3])
    TotZon[msk3] = (
        OZ_DSUM[idxs]
        + ((OZ_ZETA[idxs] - z[msk3]) / (OZ_ZETA[idxs] - OZ_ZETA[idxs - 1]))
        * OZ_DEPTH[idxs]
    )
    return TotZon


def ozone_rate(z):
    """Analytic ozone-column slope -dTotZon/dz (atm-cm per km), z in km.

    ``ozone_losses`` (TotZon) is piecewise-linear in z, so its derivative
    is piecewise-constant -- the exact within-segment slope using the same
    searchsorted('left') segment convention. Replaces the old central
    finite difference (two ozone_losses evaluations + 2*searchsorted),
    which only smeared the breakpoints.
    """
    # Dense form: one searchsorted over all z + np.where selection, instead
    # of boolean-mask scatter (which gathered/wrote the large (n,k,n_slant_sub)
    # array element-wise). idx is clipped to [1, len-1] so the segment-slope
    # gather is always in-bounds; the values it produces outside [5.35, 100)
    # are discarded by the np.where masks below.
    idx = np.clip(np.searchsorted(OZ_ZETA, z), 1, len(OZ_ZETA) - 1)
    seg = OZ_DEPTH[idx] / (OZ_ZETA[idx] - OZ_ZETA[idx - 1])
    rate = np.where(z < 5.35, 15.0 / 5.35, seg)
    rate[z >= 100.0] = 0.0
    return rate


# -- Exact layer-walk tables derived from the profiles above. -----------------
# The ozone column is the path integral of the *piecewise-constant* ozone rate
# (-dTotZon/dz), so it has an exact analytic layer-walk over the ozone altitude
# breakpoints -- no GL sub-quadrature, no per-node searchsorted (and exact
# where 8-point GL was ~5% off at the rate discontinuities). 12 altitude
# boundaries ``[0, *OZ_ZETA]`` and 11 per-segment constant rates (``ozone_rate``
# at each segment midpoint; the rate is constant across a segment, so the
# midpoint is exact). See :func:`ozone_column`.
OZONE_SEGMENT_BOUNDS = np.concatenate([[0.0], np.asarray(OZ_ZETA, np.float64)])
OZONE_SEGMENT_RATES = ozone_rate(
    0.5 * (OZONE_SEGMENT_BOUNDS[:-1] + OZONE_SEGMENT_BOUNDS[1:])
)

# Aerosol layer-walk table. The per-km OD difference of AOD55 is a piecewise-
# constant extinction per km -- the same structure as the ozone rate. So the
# *slant* aerosol column has an exact analytic curved-atmosphere walk (see
# :func:`aerosol_column`), replacing the former plane-parallel
# ``vertical_OD / cos(theta)`` (which diverges at grazing incidence).
# 31 altitude boundaries [0, 1, ..., 30] km, 30 per-km extinctions.
_dfaOD55 = [_F32(AOD55[i] - AOD55[i + 1]) for i in range(29)]
_dfaOD55.append(_F32(0))
AEROSOL_SEGMENT_BOUNDS = np.arange(31.0)
AEROSOL_EXTINCTION = np.asarray(np.array(_dfaOD55, dtype=_F32), dtype=np.float64)


def ozone_column(L_nodes, L_max, betaE, oz_zbnd, oz_seg_rate, R):
    """Analytic ozone column from each node to L_max (exact layer-walk).

    The ozone rate ``-dTotZon/dz`` is piecewise-constant in altitude, so the
    slant ozone column ``integral rate(z(L)) dL`` from a node to the visible top
    is just the sum, over the ozone altitude segments, of each segment's constant
    rate times the path length the ray spends in it. That path length is
    ``L(z_top_seg) - L(z_bot_seg)`` (via :func:`lexpr`) clamped to the node->top
    interval. Exact (no quadrature) and free of the per-node ``searchsorted`` and
    the ``(n, k, n_slant_sub)`` GL temporaries the former sub-quadrature needed --
    and more accurate, since Gauss-Legendre converged poorly across the rate's
    discontinuities (~5% median, ~30% max error at 8 nodes).

    Parameters
    ----------
    L_nodes : ndarray, shape (n_showers, n_nodes)
    L_max : ndarray, shape (n_showers,)
    betaE : ndarray, shape (n_showers,)
    oz_zbnd : ndarray, shape (n_seg + 1,)
        Ascending ozone altitude boundaries (km), ``[0, *OzZeta]``.
    oz_seg_rate : ndarray, shape (n_seg,)
        Constant ozone rate within each ``[oz_zbnd[j], oz_zbnd[j+1]]`` segment.
    R : float
        Earth radius (km).

    Returns
    -------
    ZonZ : ndarray, shape (n_showers, n_nodes)
    """
    # Propagation length of each ozone boundary, clamped into each ray's
    # [L_node, L_max] window: a segment outside the window contributes zero.
    Lb = lexpr(oz_zbnd[:, None], betaE[None, :], R=R)  # (n_seg+1, n_showers)
    Lc = np.clip(Lb[:, :, None], L_nodes[None], L_max[None, :, None])
    ZonZ = np.einsum("s,snk->nk", oz_seg_rate, Lc[1:] - Lc[:-1])
    # An ozone column is non-negative; a degenerate (decay-above-atmosphere)
    # shower has L_node >= L_max so every segment clamps to zero width -> 0.
    return np.maximum(ZonZ, 0.0)


def aerosol_column(L_nodes, L_max, betaE, aero_zbnd, aero_ext, R):
    """Slant aerosol optical depth from each node to the top of the aerosol layer.

    Identical in form to :func:`ozone_column`: the aerosol extinction
    (``dfaOD55``, the per-km vertical optical depth) is piecewise-constant in
    altitude, so the slant column ``integral ext(z(L)) dL`` is the sum over the
    1 km aerosol bands of each band's constant extinction times the path length
    the ray spends in it, ``L(z_top) - L(z_bot)`` (via :func:`lexpr`) clamped to
    the node->top window.

    This is the **curved-atmosphere** slant column. It replaces the former
    plane-parallel ``vertical_OD / cos(theta)``, which (like ``main``) diverges
    as ``cos(theta) -> 0`` at grazing incidence and so over-attenuates -- often
    zeroing -- the most grazing showers. For a vertical shower the walk reduces
    exactly to the vertical column ``aOD55(z_node)``.

    Parameters
    ----------
    L_nodes : ndarray, shape (n_showers, n_nodes)
    L_max : ndarray, shape (n_showers,)
    betaE : ndarray, shape (n_showers,)
    aero_zbnd : ndarray, shape (n_band + 1,)
        Ascending aerosol altitude boundaries (km), ``[0, 1, ..., 30]``.
    aero_ext : ndarray, shape (n_band,)
        Constant aerosol extinction (optical depth per km) within each band.
    R : float
        Earth radius (km).

    Returns
    -------
    AODepth : ndarray, shape (n_showers, n_nodes)
        Slant aerosol optical depth (non-negative).
    """
    Lb = lexpr(aero_zbnd[:, None], betaE[None, :], R=R)  # (n_band+1, n_showers)
    Lc = np.clip(Lb[:, :, None], L_nodes[None], L_max[None, :, None])
    AODepth = np.einsum("s,snk->nk", aero_ext, Lc[1:] - Lc[:-1])
    return np.maximum(AODepth, 0.0)


def refractive_index(z):
    """Refractive index of air from altitude (z) by vertical column_depth (g/cm²).
    numerical parameterization for 1976 STD atmosphere.
    """

    # formerly grammage
    Xv = np.empty_like(z)
    mask1 = z < 11
    mask2 = (z >= 11) & (z < 25)
    mask3 = z >= 25
    Xv[mask1] = ((z[mask1] - 44.34) / -11.861) ** (1 / 0.19)
    Xv[mask2] = np.exp((z[mask2] - 45.5) / -6.34)
    Xv[mask3] = np.exp(13.841 - np.sqrt(28.920 + 3.344 * z[mask3]))

    return 1.0 + 0.000296 * (Xv / 1032.9414) * (273.2 / (204.0 + 0.091 * Xv))
