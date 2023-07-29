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

from __future__ import annotations

import numpy as np

from ... import constants as const

H_b = const.std_atm_geopotential_height
Lm_b = const.std_atm_lack_rate
T_b = const.std_atm_temperature
P_b = const.std_atm_pressure
gmr = const.std_atm_gmr


def us_std_atm_altitude_from_pressure(P):
    P = np.asarray(P)

    i = np.zeros_like(P, dtype=int)
    for j in range(1, len(P_b)):
        i[P_b[j] >= P] = j

    H = np.full(P.shape, H_b[i])
    m = Lm_b[i] == 0
    x = P > 0
    H[m & x] += T_b[i][m & x] * (1.0 / gmr) * (np.log(P_b[i][m & x] / P[m & x]))
    H[~m & x] += (T_b[i][~m & x] / Lm_b[i][~m & x]) * (
        (P_b[i][~m & x] / P[~m & x]) ** ((1.0 / gmr) * Lm_b[i][~m & x]) - 1
    )

    z = np.empty_like(H)
    z[x] = const.earth_radius * H[x] / (const.earth_radius - H[x])
    z[~x] = np.inf
    return z


def us_std_atm_pressure_from_altitude(z):
    z = np.asarray(z)
    x = z < np.inf
    h = np.empty_like(z)
    h[x] = z[x] * const.earth_radius / (z[x] + const.earth_radius)
    h[~x] = np.inf

    i = np.zeros_like(h, dtype=int)
    for j in range(1, len(H_b)):
        i[H_b[j] <= h] = j

    P = np.full(h.shape, P_b[i])
    m = Lm_b[i] == 0
    P[m & x] *= np.exp((-gmr / T_b[i][m & x]) * (h[m & x] - H_b[i][m & x]))
    P[~m & x] *= (
        T_b[i][~m & x]
        / (T_b[i][~m & x] + Lm_b[i][~m & x] * (h[~m & x] - H_b[i][~m & x]))
    ) ** (gmr / Lm_b[i][~m & x])

    return P
