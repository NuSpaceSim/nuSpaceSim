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

import numpy as np


def viewing_angle(beta_tr, Zdet, Re):
    return np.arcsin((Re / (Re + Zdet)) * np.cos(beta_tr))


def propagation_angle(beta_tr, z, Re):
    return np.arccos((Re / (Re + z)) * np.cos(beta_tr))


def propagation_theta(beta_tr, z, Re):
    return propagation_angle(beta_tr, z, Re)


def length_along_prop_axis(z_start, z_stop, beta_tr, Re):
    L1 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_stop + z_stop ** 2
    L2 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_start + z_start ** 2
    L = np.sqrt(L1) - np.sqrt(L2)
    return L


def deriv_length_along_prop_axis(z_stop, beta_tr, Re):
    L1 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_stop + z_stop ** 2
    L = (Re + z_stop) / np.sqrt(L1)
    return L


def altitude_along_prop_axis(L, z_start, beta_tr, Re):
    r1 = Re ** 2
    r2 = 2 * Re * z_start
    r3 = z_start ** 2
    return -Re + np.sqrt(
        L ** 2 + 2 * L * np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3) + r1 + r2 + r3
    )


def deriv_altitude_along_prop_axis(L, z_start, beta_tr, Re):
    r1 = Re ** 2
    r2 = 2 * Re * z_start
    r3 = z_start ** 2
    r4 = np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3)
    denom = np.sqrt(L ** 2 + 2 * L * r4 + r1 + r2 + r3)
    numer = (Re + z_start) * ((L) / r4 + 1)
    return numer / denom


def gain_in_altitude_along_prop_axis(L, z_start, beta_tr, Re):
    return altitude_along_prop_axis(L, z_start, beta_tr, Re) - z_start


def distance_to_detector(beta_tr, z, z_det, earth_radius):
    theta_view = viewing_angle(beta_tr, z_det, earth_radius)
    theta_prop = propagation_angle(beta_tr, z, earth_radius)
    ang_e = np.pi / 2 - theta_view - theta_prop
    return np.sin(ang_e) / np.sin(theta_view) * (z + earth_radius)
