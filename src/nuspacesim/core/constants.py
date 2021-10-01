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
    Top level fundimental constants and the Fund_Constants class.
"""

from __future__ import annotations

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from dataclasses import dataclass
import numpy as np

earth_radius = 6371.0  # in KM
low_earth_orbit = 525.0  # in KM
atmosphere_end = 65.0  # in KM
c = 2.9979246e5
massTau = 1.77686
mean_Tau_life = 2.903e-13


@dataclass
class Fund_Constants:
    r"""Fundimental constants used in nuspacesim simulations.

    Attributes
    ----------
    earth_radius: float
        Spherical Radius of the earth (KM). Default = 6371 KM.
    c: float
        Speed of light in vacuum (m/s).
    massTau: float
        Tau Lepton mass (eV).
    mean_Tau_life: float
        Mean Tau Lepton Lifetime (seconds).
    pi: float
        Pi. Default = np.pi
    """

    earth_radius: float = earth_radius
    c: float = c
    massTau: float = massTau
    mean_Tau_life: float = mean_Tau_life
    pi: float = np.pi

    @cached_property
    def inv_mean_Tau_life(self) -> float:
        """1 / mean tau lifetime."""
        return 1.0 / self.mean_Tau_life  # [s^-1]

    def __call__(self) -> dict[str, tuple[float, str]]:
        r"""Dictionary representation of Fundimental Constants instance.

        Groups the data member values with descriptive comments in a tuple. Adds
        keyword names no longer than eight characters. This is useful for setting the
        FITS Header Keywords in the simulation ouput file. Descriptive comments are
        shorter than 80 characters, so as to conform to FITS standards.

        Returns
        -------
        dict
            Representation of the data members with comments.
        """

        return {
            "R_Earth": (self.earth_radius, "FundimentalConstants: Earth Radius"),
            "c": (self.c, "FundimentalConstants: speed of light"),
            "massTau": (self.massTau, "FundimentalConstants: massTau"),
            "uTauLife": (self.mean_Tau_life, "FundimentalConstants: mean Tau Lifetime"),
            "pi": (self.pi, "FundimentalConstants: pi"),
        }
