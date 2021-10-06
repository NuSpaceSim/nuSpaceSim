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

"""Module holding conifiguration class definitions."""

####

from __future__ import annotations

from typing import Union
from dataclasses import dataclass

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from numpy import radians, log10, sin

from . import constants as const

__all__ = [
    "DetectorCharacteristics",
    "SimulationParameters",
    "NssConfig",
]


@dataclass
class DetectorCharacteristics:
    r"""Dataclass holding Detector Characteristics.

    """

    altitude: float = 525.0
    """ Altitude from sea-level. Default = 525 KM """
    ra_start: float = 0.0
    """ Right Ascencion. Default = 0.0 """
    dec_start: float = 0.0
    """ Declination. Default = 0.0 """
    telescope_effective_area: float = 2.5
    '"" Effective area of the detector telescope. Default = 2.5 sq.meters ""'
    quantum_efficiency: float = 0.2
    """ Quantum Efficiency of the detector telescope. Default = 0.2 """
    photo_electron_threshold: float = 10.0
    """Photo Electron Threshold, Number Photo electrons = 10"""

    def __call__(self) -> dict[str, tuple[float, str]]:
        r"""Dictionary representation of DetectorCharacteristics instance.

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
            "detAlt": (self.altitude, "Detector: Altitude"),
            "raStart": (self.ra_start, "Detector: Initial Right Ascencion"),
            "decStart": (self.dec_start, "Detector: Initial Declination"),
            "telEffAr": (
                self.telescope_effective_area,
                "Detector: Telescope Effective Area",
            ),
            "quantEff": (self.quantum_efficiency, "Detector: Quantum Efficiency"),
            "phEthres": (
                self.photo_electron_threshold,
                "Detector: Photo-Electron Threshold",
            ),
        }


@dataclass
class SimulationParameters:
    r"""Dataclass holding Simulation Parameters.
    """

    N: int = 1000
    """Number of thrown trajectories. Default = 1000"""
    theta_ch_max: float = radians(3.0)
    """Maximum Cherenkov Angle in radians. Default = π/60 radians (3 degrees)."""
    nu_tau_energy: float = 1e8
    """Energy of the tau neutrinos in GeV. Default = 1e8 GeV."""
    e_shower_frac: float = 0.5
    """Fraction of ETau in Shower. Default = 0.5."""
    ang_from_limb: float = radians(7.0)
    """Angle From Limb. Default = π/25.714 radians (7 degrees)."""
    max_azimuth_angle: float = radians(360.0)
    """Maximum Azimuthal Angle. Default = 2π radians (360 degrees)."""

    @cached_property
    def log_nu_tau_energy(self) -> float:
        """log base 10 of nu_tau_energy."""
        return log10(self.nu_tau_energy)

    @cached_property
    def sin_theta_ch_max(self) -> float:
        """sin of theta_ch_max."""
        return sin(self.theta_ch_max)

    def __call__(self) -> dict[str, tuple[Union[int, float], str]]:
        r"""Dictionary representation of SimulationParameters instance.

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
            "N": (self.N, "Simulation: thrown neutrinos"),
            "thChMax": (self.theta_ch_max, "Simulation: Maximum Cherenkov Angle"),
            "nuTauEn": (self.nu_tau_energy, "Simulation: nutau energy (GeV)"),
            "eShwFrac": (self.e_shower_frac, "Simulation: Fraction of Etau in Shower"),
            "angLimb": (self.ang_from_limb, "Simulation: Angle From Limb"),
            "maxAzAng": (self.max_azimuth_angle, "Simulation: Maximum Azimuthal Angle"),
        }


@dataclass
class NssConfig:
    r"""Necessary Configuration Data for NuSpaceSim.

    An :class:`NssConfig` is a contianer object holding all of the other nuSpaceSim
    configuration objects for a simplified access API. Instances of :class:`NssConfig`
    objects can be serialized to XML.

    .. :ref:`XML<xml_config.create_xml>`.

    """

    detector: DetectorCharacteristics = DetectorCharacteristics()
    """The Detector Characteristics."""
    simulation: SimulationParameters = SimulationParameters()
    """The Simulation Parameters."""
    constants: const.Fund_Constants = const.Fund_Constants()
    """The Fudimental physical constants."""
