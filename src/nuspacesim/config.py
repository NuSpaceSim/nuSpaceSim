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

from dataclasses import dataclass
from typing import Any, Union

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

from numpy import nan, radians, sin

from . import constants as const

__all__ = [
    "DetectorCharacteristics",
    "SimulationParameters",
    "NssConfig",
    "FileSpectrum",
    "MonoSpectrum",
    "PowerSpectrum",
]


@dataclass
class DetectorCharacteristics:
    r"""Dataclass holding Detector Characteristics."""

    method: str = "Optical"
    """ Type of detector: Default = Optical """
    altitude: float = 525.0
    """ Altitude from sea-level. Default = 525 KM """
    ra_start: float = 0.0
    """ Right Ascencion (Degrees). Default = 0.0 """
    dec_start: float = 0.0
    """ Declination (Degrees). Default = 0.0 """
    telescope_effective_area: float = 2.5
    '"" Effective area of the detector telescope. Default = 2.5 sq.meters ""'
    quantum_efficiency: float = 0.2
    """ Quantum Efficiency of the detector telescope. Default = 0.2 """
    photo_electron_threshold: float = 10.0
    """Photo Electron Threshold, Number Photo electrons = 10"""
    low_freq: float = 30.0
    """ Low end for radio band in MHz: Default = 30 """
    high_freq: float = 300.0
    """ High end of radio band in MHz: Default = 300 """
    det_SNR_thres: float = 5.0
    """ SNR threshold for radio triggering: Default = 5 """
    det_Nant: int = 10
    """ Number of radio antennas: Default = 10 """
    det_gain: float = 1.8
    """ Antenna gain in dB: Default = 1.8 """

    def __call__(self) -> dict[str, tuple[Any, str]]:
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
            "method": (self.method, "Detector: Method (Optical, radio or both"),
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
            "lowFreq": (self.low_freq, "Detector: Antenna Low Frequency"),
            "highFreq": (self.high_freq, "Detector: Antenna High Frequency"),
            "SNRthres": (self.det_SNR_thres, "Detector: Radio SNR threshold"),
            "detNant": (self.det_Nant, "Detector: Number of Antennas"),
            "detGain": (self.det_gain, "Detector: Antenna Gain"),
        }


@dataclass
class MonoSpectrum:
    log_nu_tau_energy: float = 8.0
    """Log Energy of the tau neutrinos in GeV."""

    def __call__(self) -> dict:
        return {
            "specType": ("Mono", "Simulation: nutau energy spectrum"),
            "specPara": (self.log_nu_tau_energy, "Simulation: Log Energy (GeV)"),
        }


@dataclass
class PowerSpectrum:
    index: float = 2.0
    """Power Law Log Energy of the tau neutrinos in GeV."""

    lower_bound: float = 6.0
    """Lower Bound Log nu_tau Energy GeV."""

    upper_bound: float = 12.0
    """Upper Bound Log nu_tau Energy GeV."""

    def __call__(self) -> dict:
        return {
            "specType": ("Power", "Simulation: nutau Power Law energy spectrum"),
            "specPara": (self.index, "Simulation: Power Law Index"),
            "specLow": (self.lower_bound, "Simulation: Energy Lower Bound"),
            "specHigh": (self.upper_bound, "Simulation: Energy Upper Bound"),
        }


@dataclass
class FileSpectrum:
    path: str = ""
    """File path to user defined spectrum file"""

    def __call__(self) -> dict:
        return {
            "specType": ("File", "Simulation: Nutau User Defined energy spectrum"),
            "specFile": (self.path, "Simulation: FilePath"),
        }


@dataclass
class SimulationParameters:
    """Dataclass holding Simulation Parameters."""

    N: int = 10000
    """Number of thrown trajectories. Default = 1000"""
    theta_ch_max: float = radians(3.0)
    """Maximum Cherenkov Angle in radians. Default = π/60 radians (3 degrees)."""
    spectrum: Union[MonoSpectrum, PowerSpectrum, FileSpectrum] = MonoSpectrum()
    """Distribution from which to draw nu_tau energies."""
    e_shower_frac: float = 0.5
    """Fraction of ETau in Shower. Default = 0.5."""
    ang_from_limb: float = radians(7.0)
    """Angle From Limb. Default = π/25.714 radians (7 degrees)."""
    max_azimuth_angle: float = radians(360.0)
    """Maximum Azimuthal Angle. Default = 2π radians (360 degrees)."""
    model_ionosphere: int = 0
    """Model ionosphere for radio propagation?. Default = 0 (false)."""
    TEC: float = 10.0
    """Total Electron Content for ionospheric propagation. Default = 10."""
    TECerr: float = 0.1
    """Error for TEC reconstruction. Default = 0.1"""

    @cached_property
    def log_nu_tau_energy(self) -> float:
        """log10 of nu_tau_energy."""
        if isinstance(self.spectrum, MonoSpectrum):
            return self.spectrum.log_nu_tau_energy
        else:
            return nan

    @cached_property
    def nu_tau_energy(self) -> float:
        """10 ^ log_nu_tau_energy."""
        if isinstance(self.spectrum, MonoSpectrum):
            return 10 ** self.log_nu_tau_energy
        else:
            return nan

    @cached_property
    def sin_theta_ch_max(self) -> float:
        """sin of theta_ch_max."""
        return sin(self.theta_ch_max)

    def __call__(self) -> dict[str, tuple[Union[int, float, str], str]]:
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
        d = {
            "N": (self.N, "Simulation: thrown neutrinos"),
            "thChMax": (self.theta_ch_max, "Simulation: Maximum Cherenkov Angle"),
            "specUnit": ("log(GeV)", "Simulation: Energy spectrum units"),
            "eShwFrac": (self.e_shower_frac, "Simulation: Fraction of Etau in Shower"),
            "angLimb": (self.ang_from_limb, "Simulation: Angle From Limb"),
            "maxAzAng": (self.max_azimuth_angle, "Simulation: Maximum Azimuthal Angle"),
            "ionosph": (
                self.model_ionosphere,
                "Simulation: Radio ionosphere model flg",
            ),
            "TEC": (self.TEC, "Simulation: Actual slant TEC value"),
            "TECerr": (self.TECerr, "Simulation: Uniform distr. err: TEC est."),
        }

        d.update(self.spectrum())

        return d


@dataclass
class NssConfig:
    r"""Necessary Configuration Data for NuSpaceSim.

    An :class:`NssConfig` is a container object holding all of the other nuSpaceSim
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
