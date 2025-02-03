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
"""Module holding conifiguration class definitions."""

####

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Union

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.units import Quantity
from pydantic import (  # ValidationError,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import tomli_w

__all__ = [
    "NssConfig",
    "Detector",
    "Simulation",
    "config_from_toml",
    "create_toml",
    "config_from_fits",
]


def parse_units(value: Union[Quantity, float, str], unit: u.Unit) -> float:
    if isinstance(value, (Quantity, str)):
        return Quantity(value).to(unit).value
    else:
        return Quantity(value, unit).value


class Detector(BaseModel):
    r"""Dataclass holding Detector Characteristics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    class InitialPos(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        altitude: float = Quantity(525.0, u.km).value
        """ Altitude from sea-level (KM). """
        latitude: float = Quantity(0.0, u.rad).value
        """ Right Ascencion (Radians). """
        longitude: float = Quantity(0.0, u.rad).value
        """ Declination (Radians). """

        @field_validator("altitude", mode="before")
        @classmethod
        def valid_distkm(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.km)

        @field_validator("latitude", "longitude", mode="before")
        @classmethod
        def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.rad)

        @field_serializer("altitude")
        def serialize_km(self, altitude: float) -> str:
            return str(Quantity(altitude, u.km))

        @field_serializer("latitude", "longitude")
        def serialize_rad(self, x: float) -> str:
            return str(Quantity(x, u.rad).to(u.deg))

    class SunMoon(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        sun_moon_cuts: bool = True
        """ Apply cut for sun and moon: Default = True """
        sun_alt_cut: float = Quantity(np.radians(-18.0), u.rad).value
        """ Sun altitude beyond which no observations are possible: Default = -18 deg """
        moon_alt_cut: float = Quantity(np.radians(0.0), u.rad).value
        """ Moon altitude beyond which no observations are possible: Default = 0 """
        moon_min_phase_angle_cut: float = Quantity(np.radians(150.0), u.rad).value
        """ Moon phase angle below which, when moon is above moon_alt_cut no observations are possible: Default = 150 deg"""

        @field_validator(
            "sun_alt_cut", "moon_alt_cut", "moon_min_phase_angle_cut", mode="before"
        )
        @classmethod
        def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.rad)

        @field_serializer("sun_alt_cut", "moon_alt_cut", "moon_min_phase_angle_cut")
        def serialize_rad(self, x: float) -> str:
            return str(Quantity(x, u.rad).to(u.deg))

    class Optical(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        enable: bool = True
        telescope_effective_area: float = 2.5  # Quantity(2.5, u.m**2)
        """ Effective area of the detector telescope (sq.meters). """
        quantum_efficiency: float = 0.2
        """ Quantum Efficiency of the detector telescope. """
        photo_electron_threshold: float = 10
        """ Photo Electron Threshold, Number Photo electrons. """

        @field_validator("telescope_effective_area", mode="before")
        @classmethod
        def valid_aream2(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.m**2)

        @field_serializer("telescope_effective_area")
        def serialize_aream2(self, x: float) -> str:
            return str(Quantity(x, u.m**2))

    class Radio(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        enable: bool = True
        low_frequency: float = Quantity(30.0, u.MHz).value
        """ Low end for radio band in MHz: Default = 30 """
        high_frequency: float = Quantity(300.0, u.MHz).value
        """ High end of radio band in MHz: Default = 300 """
        snr_threshold: float = 5.0
        """ SNR threshold for radio triggering: Default = 5 """
        nantennas: int = 10
        """ Number of radio antennas: Default = 10 """
        gain: float = Quantity(1.8, u.dB).value
        """ Antenna gain in dB: Default = 1.8 """

        @field_validator("low_frequency", "high_frequency", mode="before")
        @classmethod
        def valid_freqMHz(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.MHz)

        @field_validator("gain", mode="before")
        @classmethod
        def valid_powerdB(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.dB)

        @field_serializer("low_frequency", "high_frequency")
        def serialize_freqMHz(self, x: float) -> str:
            return str(Quantity(x, u.MHz))

        @field_serializer("gain")
        def serialize_dB(self, x: float) -> str:
            return str(Quantity(x, u.dB))

        @model_validator(mode="after")
        def validate_high_frequency(self):
            if self.high_frequency <= self.low_frequency:
                raise ValueError("High frequency must be greater than low frequency")
            return self

    name: str = "Default Name"
    initial_position: InitialPos = InitialPos()
    """Initial conditions for detector"""
    sun_moon: Optional[SunMoon] = SunMoon()
    """[Target only] Detector sensitivity to effects of the sun and moon"""
    optical: Optional[Optical] = Optical()
    """Characteristics of the optical detector"""
    radio: Optional[Radio] = Radio()
    """Characteristics of the radio detector"""


###################################################################


class Simulation(BaseModel):
    """Model holding Simulation Parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ################ Radio Ionosphere classes ################

    class Ionosphere(BaseModel):
        enable: bool = True
        total_electron_content: float = 10.0
        """Total Electron Content for ionospheric propagation. """
        total_electron_error: float = 0.1
        """Error for TEC reconstruction. """

    ################ tau_shower classes ################

    class NuPyPropShower(BaseModel):
        id: Literal["nupyprop", "nuleptonsim", "nupyprop_bdhm"] = "nupyprop"
        etau_frac: float = 0.5
        """Fraction of ETau in Shower. Default = 0.5."""
        table_version: str = "3"
        """Version of tau conversion tables."""

    ################ spectrum classes ################

    class MonoSpectrum(BaseModel):
        id: Literal["monospectrum"] = "monospectrum"
        log_nu_energy: float = 8.0
        """Log Energy of the tau neutrinos in GeV."""

    class PowerSpectrum(BaseModel):
        id: Literal["powerspectrum"] = "powerspectrum"
        index: float = 2.0
        """Power Law Log Energy of the tau neutrinos in GeV."""
        lower_bound: float = 6.0
        """Lower Bound Log nu_tau Energy GeV."""
        upper_bound: float = 12.0
        """Upper Bound Log nu_tau Energy GeV."""

    ################ Cloud Model classes ################

    class NoCloud(BaseModel):
        id: Literal["no_cloud"] = "no_cloud"

    class MonoCloud(BaseModel):
        id: Literal["monocloud"] = "monocloud"
        altitude: float = float("-inf")
        """Altitude of monoheight cloud."""

    class PressureMapCloud(BaseModel):
        id: Literal["pressure_map"] = "pressure_map"
        month: int = 1
        """Cloud Map Month integer 1-12 inclusive."""
        version: Union[str, int] = 0
        """Cloud Map File Version."""

        @field_validator("month", mode="before")
        @classmethod
        def valid_month(cls, date: str | int | datetime) -> int:
            if isinstance(date, datetime):
                return date.month
            if isinstance(date, int):
                if date < 1 or date > 12:
                    raise ValueError(f"Provided month {date} is invalid")
                return date
            if isinstance(date, str):
                try:
                    return (datetime.strptime(date, "%m")).month
                except ValueError:
                    pass
                try:
                    return (datetime.strptime(date, "%B")).month
                except ValueError:
                    pass
                try:
                    return (datetime.strptime(date, "%b")).month
                except ValueError:
                    pass
                raise ValueError(
                    f"date {date} does not match valid month patterns (%m, %B, %b)"
                )

    class TargetOfOpportunity(BaseModel):
        source_RA: float = 0.0
        """Right Ascension of the source"""
        source_DEC: float = 0.0
        """Declination of the source"""
        source_date: str = "2022-06-02T01:00:00"
        """Date of source observation"""
        source_date_format: str = "isot"
        """Date of the event and format"""
        source_obst: float = 86400  # 24.0 * 60.0 * 60.0
        """Observation time (s). Default = 1 day"""

        @field_validator("source_RA", "source_DEC", mode="before")
        @classmethod
        def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.rad)

        @field_serializer("source_RA", "source_DEC")
        def serialize_rad(self, x: float) -> str:
            return str(Quantity(x, u.rad).to(u.deg))

    ################################################################################

    mode: Literal["Diffuse", "Target"] = "Diffuse"
    """ The Simulation Mode """
    thrown_events: int = 1000
    """ Number of thrown event trajectories. """
    max_cherenkov_angle: float = np.radians(3)
    """ Maximum Cherenkov Angle (Radians). """
    max_azimuth_angle: float = np.radians(360)
    """ Maximum Azimuthal Angle (Radians). """
    angle_from_limb: float = np.radians(7)
    """ Angle From Limb. Default (Radians). """
    cherenkov_light_engine: Literal["Default"] = "Default"  # , "CHASM" , "EASCherSim"
    ionosphere: Optional[Ionosphere] = Ionosphere()
    tau_shower: NuPyPropShower = NuPyPropShower()
    """ Tau Shower Generator. """
    spectrum: Union[MonoSpectrum, PowerSpectrum] = Field(
        default=MonoSpectrum(), discriminator="id"
    )
    """ Distribution from which to draw nu_tau energies. """
    cloud_model: Union[NoCloud, MonoCloud, PressureMapCloud] = Field(
        default=NoCloud(), discriminator="id"
    )
    target: Optional[TargetOfOpportunity] = TargetOfOpportunity()

    @field_validator(
        "max_cherenkov_angle", "max_azimuth_angle", "angle_from_limb", mode="before"
    )
    @classmethod
    def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
        return parse_units(x, u.rad)

    @field_serializer("max_cherenkov_angle", "max_azimuth_angle", "angle_from_limb")
    def serialize_rad(self, x: float) -> str:
        return str(Quantity(x, u.rad).to(u.deg))


class NssConfig(BaseModel):
    r"""Necessary Configuration Data for NuSpaceSim.

    An :class:`NssConfig` is a container object holding all of the other nuSpaceSim
    configuration objects for a simplified access API. Instances of :class:`NssConfig`
    objects can be serialized to TOML.
    """

    title: str = "NuSpaceSim"
    detector: Detector = Detector()
    """The Detector Characteristics."""
    simulation: Simulation = Simulation()
    """The Simulation Parameters."""


def config_from_toml(filename: str) -> NssConfig:
    with open(filename, "rb") as f:
        c = tomllib.load(f)
        return NssConfig(**c)


def create_toml(filename: str, c: NssConfig):
    with open(filename, "wb") as f:
        tomli_w.dump(c.model_dump(), f)


def config_from_fits(filename: str) -> NssConfig:
    hdul = fits.open(filename, mode="readonly")
    h = hdul[1].header

    # header config (v)alue assocciated with partial key string.
    def v(key: str):
        fullkey = "Config " + key
        if fullkey not in h:
            raise KeyError(fullkey)
        return h[fullkey]

    # header (d)etector config value assocciated with partial key string.
    def d(key: str):
        return v("detector " + key)

    # header (s)etector config value assocciated with partial key string.
    def s(key: str):
        return v("simulation " + key)

    c = {
        "detector": {
            "initial_position": {
                "altitude": d("initial_position altitude"),
                "latitude": d("initial_position latitude"),
                "longitude": d("initial_position latitude"),
            },
            "name": d("name"),
            "optical": {
                "photo_electron_threshold": d("optical photo_electron_threshold"),
                "quantum_efficiency": d("optical quantum_efficiency"),
                "telescope_effective_area": d("optical telescope_effective_area"),
            },
            "radio": {
                "gain": d("radio gain"),
                "high_frequency": d("radio high_frequency"),
                "low_frequency": d("radio low_frequency"),
                "nantennas": d("radio nantennas"),
                "snr_threshold": d("radio snr_threshold"),
            },
        },
        "simulation": {
            "angle_from_limb": s("angle_from_limb"),
            "cherenkov_light_engine": s("cherenkov_light_engine"),
            "cloud_model": {"id": s("cloud_model id")},
            "ionosphere": {
                "total_electron_content": s("ionosphere total_electron_content"),
                "total_electron_error": s("ionosphere total_electron_error"),
            },
            "max_azimuth_angle": s("max_azimuth_angle"),
            "max_cherenkov_angle": s("max_cherenkov_angle"),
            "mode": s("mode"),
            "spectrum": {
                "id": s("spectrum id"),
                "log_nu_energy": s("spectrum log_nu_energy"),
            },
            "tau_shower": {
                "etau_frac": s("tau_shower etau_frac"),
                "id": s("tau_shower id"),
                "table_version": s("tau_shower table_version"),
            },
            "thrown_events": s("thrown_events"),
        },
        "title": h["Config title"],
    }

    return NssConfig(**c)
