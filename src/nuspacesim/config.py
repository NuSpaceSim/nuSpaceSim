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
from typing import Any, Literal, Optional, Union

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
        # model_config = ConfigDict(arbitrary_types_allowed=True)
        altitude: float = Quantity(525.0, u.km).value
        """ Altitude from sea-level (KM). """
        latitude: float = Quantity(0.0, u.rad).value
        """ Earth Latitude (Radians). """
        longitude: float = Quantity(0.0, u.rad).value
        """ Earth Longitude (Radians). """

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

    ################ Detector Flight classes ################

    class Stationary(BaseModel):
        id: Literal["stationary"] = "stationary"
        duration: float = Quantity(86400, u.second).value
        """ Flight duration (seconds). """
        start_date: str = "2022-06-02T01:00:00"
        # """Date of observation"""
        start_date_format: str = "isot"
        # """Observation date and time format"""

        @field_validator("duration", mode="before")
        @classmethod
        def valid_timeday(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.second)

        @field_serializer("duration")
        def serialize_day(self, duration: float) -> str:
            return str(Quantity(duration, u.second).to(u.d))

    ################ Detector Field of View Classes ################

    class FieldOfView(BaseModel):
        nadir_span: float = Quantity(np.radians(7.0), u.rad).value
        """ Span of the detector in nadir/zenith (radians) """
        azimuth_span: float = Quantity(np.radians(360.0), u.rad).value
        """ Span of the detector in azimuth (radians) """

        @field_validator("nadir_span", "azimuth_span", mode="before")
        @classmethod
        def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.rad)

        @field_serializer("nadir_span", "azimuth_span")
        def serialize_rad(self, x: float) -> str:
            return str(Quantity(x, u.rad).to(u.deg))

    ################ Detector Pointing classes ################

    class LimbPoint(BaseModel):
        id: Literal["from_limb"] = "from_limb"
        nadir_center_wrt_limb: float = Quantity(np.radians(-3.5), u.rad).value
        """ Nadir angle of detector center w.r.t. to the Earth's limb. Default (Radians). """
        azimuth_center: float = Quantity(np.radians(0.0), u.rad).value
        """ Azimuthal angle of the center of the detector (E is 0 deg, N is 90 deg): Default = 0 """

        @field_validator("nadir_center_wrt_limb", "azimuth_center", mode="before")
        @classmethod
        def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.rad)

        @field_serializer("nadir_center_wrt_limb", "azimuth_center")
        def serialize_rad(self, x: float) -> str:
            return str(Quantity(x, u.rad).to(u.deg))

    class DetRefPoint(BaseModel):
        id: Literal["detector_reference"] = "detector_reference"
        tilt_angle_center_wrt_horiz: float = Quantity(np.radians(0.0), u.rad).value
        """ Tilt angle of the center of the detector w.r.t. to the detector horizontal: Default = 0 """
        azimuth_center_det_ref: float = Quantity(np.radians(0.0), u.rad).value
        """ Azimuthal angle of the center of the detector w.r.t. to the detector horizontal (E is 0 deg, N is 90 deg): Default = 0 """

        @field_validator(
            "tilt_angle_center_wrt_horiz", "azimuth_center_det_ref", mode="before"
        )
        @classmethod
        def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.rad)

        @field_serializer("tilt_angle_center_wrt_horiz", "azimuth_center_det_ref")
        def serialize_rad(self, x: float) -> str:
            return str(Quantity(x, u.rad).to(u.deg))

    class SourceTracking(BaseModel):
        id: Literal["source_tracking"] = "source_tracking"
        tracked_source_RA: float = 0.0
        """Right Ascension of the tracked source"""
        tracked_source_DEC: float = 0.0
        """Declination of the tracked source"""
        obs_date_and_time: str = "2022-06-02T01:00:00"
        """Date of observation"""
        obs_date_and_time_format: str = "isot"
        """Observation date and time format"""

        @field_validator("tracked_source_RA", "tracked_source_DEC", mode="before")
        @classmethod
        def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.rad)

        @field_serializer("tracked_source_RA", "tracked_source_DEC")
        def serialize_rad(self, x: float) -> str:
            return str(Quantity(x, u.rad).to(u.deg))

    ################ Sun & Moon classes ################

    class NoSunMoonCuts(BaseModel):
        id: Literal["no_sun_moon_cuts"] = "no_sun_moon_cuts"

    class SunMoonCuts(BaseModel):
        id: Literal["apply_sun_moon_cuts"] = "apply_sun_moon_cuts"
        model_config = ConfigDict(arbitrary_types_allowed=True)
        # sun_moon_cuts: bool = True
        """ Apply cut for sun and moon: Default = False (default is baseline Diffuse calculation) """
        sun_alt_cut: float = Quantity(np.radians(-40.5), u.rad).value
        """ Sun altitude beyond which no observations are possible: Default = -40.5 deg (elevation angle of Earth's limb for detector flying at 525 km altitude) """
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

    ################ Optical classes ################

    class Optical(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        enable: bool = True
        telescope_effective_area: float = 2.5  # Quantity(2.5, u.m**2)
        """ Effective area of the detector telescope (sq.meters). """
        quantum_efficiency: float = 0.2
        """ Quantum Efficiency of the detector telescope. """
        photo_electron_threshold: float = 10
        """ Photo Electron Threshold, Number Photo electrons. """
        duty_cycle: float = 0.2

        @field_validator("telescope_effective_area", mode="before")
        @classmethod
        def valid_aream2(cls, x: Union[Quantity, float, str]) -> float:
            return parse_units(x, u.m**2)

        @field_serializer("telescope_effective_area")
        def serialize_aream2(self, x: float) -> str:
            return str(Quantity(x, u.m**2))

    ################ Radio classes ################

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
    """Initial position for detector"""
    flight: Optional[Stationary] = Stationary()
    """Flight information for detector"""
    field_of_view: FieldOfView = FieldOfView()
    """Span of detector in azimuth and nadir/zenith"""
    pointing: Union[LimbPoint, DetRefPoint, SourceTracking] = Field(
        default=LimbPoint(), discriminator="id"
    )
    """Direction in which the center of the detector is pointing (w.r.t. Earth's limb or the telescope's horizontal or centered on a source)"""
    sun_moon: Optional[Union[NoSunMoonCuts, SunMoonCuts]] = Field(
        default=NoSunMoonCuts(), discriminator="id"
    )
    """Account for effects of the Sun and the Moon on detector acceptance"""
    optical: Optional[Optical] = Optical()
    """Characteristics of the optical detector"""
    radio: Optional[Radio] = Radio()
    """Characteristics of the radio detector"""


###################################################################


class Simulation(BaseModel):
    """Model holding Simulation Parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ################ Integration Method classes ################

    class MonteCarlo(BaseModel):
        id: Literal["monte_carlo"] = "monte_carlo"
        num_events_per_time_bin: int = 1000
        # num_time_bins: int = 1
        """ Number of time bins (1 for instantaneous acceptance or time-integrated exposure calculations; actual number requested for time-differential or time-averaged calculations) """

    class TargetApprox(BaseModel):
        id: Literal["target_approx"] = "target_approx"
        # num_time_bins: int = 86400
        """ Number of time bins (1 for acceptance-only or time-integrated (diffuse) sensitivity results; actual number requested for all other types of calculations) """

    class Cubature(BaseModel):
        id: Literal["cubature"] = "cubature"
        deg: int = 7
        """ Degree of the polynomial fit """
        num_events_per_node: int = 1000
        # num_time_bins: int = 86400
        """ Number of time bins (1 for acceptance-only or time-integrated (diffuse) sensitivity results; actual number requested for all other types of calculations) """

        @field_validator("deg", mode="before")
        @classmethod
        def valid_degree(cls, x: int) -> int:
            if isinstance(x, int):
                if x < 0 or x > 50:
                    raise ValueError(f"Provided degree {x} is invalid.")
                return x

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

    ################ Target-of-Opportunity classes ################

    class NoSource(BaseModel):
        id: Literal["no_target"] = "no_target"

    class SinglePointSource(BaseModel):
        id: Literal["single_pt_source"] = "single_pt_source"
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

        # @model_validator(mode="before")
        # @classmethod
        # def valid_target_type(cls, data: Any) -> Any:
        #    if isinstance(data, None):
        #        return ""
        #    return data

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
    # thrown_events: int = 1000
    # """ Number of thrown event trajectories. """
    num_time_bins: int = 1
    """ Number of time bins (1 for instantaneous acceptance or time-integrated exposure calculations; actual number requested for time-differential or time-averaged calculations) """
    max_cherenkov_angle: float = np.radians(3.0)
    """ Maximum Cherenkov Angle (Radians). """
    # max_azimuth_angle: float = np.radians(360)
    # """ Maximum Azimuthal Angle (Radians). """
    # angle_from_limb: float = np.radians(7)
    # """ Angle From Limb. Default (Radians). """
    eas_long_profile: Literal[
        "Greisen",
        "Gaisser-Hillas Parameterized",
        "Gaisser-Hillas Fluctuated",
        "Default",
    ] = "Greisen"
    """EAS Longitudinal Profile model: Default = 'Greisen'"""

    @field_validator("eas_long_profile", mode="before")
    @classmethod
    def validate_eas_long_profile(cls, value: str) -> str:
        if value == "Default":
            return "Greisen"
        return value

    cherenkov_light_engine: Literal["nuspacesim", "Default"] = (
        "nuspacesim"  # "CHASM", "EASCherSim"
    )
    """cherenkov light engine model: Default = 'nuspacesim'"""

    @field_validator("cherenkov_light_engine", mode="before")
    @classmethod
    def validate_cherenkov_light_engine(cls, value: str) -> str:
        if value == "Default":
            return "nuspacesim"
        return value

    integ_method: Union[MonteCarlo, TargetApprox, Cubature] = Field(
        default=MonteCarlo(), discriminator="id"
    )
    """ The Method of Integration """
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
    target: Optional[Union[NoSource, SinglePointSource]] = Field(
        default=NoSource(), discriminator="id"
    )

    # target: Optional[TargetOfOpportunity] = TargetOfOpportunity()
    # target: Optional[Union[TargetOfOpportunity, str]] = "no targets"
    # target: Optional[Union[TargetOfOpportunity, None]] = None

    # @field_validator("target", mode="before")
    # @classmethod
    # def validate_target(cls, x: Any) -> Any:
    #    if x is None:
    #        return ""
    #    return x

    # @field_validator("target", mode="before")
    # @classmethod
    # def validate_target(cls, value: str) -> str:
    #    if value == "Default":
    #        return "no targets"
    #    else:
    #        return TargetOfOpportunity()

    @field_validator("max_cherenkov_angle", mode="before")
    @classmethod
    def valid_anglerad(cls, x: Union[Quantity, float, str]) -> float:
        return parse_units(x, u.rad)

    @field_serializer("max_cherenkov_angle")
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
            raise KeyError(f"Missing required key '{fullkey}' in FITS header.")
        return h[fullkey]

    # header (d)etector config value assocciated with partial key string.
    def d(key: str):
        try:
            return v("detector " + key)
        except KeyError as e:
            raise KeyError(f"Detector configuration key error: {e}")

    # header (s)etector config value assocciated with partial key string.
    def s(key: str):
        try:
            return v("simulation " + key)
        except KeyError as e:
            raise KeyError(f"Simulation configuration key error: {e}")

    c = {
        "detector": {
            "initial_position": {
                "altitude": d("initial_position altitude"),
                "latitude": d("initial_position latitude"),
                "longitude": d("initial_position longitude"),
            },
            "name": d("name"),
            "flight": {
                "id": d("flight id"),
                "duration": d("flight duration"),
                "start_date": d("flight start_date"),
                "start_date_format": d("flight start_date_format"),
            },
            "field_of_view": {
                "nadir_span": d("field_of_view nadir_span"),
                "azimuth_span": d("field_of_view azimuth_span"),
            },
            "pointing": {
                "id": d("pointing id"),
                "nadir_center": d("pointing nadir_center"),
                "azimuth_center": d("pointing azimuth_center"),
            },
            "optical": {
                "photo_electron_threshold": d("optical photo_electron_threshold"),
                "quantum_efficiency": d("optical quantum_efficiency"),
                "telescope_effective_area": d("optical telescope_effective_area"),
                "duty_cycle": d("optical duty_cycle"),
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
            # "angle_from_limb": s("angle_from_limb"),
            "cherenkov_light_engine": s("cherenkov_light_engine"),
            "cloud_model": {"id": s("cloud_model id")},
            "ionosphere": {
                "total_electron_content": s("ionosphere total_electron_content"),
                "total_electron_error": s("ionosphere total_electron_error"),
            },
            # "azimuth_span": s("azimuth_span"),
            # "azimuth_center": s("azimuth_center"),
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
            # "thrown_events": s("thrown_events"),
            "num_time_bins": s("num_time_bins"),
            "integ_method": {"id": s("integ_method id")},
        },
        "title": h["Config title"],
    }

    return NssConfig(**c)
