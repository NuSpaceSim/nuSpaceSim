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
from typing import Annotated, Literal, Optional, Union

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.units import Quantity
from pydantic import (  # ValidationError,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from .utils.misc import unflatten_dict

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


def _unit_float(unit: u.Unit, display: Optional[u.Unit] = None):
    """Build an ``Annotated[float, ...]`` type carrying its physical unit once.

    The returned type bundles a ``BeforeValidator`` that accepts a float, a
    units-bearing string (``"525.0 km"``), or an astropy :class:`Quantity` and
    coerces it to a bare float in ``unit``, plus a ``PlainSerializer`` that
    re-attaches the unit (rendered in ``display``) on ``model_dump``. This is
    the single source of truth for a quantity field -- declaring
    ``altitude: Kilometers = 525.0`` replaces the prior trio of an explicit
    default, a per-field ``field_validator``, and a per-field ``field_serializer``
    that each repeated the unit.
    """
    disp = unit if display is None else display
    return Annotated[
        float,
        BeforeValidator(lambda x: parse_units(x, unit)),
        PlainSerializer(lambda v: str(Quantity(v, unit).to(disp)), return_type=str),
    ]


# Reusable quantity field types. The unit (and its display form) is named once
# here; every field below references one of these instead of restating units in
# a default + validator + serializer.
Kilometers = _unit_float(u.km)
Radians = _unit_float(u.rad, u.deg)  # stored as radians, displayed in degrees
MegaHertz = _unit_float(u.MHz)
Decibels = _unit_float(u.dB)
SquareMeters = _unit_float(u.m**2)


class Detector(BaseModel):
    r"""Dataclass holding Detector Characteristics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    class InitialPos(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        altitude: Kilometers = 525.0
        """ Altitude from sea-level (KM). """
        latitude: Radians = 0.0
        """ Right Ascencion (Radians). """
        longitude: Radians = 0.0
        """ Declination (Radians). """

    class SunMoon(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        sun_moon_cuts: bool = True
        """ Apply cut for sun and moon: Default = True """
        sun_alt_cut: Radians = np.radians(-18.0)
        """ Sun altitude beyond which no observations are possible: Default = -18 deg """
        moon_alt_cut: Radians = np.radians(0.0)
        """ Moon altitude beyond which no observations are possible: Default = 0 """
        moon_min_phase_angle_cut: Radians = np.radians(150.0)
        """ Moon phase angle below which, when moon is above moon_alt_cut no observations are possible: Default = 150 deg"""

    class Optical(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        enable: bool = True
        telescope_effective_area: SquareMeters = 2.5
        """ Effective area of the detector telescope (sq.meters). """
        quantum_efficiency: float = 0.2
        """ Quantum Efficiency of the detector telescope. """
        photo_electron_threshold: float = 10
        """ Photo Electron Threshold, Number Photo electrons. """

    class Radio(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        enable: bool = True
        low_frequency: MegaHertz = 30.0
        """ Low end for radio band in MHz: Default = 30 """
        high_frequency: MegaHertz = 300.0
        """ High end of radio band in MHz: Default = 300 """
        snr_threshold: float = 5.0
        """ SNR threshold for radio triggering: Default = 5 """
        nantennas: int = 10
        """ Number of radio antennas: Default = 10 """
        gain: Decibels = 1.8
        """ Antenna gain in dB: Default = 1.8 """

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

    ################ Optical Cherenkov quadrature classes ################

    class CherenkovQuadrature(BaseModel):
        """Gauss-Legendre node-count knobs for the vectorized CphotAng kernel.

        These tune the accuracy/cost of the optical Cherenkov photon-density
        quadrature in :meth:`CphotAng.run`. Every field is optional with a
        default equal to ``run()``'s own default, so omitting the
        ``[simulation.cherenkov_quadrature]`` table (as in any pre-existing
        config file) reproduces the current behavior exactly.
        """

        n_nodes: int = 12
        """GL nodes along the longitudinal (slant-depth) shower grid. Default 12."""
        n_slant_sub: int = 8
        """GL sub-quadrature nodes for the slant-depth & ozone column integrals. Default 8."""
        n_energy_low: int = 3
        """GL nodes on the low-energy panel [eCthres, 1 GeV]. Default 3."""
        n_energy_high: int = 8
        """GL nodes on the high-energy panel [1 GeV, Eshow]. Default 8."""

    ################ Streaming Monte-Carlo classes ################

    class Streaming(BaseModel):
        """Knobs for the streaming, distributed optical Monte-Carlo path.

        Active only under ``nuspacesim run --stream`` (the default one-shot path
        ignores these). Events are thrown in batches across the dask cluster;
        the integral accumulates as an exact additive sketch and the run stops
        when its relative uncertainty reaches ``rel_unc_target`` (or the thrown
        count hits the ``count`` backstop, or ``max_walltime_s`` elapses). Only
        ``reservoir_size`` representative events are written to disk. Every field
        is optional with a default, so the table may be omitted entirely.
        """

        batch_size: int = 5000
        """Trajectories thrown per streamed batch (the unit of parallel work)."""
        reservoir_size: int = 10000
        """k: representative events kept (weighted) and materialized to disk."""
        rel_unc_target: float = 0.05
        """Stop once the optical MC integral's relative uncertainty reaches this."""
        min_thrown: int = 50000
        """Warmup: do not honor ``rel_unc_target`` before this many thrown events."""
        max_walltime_s: float = 0.0
        """Wall-clock backstop in seconds; ``0`` (or negative) disables it."""
        reservoir_weighting: Literal["contribution", "uniform"] = "contribution"
        """Reservoir weight: ``|contribution|`` (importance) or ``uniform``."""

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
        model_config = ConfigDict(arbitrary_types_allowed=True)
        source_RA: Radians = 0.0
        """Right Ascension of the source"""
        source_DEC: Radians = 0.0
        """Declination of the source"""
        source_date: str = "2022-06-02T01:00:00"
        """Date of source observation"""
        source_date_format: str = "isot"
        """Date of the event and format"""
        source_obst: float = 86400  # 24.0 * 60.0 * 60.0
        """Observation time (s). Default = 1 day"""

    ################################################################################

    mode: Literal["Diffuse", "Target"] = "Diffuse"
    """ The Simulation Mode """
    thrown_events: int = 1000
    """ Number of thrown event trajectories. """
    max_cherenkov_angle: Radians = np.radians(3)
    """ Maximum Cherenkov Angle (Radians). """
    max_azimuth_angle: Radians = np.radians(360)
    """ Maximum Azimuthal Angle (Radians). """
    angle_from_limb: Radians = np.radians(7)
    """ Angle From Limb. Default (Radians). """
    eas_long_profile: Literal[
        "Greisen",
        "Gaisser-Hillas Parameterized",
        "Gaisser-Hillas Fluctuated",
        "Default",
    ] = "Greisen"
    """EAS Longitudinal Profile model: Default = 'Greisen'"""

    use_refactored_photon_sum: bool = False
    """Use Phase-B refactored photon-sum kernel in EAS optical simulation."""
    refactored_photon_sum_variant: Literal["v1", "v2", "v3", "v4", "v6"] = "v2"
    """Refactored photon-sum variant key (used when use_refactored_photon_sum=True)."""

    @field_validator("eas_long_profile", mode="before")
    @classmethod
    def validate_eas_long_profile(cls, value: str) -> str:
        if value == "Default":
            return "Greisen"
        return value

    @field_validator("use_refactored_photon_sum", mode="before")
    @classmethod
    def validate_use_refactored_photon_sum(cls, value: bool) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        raise ValueError(
            "use_refactored_photon_sum must be a boolean; "
            f"got {type(value).__name__}"
        )

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

    ionosphere: Optional[Ionosphere] = Ionosphere()
    cherenkov_quadrature: CherenkovQuadrature = CherenkovQuadrature()
    """Optical Cherenkov quadrature node-count knobs (optional; defaults match CphotAng)."""
    streaming: Streaming = Streaming()
    """Streaming distributed optical MC knobs (optional; used by ``run --stream``)."""
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
    """Reconstruct an :class:`NssConfig` from a results FITS header.

    The generic inverse of the writer in ``results_table.init``, which dumps
    ``config.model_dump()`` flattened under space-separated ``Config <path>``
    HIERARCH keys. Here we collect every such key, strip the prefix, and
    :func:`unflatten_dict` rebuilds the nested mapping that ``NssConfig``
    validates. Because both directions derive purely from the model's own
    structure, adding or renaming a field needs no change here -- unlike the
    former hand-transcribed key list, which had to mirror the schema by hand
    (and silently drifted, e.g. loading ``latitude`` into ``longitude``).
    """
    hdul = fits.open(filename, mode="readonly")
    h = hdul[1].header

    prefix = "Config "
    flat = {key[len(prefix) :]: h[key] for key in h.keys() if key.startswith(prefix)}
    if not flat:
        raise KeyError(f"No '{prefix}...' configuration keys found in FITS header.")

    return NssConfig(**unflatten_dict(flat, sep=" "))
