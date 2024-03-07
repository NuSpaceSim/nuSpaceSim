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

import tempfile
from datetime import datetime

import astropy.units as u
import numpy as np
import pytest
from pydantic import ValidationError

from nuspacesim.config import (
    Detector,
    NssConfig,
    Simulation,
    config_from_toml,
    create_toml,
)


def test_detector_initial_position_default():
    ip = Detector.InitialPos()
    assert ip.altitude == 525.0
    assert ip.latitude == 0
    assert ip.longitude == 0

    assert ip.model_dump() == {
        "altitude": "525.0 km",
        "latitude": "0.0 deg",
        "longitude": "0.0 deg",
    }


def test_detector_initial_position_units():
    ip = Detector.InitialPos(
        altitude=50000 * u.m, latitude=10.0 * u.degree, longitude=np.radians(10.0)
    )
    assert ip.altitude == 50
    assert ip.latitude == np.radians(10)
    assert ip.longitude == np.radians(10)

    assert ip.model_dump() == {
        "altitude": "50.0 km",
        "latitude": "10.0 deg",
        "longitude": "10.0 deg",
    }


def test_detector_initial_position_bad_units():
    with pytest.raises(ValidationError):
        Detector.InitialPos(
            altitude=50000 * u.s, latitude=10.0 * u.s, longitude=10.0 * u.s
        )


def test_detector_optical_default():
    a = Detector.Optical()
    assert a.telescope_effective_area == 2.5
    assert a.quantum_efficiency == 0.2
    assert a.photo_electron_threshold == 10

    assert a.model_dump() == {
        "enable": True,
        "telescope_effective_area": "2.5 m2",
        "quantum_efficiency": 0.2,
        "photo_electron_threshold": 10.0,
    }


def test_detector_optical_units():
    a = Detector.Optical(
        telescope_effective_area=16,
        quantum_efficiency=0.5,
        photo_electron_threshold=100,
    )
    assert a.telescope_effective_area == 16
    assert a.quantum_efficiency == 0.5
    assert a.photo_electron_threshold == 100

    assert a.model_dump() == {
        "enable": True,
        "telescope_effective_area": "16.0 m2",
        "quantum_efficiency": 0.5,
        "photo_electron_threshold": 100.0,
    }

    b = Detector.Optical(telescope_effective_area=2.5e4 * u.cm * u.cm)
    assert b.telescope_effective_area == 2.5
    assert b.quantum_efficiency == 0.2
    assert b.photo_electron_threshold == 10

    assert b.model_dump() == {
        "enable": True,
        "telescope_effective_area": "2.5 m2",
        "quantum_efficiency": 0.2,
        "photo_electron_threshold": 10.0,
    }


def test_detector_optical_bad_units():
    with pytest.raises(ValidationError):
        Detector.Optical(telescope_effective_area=2.5 * u.cm)


def test_detector_radio_default():
    radio = Detector.Radio()
    assert radio.low_frequency == 30.0
    assert radio.high_frequency == 300.0
    assert radio.snr_threshold == 5.0
    assert radio.nantennas == 10
    assert radio.gain == 1.8

    assert radio.model_dump() == {
        "enable": True,
        "low_frequency": "30.0 MHz",
        "high_frequency": "300.0 MHz",
        "snr_threshold": 5.0,
        "nantennas": 10,
        "gain": "1.8 dB",
    }


def test_detector_radio_units():
    radio = Detector.Radio(
        low_frequency=40.0,
        high_frequency=200.0,
        snr_threshold=7.0,
        nantennas=15,
        gain=2.5,
    )
    assert radio.low_frequency == 40.0
    assert radio.high_frequency == 200.0
    assert radio.snr_threshold == 7.0
    assert radio.nantennas == 15
    assert radio.gain == 2.5

    assert radio.model_dump() == {
        "enable": True,
        "low_frequency": "40.0 MHz",
        "high_frequency": "200.0 MHz",
        "snr_threshold": 7.0,
        "nantennas": 15,
        "gain": "2.5 dB",
    }

    radio_with_units = Detector.Radio(
        low_frequency=1 * u.kHz, high_frequency=1 * u.GHz, gain=3.0 * u.dB
    )
    assert radio_with_units.low_frequency == 0.001
    assert radio_with_units.high_frequency == 1000
    assert radio_with_units.gain == 3.0

    assert radio_with_units.model_dump() == {
        "enable": True,
        "low_frequency": "0.001 MHz",
        "high_frequency": "1000.0 MHz",
        "snr_threshold": 5.0,
        "nantennas": 10,
        "gain": "3.0 dB",
    }


def test_detector_radio_bad_units():
    with pytest.raises(ValidationError):
        Detector.Radio(low_frequency=2.5 * u.cm)


def test_detector_radio_invalid():
    with pytest.raises(ValidationError):
        Detector.Radio(low_frequency=2, high_frequency=1)


def test_detector_default():
    detector = Detector()
    assert detector.name == "Default Name"
    assert detector.initial_position.altitude == 525.0
    assert detector.initial_position.latitude == 0.0
    assert detector.initial_position.longitude == 0.0
    assert detector.optical.telescope_effective_area == 2.5
    assert detector.optical.quantum_efficiency == 0.2
    assert detector.optical.photo_electron_threshold == 10
    assert detector.radio.low_frequency == 30.0
    assert detector.radio.high_frequency == 300.0
    assert detector.radio.snr_threshold == 5.0
    assert detector.radio.nantennas == 10
    assert detector.radio.gain == 1.8


def test_detector_custom_values():
    a = Detector(
        name="Custom Detector",
        initial_position=Detector.InitialPos(
            altitude=600.0, latitude=1.0, longitude=2.0
        ),
        optical=Detector.Optical(
            telescope_effective_area=3.0,
            quantum_efficiency=0.5,
            photo_electron_threshold=20,
        ),
        radio=Detector.Radio(
            low_frequency=40.0,
            high_frequency=200.0,
            snr_threshold=7.0,
            nantennas=15,
            gain=2.5,
        ),
    )
    assert a.name == "Custom Detector"
    assert a.initial_position.altitude == 600.0
    assert a.initial_position.latitude == 1.0
    assert a.initial_position.longitude == 2.0
    assert a.optical.telescope_effective_area == 3.0
    assert a.optical.quantum_efficiency == 0.5
    assert a.optical.photo_electron_threshold == 20
    assert a.radio.low_frequency == 40.0
    assert a.radio.high_frequency == 200.0
    assert a.radio.snr_threshold == 7.0
    assert a.radio.nantennas == 15
    assert a.radio.gain == 2.5


def test_detector_serialization():
    assert Detector().model_dump() == {
        "name": "Default Name",
        "initial_position": {
            "altitude": "525.0 km",
            "latitude": "0.0 deg",
            "longitude": "0.0 deg",
        },
        "sun_moon": {
            "moon_alt_cut": "0.0 deg",
            "moon_min_phase_angle_cut": "150.0 deg",
            "sun_alt_cut": "-18.0 deg",
            "sun_moon_cuts": True,
        },
        "optical": {
            "enable": True,
            "telescope_effective_area": "2.5 m2",
            "quantum_efficiency": 0.2,
            "photo_electron_threshold": 10.0,
        },
        "radio": {
            "enable": True,
            "low_frequency": "30.0 MHz",
            "high_frequency": "300.0 MHz",
            "snr_threshold": 5.0,
            "nantennas": 10,
            "gain": "1.8 dB",
        },
    }


################################################################################


def test_default_simulation():
    a = Simulation()
    assert a.mode == "Diffuse"
    assert a.thrown_events == 1000
    assert a.max_cherenkov_angle == np.radians(3.0)
    assert a.max_azimuth_angle == np.radians(360.0)
    assert a.angle_from_limb == np.radians(7.0)
    assert a.cherenkov_light_engine == "Default"
    assert a.ionosphere is not None
    assert a.ionosphere.total_electron_content == 10.0
    assert a.ionosphere.total_electron_error == 0.1
    assert a.tau_shower.id == "nupyprop"
    assert a.tau_shower.etau_frac == 0.5
    assert a.tau_shower.table_version == "3"
    assert isinstance(a.spectrum, Simulation.MonoSpectrum)
    assert isinstance(a.cloud_model, Simulation.NoCloud)

    assert a.model_dump() == {
        "mode": "Diffuse",
        "thrown_events": 1000,
        "max_cherenkov_angle": "3.0000000000000004 deg",
        "max_azimuth_angle": "360.0 deg",
        "angle_from_limb": "7.0 deg",
        "cherenkov_light_engine": "Default",
        "ionosphere": {
            "enable": True,
            "total_electron_content": 10.0,
            "total_electron_error": 0.1,
        },
        "tau_shower": {"id": "nupyprop", "etau_frac": 0.5, "table_version": "3"},
        "spectrum": {"id": "monospectrum", "log_nu_energy": 8.0},
        "cloud_model": {"id": "no_cloud"},
        "target": {
            "source_DEC": "0.0 deg",
            "source_RA": "0.0 deg",
            "source_date": "2022-06-02T01:00:00",
            "source_date_format": "isot",
            "source_obst": 86400,
        },
    }


def test_custom_simulation():
    a = Simulation(
        mode="Target",
        thrown_events=500,
        max_cherenkov_angle=3.5 * u.deg,
        max_azimuth_angle=270.0 * u.deg,
        angle_from_limb=10.0 * u.deg,
        ionosphere=Simulation.Ionosphere(total_electron_content=15.0),
        tau_shower=Simulation.NuPyPropShower(etau_frac=0.6),
        spectrum=Simulation.PowerSpectrum(index=2.5, lower_bound=7.0, upper_bound=11.0),
        cloud_model=Simulation.MonoCloud(altitude=20.0),
    )

    assert a.mode == "Target"
    assert a.thrown_events == 500
    assert a.max_cherenkov_angle == np.radians(3.5)
    assert a.max_azimuth_angle == np.radians(270.0)
    assert a.angle_from_limb == np.radians(10.0)
    assert a.ionosphere is not None
    assert a.ionosphere.total_electron_content == 15.0
    assert a.tau_shower.etau_frac == 0.6
    assert isinstance(a.spectrum, Simulation.PowerSpectrum)
    assert isinstance(a.cloud_model, Simulation.MonoCloud)
    assert a.spectrum.index == 2.5
    assert a.spectrum.lower_bound == 7.0
    assert a.spectrum.upper_bound == 11.0
    assert a.cloud_model.altitude == 20.0


def test_invalid_cherenkov_angle():
    with pytest.raises(TypeError):
        Simulation(max_cherenkov_angle="invalid_angle")


def test_invalid_ionosphere_content():
    with pytest.raises(ValidationError):
        Simulation(
            ionosphere=Simulation.Ionosphere(total_electron_content="invalid_content")
        )


def test_invalid_spectrum_type():
    with pytest.raises(ValidationError):
        Simulation(spectrum="invalid_spectrum_type")


def test_invalid_spectrum_values():
    with pytest.raises(ValidationError):
        Simulation(spectrum=Simulation.MonoSpectrum(log_nu_energy="invalid_value"))


def test_invalid_cloud_model_type():
    with pytest.raises(ValidationError):
        Simulation(cloud_model="invalid_cloud_model_type")


def test_invalid_cloud_model_values():
    with pytest.raises(ValidationError):
        Simulation(cloud_model=Simulation.MonoCloud(altitude="invalid_value"))


def test_pressure_map_cloud_default():
    cloud = Simulation.PressureMapCloud()
    assert cloud.id == "pressure_map"
    assert cloud.month == 1
    assert cloud.version == 0


def test_pressure_map_cloud_custom_values():
    cloud = Simulation.PressureMapCloud(month=6, version="2023")
    assert cloud.id == "pressure_map"
    assert cloud.month == 6
    assert cloud.version == "2023"


def test_pressure_map_cloud_serialization():
    cloud = Simulation.PressureMapCloud(month=8, version=2)
    assert cloud.model_dump() == {"id": "pressure_map", "month": 8, "version": 2}


def test_pressure_map_cloud_valid_month():
    assert Simulation.PressureMapCloud.valid_month(6) == 6
    assert Simulation.PressureMapCloud.valid_month("06") == 6
    assert Simulation.PressureMapCloud.valid_month("June") == 6
    assert Simulation.PressureMapCloud.valid_month("Jun") == 6
    assert (
        Simulation.PressureMapCloud.valid_month(
            datetime.strptime("2023-06-01", "%Y-%m-%d")
        )
        == 6
    )


def test_pressure_map_cloud_invalid_month():
    with pytest.raises(ValueError):
        Simulation.PressureMapCloud.valid_month(13)

    with pytest.raises(ValueError):
        Simulation.PressureMapCloud.valid_month("Invalid")


#######################################################################################
def test_config_serialization():
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".toml", delete=False
    ) as tmpfile:
        tmpfile_name = tmpfile.name
        a = NssConfig()
        create_toml(tmpfile_name, a)
        loaded_config = config_from_toml(tmpfile_name)

    assert loaded_config.model_dump() == a.model_dump()
