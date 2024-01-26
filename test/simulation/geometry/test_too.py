import numpy as np
import pytest
from astropy.time import Time

from nuspacesim.config import NssConfig
from nuspacesim.simulation.geometry import too


@pytest.fixture
def nss_config_event():
    conf = NssConfig()
    conf.simulation.too.source_RA = np.radians(22)
    conf.simulation.too.source_DEC = np.radians(-45)
    conf.detector.initial_position.altitude = 33.0
    conf.detector.initial_position.latitude = np.radians(0.0)
    conf.detector.initial_position.longitude = np.radians(10.0)
    conf.detector.sun_moon.sun_moon_cuts = True
    conf.detector.sun_moon.sun_alt_cut = np.radians(-12)
    conf.detector.sun_moon.moon_alt_cut = np.radians(0)
    conf.detector.sun_moon.moon_min_phase_angle_cut = np.radians(90)

    conf.simulation.thrown_events: int = 10000
    conf.simulation.max_cherenkov_angle: float = np.radians(3.0)
    conf.simulation.mode: str = "ToO"
    conf.simulation.too.source_RA: float = 0.0
    conf.simulation.too.source_DEC: float = 0.0
    conf.simulation.too.source_date: str = "2022-06-02T01:00:00.000"
    conf.simulation.too.source_date_format: str = "isot"
    conf.simulation.too.source_obst: float = 24 * 60 * 60
    return conf


@pytest.fixture
def too_event(nss_config_event):
    too_event = too.ToOEvent(nss_config_event)
    return too_event


def test_too_construction(nss_config_event, too_event):
    assert too_event.sun_alt_cut == nss_config_event.detector.sun_moon.sun_alt_cut
    assert too_event.moon_alt_cut == nss_config_event.detector.sun_moon.moon_alt_cut
    assert (
        too_event.MoonMinPhaseAngleCut
        == nss_config_event.detector.sun_moon.moon_min_phase_angle_cut
    )

    assert too_event.sourceOBSTime == nss_config_event.simulation.too.source_obst

    assert too_event.eventtime.utc.isot == nss_config_event.simulation.too.source_date
    assert (
        too_event.eventcoords.icrs.ra.deg == nss_config_event.simulation.too.source_RA
    )
    assert (
        too_event.eventcoords.icrs.dec.deg == nss_config_event.simulation.too.source_DEC
    )

    assert too_event.detcords.lat.rad == pytest.approx(
        nss_config_event.detector.initial_position.latitude
    )
    assert too_event.detcords.lon.rad == pytest.approx(
        nss_config_event.detector.initial_position.longitude
    )
    assert too_event.detcords.height.value == pytest.approx(
        nss_config_event.detector.initial_position.altitude * 1000
    )


def test_moon_phase_angle(too_event):
    fullmoon_date = "2022-11-08"
    newmoon_date = "2022-11-24"
    fullmoon = Time(fullmoon_date, format="iso", scale="utc")
    newmoon = Time(newmoon_date, format="iso", scale="utc")
    assert too_event.moon_phase_angle(fullmoon).value == pytest.approx(0, abs=1e-1)
    assert too_event.moon_phase_angle(newmoon).value == pytest.approx(
        np.radians(180), abs=1e-1
    )


def test_sun_moon_cuts(too_event):
    fullmoon_date = "2022-11-08"
    newmoon_date = "2022-11-24"

    # Case sun down
    fullmoon = Time(fullmoon_date, format="iso", scale="utc")
    newmoon = Time(newmoon_date, format="iso", scale="utc")
    assert too_event.sun_moon_cut(newmoon)
    assert not too_event.sun_moon_cut(fullmoon)

    # Case sun up
    newmoon_day = Time(newmoon_date + "T12:00:00", format="isot", scale="utc")
    fullmoon_day = Time(fullmoon_date + "T12:00:00", format="isot", scale="utc")
    assert not too_event.sun_moon_cut(newmoon_day)
    assert not too_event.sun_moon_cut(fullmoon_day)

    # Case Moon up, sun down, almost new moon
    darkmoon_date = "2022-11-29"
    dark = Time(darkmoon_date + "T22:00:00", format="isot", scale="utc")
    assert too_event.sun_moon_cut(dark)
