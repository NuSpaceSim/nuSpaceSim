import numpy as np
import pytest
from astropy.time import Time

from nuspacesim.config import NssConfig
from nuspacesim.simulation.geometry import region_geometry


@pytest.fixture
def nss_config_event():
    conf = NssConfig()
    conf.simulation.target.source_RA = np.radians(100)
    conf.simulation.target.source_DEC = 0
    conf.detector.initial_position.altitude = 33.0
    conf.detector.initial_position.latitude = 0.0
    conf.detector.initial_position.longitude = np.radians(10.0)
    conf.detector.sun_moon.sun_moon_cuts = True
    conf.detector.sun_moon.sun_alt_cut = np.radians(-12)
    conf.detector.sun_moon.moon_alt_cut = 0
    conf.detector.sun_moon.moon_min_phase_angle_cut = np.radians(90)

    conf.simulation.thrown_events: int = 10000
    conf.simulation.max_cherenkov_angle: float = np.radians(3.0)
    conf.simulation.mode: str = "Target"
    conf.simulation.target.source_RA: float = 0.0
    conf.simulation.target.source_DEC: float = 0.0
    conf.simulation.target.source_date: str = "2022-03-21T00:00:00.000"
    conf.simulation.target.source_date_format: str = "isot"
    conf.simulation.target.source_obst: float = 24 * 60 * 60
    return conf


@pytest.fixture
def region(nss_config_event):
    region_geom = region_geometry.RegionGeomToO(nss_config_event)
    return region_geom


def test_too_traj(region):
    u = np.array([0, 1, 2], dtype=float)
    region.throw(u)
    assert np.rad2deg(region.sourceNadRad[0]) == pytest.approx(
        np.rad2deg(region.sourceNadRad[1]) - 360 / 365, abs=0.01
    )


def test_too_times(region):
    assert len(region.generate_times(1000)) == 1000
