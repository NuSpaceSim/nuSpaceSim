import numpy as np
import pytest
from astropy.time import Time

from nuspacesim.config import NssConfig
from nuspacesim.simulation.geometry import region_geometry


@pytest.fixture
def nss_config_event():
    conf = NssConfig()
    conf.simulation.source_RA = 100
    conf.simulation.source_DEC = 0
    conf.detector.altitude = 33.0
    conf.detector.detlat = 0.0
    conf.detector.detlong = 10.0
    conf.detector.sun_moon_cuts = True
    conf.detector.sun_alt_cut = -12
    conf.detector.moon_alt_cut = 0
    conf.detector.MoonMinPhaseAngleCut = np.radians(90)

    conf.simulation.N: int = 10000
    conf.simulation.theta_ch_max: float = np.radians(3.0)
    conf.simulation.det_mode: str = "ToO"
    conf.simulation.source_RA: float = 0
    conf.simulation.source_DEC: float = 0
    conf.simulation.source_date: str = "2022-03-21T00:00:00.000"
    conf.simulation.source_date_format: str = "isot"
    conf.simulation.source_obst: float = 24 * 60 * 60
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
