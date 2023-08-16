import numpy as np

import nuspacesim.constants as const
import nuspacesim.simulation.eas_optical.atmospheric_models as atm


def test_searchsort():
    h = np.linspace(0.0, 100.0)

    i = np.zeros_like(h, dtype=int)
    for j in range(1, len(const.std_atm_geopotential_height)):
        i[const.std_atm_geopotential_height[j] <= h] = j

    assert np.all(
        (const.std_atm_geopotential_height[i] <= h)
        & (const.std_atm_geopotential_height[i + 1] > h)
    )

    p = np.linspace(const.std_atm_ground_pressure, 1.01325e5 * 3.88501e-6)

    i = np.zeros_like(p, dtype=int)
    for j in range(1, len(const.std_atm_pressure)):
        i[const.std_atm_pressure[j] >= p] = j

    assert np.all(
        (const.std_atm_pressure[i] >= p) & (const.std_atm_pressure[i + 1] < p)
    )


def test_altitude_pressure_altitude():
    z = np.linspace(0.0, 100.0, 50000)

    p = atm.us_std_atm_pressure_from_altitude(z)
    z1 = atm.us_std_atm_altitude_from_pressure(p)
    assert np.allclose(z, z1)
    assert np.all(np.isreal(z1))


def test_pressure_altitude_pressure():
    p = np.linspace(const.std_atm_ground_pressure, 1e-7, 50000)

    z = atm.us_std_atm_altitude_from_pressure(p)
    p1 = atm.us_std_atm_pressure_from_altitude(z)
    assert np.allclose(p, p1)
    assert np.all(np.isreal(p1))
