import numpy as np
from nuspacesim.simulation.geometry import rg, region_geometry
from nuspacesim.config import NssConfig


def test_rg():

    cfg = NssConfig()

    u = np.random.uniform(0, 1.0, (int(1e6), 4))
    t = rg.geo_dmc(u, cfg)

    R = region_geometry.RegionGeom(cfg)
    R.run_geo_dmc_from_ran_array_nparray(u)

    assert all(R.evMasknpArray == t[1])

    # Rmsk = R.evMasknpArray
    # tmsk = t[1]

    # assert all(R.evArray["betaTrSubN"][Rmsk] == t[1])


if __name__ == "__main__":
    test_rg()
