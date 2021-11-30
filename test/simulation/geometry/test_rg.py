import numpy as np
from nuspacesim.simulation.geometry import rg, region_geometry
from nuspacesim.config import NssConfig


def test_rg():

    cfg = NssConfig()

    u = np.float32(np.random.rand(int(1e6), 4))
    T = rg.Geo(cfg)
    T.throw(u)

    R = region_geometry.RegionGeom(cfg)
    R.run_geo_dmc_from_ran_array_nparray(u)

    assert all(R.evMasknpArray == T.event_mask)

    Rmsk = R.evMasknpArray
    Tmsk = T.event_mask

    d = np.abs(R.evArray["betaTrSubN"][Rmsk] - T.betaTrSubN[Tmsk])
    i = np.argmax(d)
    print(d[i], R.evArray["betaTrSubN"][Rmsk][i], T.betaTrSubN[Tmsk][i])
    assert np.allclose(
        R.evArray["betaTrSubN"][Rmsk], T.betaTrSubN[Tmsk], rtol=1e-2, atol=1e-2
    )


if __name__ == "__main__":
    test_rg()
