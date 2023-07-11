import numpy as np

from nuspacesim.config import NssConfig
from nuspacesim.simulation.geometry import region_geometry

from . import cpp_region_geometry


def test_rg():

    cfg = NssConfig()

    u = np.random.rand(int(1e6), 4)

    T = region_geometry.RegionGeom(cfg)
    T.throw(u.T)

    R = cpp_region_geometry.RegionGeom(cfg)
    R.run_geo_dmc_from_ran_array_nparray(u)

    Rmsk = R.evMasknpArray
    Tmsk = T.event_mask

    assert all(Rmsk == Tmsk)

    assert np.allclose(R.evArray["thetaS"][Rmsk], T.thetaS[Tmsk])
    assert np.allclose(R.evArray["phiS"][Rmsk], T.phiS[Tmsk])
    assert np.allclose(R.evArray["latS"][Rmsk], T.latS[Tmsk])
    assert np.allclose(R.evArray["longS"][Rmsk], T.longS[Tmsk])
    assert np.allclose(R.evArray["thetaTrSubV"][Rmsk], T.thetaTrSubV[Tmsk])
    assert np.allclose(R.evArray["costhetaTrSubV"][Rmsk], T.costhetaTrSubV[Tmsk])
    assert np.allclose(R.evArray["phiTrSubV"][Rmsk], T.phiTrSubV[Tmsk])
    assert np.allclose(R.evArray["thetaTrSubN"][Rmsk], T.thetaTrSubN[Tmsk])
    assert np.allclose(R.evArray["costhetaTrSubN"][Rmsk], T.costhetaTrSubN[Tmsk])
    assert np.allclose(R.evArray["betaTrSubN"][Rmsk], T.betaTrSubN[Tmsk])
    assert np.allclose(R.evArray["losPathLen"][Rmsk], T.losPathLen[Tmsk])
    assert np.allclose(R.evArray["thetaTrSubV"][Rmsk], T.thetaTrSubV[Tmsk])
    assert np.allclose(R.evArray["costhetaTrSubV"][Rmsk], T.costhetaTrSubV[Tmsk])
    assert np.allclose(R.evArray["elevAngVSubN"][Rmsk], T.elevAngVSubN[Tmsk])
    assert np.allclose(R.evArray["aziAngVSubN"][Rmsk], T.aziAngVSubN[Tmsk])

    assert not np.allclose(R.evArray["thetaS"][Rmsk], 1 + T.thetaS[Tmsk])

    assert np.allclose(R.betas(), T.betas())
    assert np.allclose(R.beta_rad(), T.beta_rad())
    assert np.allclose(R.thetas(), T.thetas())
    assert np.allclose(R.pathLens(), T.pathLens())

    triggers = np.random.uniform(size=int(1e6))[Tmsk]
    costheta = np.random.normal(size=int(1e6))[Tmsk]
    tauexitprob = np.random.uniform(size=int(1e6))[Tmsk]
    threshold = 1 / 10

    Rmci, Rmcig, Rpass = R.mcintegral(triggers, costheta, tauexitprob, threshold)
    Tmci, Tmcig, Tpass = T.mcintegral(triggers, costheta, tauexitprob, threshold)

    assert np.allclose(Rmci, Tmci)
    assert np.allclose(Rmcig, Tmcig)
    assert np.allclose(Rpass, Tpass)


if __name__ == "__main__":
    test_rg()
