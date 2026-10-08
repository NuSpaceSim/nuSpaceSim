"""The ``on_profile`` observation hook of CphotAng.run / CphotAng.__call__ / EAS.

The hook must (a) leave every return value unchanged, (b) deliver fixed-shape
``(n_showers, n_nodes)`` profiles, and (c) carry an ``index`` that maps each
profile row back to the receiving caller's event arrays.
"""

import numpy as np

from nuspacesim.config import NssConfig
from nuspacesim.simulation.eas_optical.cphotang import CphotAng, ShowerProfile
from nuspacesim.simulation.eas_optical.eas import EAS

N_NODES = 12


def _events(n=40, seed=5):
    rng = np.random.default_rng(seed)
    beta = np.radians(rng.uniform(0.5, 40.0, n))
    alt = rng.uniform(0.0, 18.0, n)
    energy = 10 ** rng.uniform(-1.5, 1.0, n)  # 100 PeV units
    return beta, alt, energy, np.zeros(n), np.zeros(n)


def test_run_hook_observes_without_changing_results():
    cpa = CphotAng(detector_altitude=525.0)
    ev = _events()
    base = cpa.run(*ev, n_nodes=N_NODES, per_wavelength=False)
    seen = []
    got = cpa.run(*ev, n_nodes=N_NODES, per_wavelength=False, on_profile=seen.append)
    for a, b in zip(base, got):
        assert np.array_equal(a, b)

    (p,) = seen
    n = len(ev[0])
    assert isinstance(p, ShowerProfile)
    assert np.array_equal(p.index, np.arange(n))
    for f in ShowerProfile.NODE_FIELDS:
        assert getattr(p, f).shape == (n, N_NODES)
        assert np.all(np.isfinite(getattr(p, f)))
    # Ordered along the axis; strictly so iff the maximum is inside the window
    # (otherwise one GL panel collapses onto the window edge).
    assert np.all(np.diff(p.X, axis=1) >= 0)
    assert np.all(np.diff(p.z, axis=1) >= 0)  # up-going: altitude increases
    half = N_NODES // 2
    collapsed = (p.X[:, 0] == p.X[:, half - 1]) | (p.X[:, half] == p.X[:, -1])
    strict = np.all(np.diff(p.X, axis=1) > 0, axis=1)
    assert np.array_equal(strict, ~collapsed)
    assert strict.any() and collapsed.any()  # both cases occur in this sample
    assert np.all(p.N >= 0)
    assert np.array_equal(p.altDec, ev[1]) and np.array_equal(p.showerEnergy, ev[2])
    assert np.array_equal(p.beta, np.maximum(ev[0], np.radians(1.0)))


def test_call_hook_matches_run_and_keeps_returns():
    cpa = CphotAng(detector_altitude=525.0)
    ev = _events()
    base = cpa(*ev, n_nodes=N_NODES, serial=True)
    seen = []
    got = cpa(*ev, n_nodes=N_NODES, serial=True, on_profile=seen.append)
    for a, b in zip(base, got):
        assert np.array_equal(a, b)

    ref = []
    cpa.run(*ev, n_nodes=N_NODES, per_wavelength=False, on_profile=ref.append)
    (p,), (r,) = seen, ref
    for f in ("index", "beta", "altDec", "showerEnergy", *ShowerProfile.NODE_FIELDS):
        assert np.array_equal(getattr(p, f), getattr(r, f)), f


def test_eas_hook_index_maps_to_in_bounds_events():
    cfg = NssConfig()
    eas = EAS(cfg)
    beta, alt, energy, lat, lon = _events(30)
    alt[[0, 7, 19]] = [-1.0, 25.0, 21.0]  # out of bounds: never simulated
    seen = []
    base = eas(beta, alt, energy, lat, lon, serial=True)
    got = eas(beta, alt, energy, lat, lon, serial=True, on_profile=seen.append)
    for a, b in zip(base, got):
        assert np.array_equal(a, b)

    (p,) = seen
    expected = np.flatnonzero((alt >= 0) & (alt <= 20))
    assert np.array_equal(p.index, expected)
    assert np.array_equal(p.altDec, alt[p.index])
    assert np.array_equal(p.showerEnergy, energy[p.index])
    assert p.N.shape == (expected.size, cfg.simulation.cherenkov_quadrature.n_nodes)
