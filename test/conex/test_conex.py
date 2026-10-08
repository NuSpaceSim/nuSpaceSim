"""CONEX output: fixed-shape schema, mapping from shower profiles, ROOT file."""

from pathlib import Path

import numpy as np
import pytest

from nuspacesim.conex import ConexWriter, build_showers, conex_path
from nuspacesim.conex.format import PROFILE, SHOWER_FIELDS, empty_header, shower_dtype
from nuspacesim.conex.writer import gaisser_hillas
from nuspacesim.simulation.eas_optical.cphotang import ShowerProfile
from nuspacesim.simulation.eas_optical.propagation import lexpr, slant_depth

K = 12
GH = (-60.0, 700.0, 3e8, 1e-6, -0.01, 70.0)  # X0, Xmax, Nmax, p3, p2, p1


def _profile(n=4):
    """Up-going showers with GH profiles on a GL grid split at Xmax."""
    x, _ = np.polynomial.legendre.leggauss(K // 2)
    lo, peak, hi = 20.0, GH[1], 1700.0
    grid = np.concatenate(
        [
            0.5 * (peak + lo) + 0.5 * (peak - lo) * x,
            0.5 * (hi + peak) + 0.5 * (hi - peak) * x,
        ]
    )
    X = np.tile(grid, (n, 1))
    alt = np.linspace(0.5, 3.0, n)
    return ShowerProfile(
        index=np.arange(n),
        beta=np.radians(np.linspace(5.0, 30.0, n)),
        altDec=alt,
        showerEnergy=np.full(n, 0.5),
        X=X,
        z=alt[:, None] + np.linspace(0.1, 8.0, K),
        N=gaisser_hillas(X, *GH),
    )


def test_schema_is_fixed_shape():
    dt = shower_dtype(K)
    assert dt.names == tuple(f.name for f in SHOWER_FIELDS)
    for f in SHOWER_FIELDS:
        assert dt[f.name].shape == ((K,) if f.shape == PROFILE else f.shape)
    assert empty_header()["OutputVersion"][0] == np.float32(2.51)


def test_build_showers_maps_the_profile():
    p = _profile()
    out = build_showers(p, np.random.default_rng(0))
    assert out.dtype == shower_dtype(K) and len(out) == 4
    L0 = lexpr(0.0, p.beta)
    Xfirst = slant_depth(L0, lexpr(p.altDec, p.beta), p.beta)
    np.testing.assert_allclose(out["X"], p.X + Xfirst[:, None], rtol=1e-6)
    np.testing.assert_allclose(out["N"], p.N, rtol=1e-6)
    np.testing.assert_allclose(out["H"], p.z * 1e3, rtol=1e-6)
    np.testing.assert_allclose(
        out["D"], (lexpr(p.z, p.beta[:, None]) - L0[:, None]) * 1e3, rtol=1e-6
    )
    np.testing.assert_allclose(out["lgE"], np.log10(0.5) + 17.0)
    np.testing.assert_allclose(out["zenith"], 90.0 + np.degrees(p.beta), rtol=1e-6)
    # The fit recovers the generating Xmax (shifted by Xfirst).
    np.testing.assert_allclose(out["Xmax"], GH[1] + Xfirst, rtol=1e-4)
    assert np.all(out["nX"] == K)


def test_build_showers_drops_unresolved_maxima():
    p = _profile(3)
    N = p.N.copy()
    N[1] = np.linspace(1.0, 2.0, K)  # still growing at the window top
    N[2] = np.linspace(2.0, 1.0, K)  # maximum before the window start
    out = build_showers(
        ShowerProfile(p.index, p.beta, p.altDec, p.showerEnergy, p.X, p.z, N),
        np.random.default_rng(0),
    )
    assert len(out) == 1


def test_written_file_reads_back(tmp_path):
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "conex_run.root"
    p = _profile()
    ConexWriter(path, rng=np.random.default_rng(1))(p)
    expected = build_showers(p, np.random.default_rng(1))
    with uproot.open(path) as f:
        shower = f["Shower"]
        assert shower["X"].typename == f"float[{K}]"  # fixed, not jagged
        got = shower.arrays(library="np")
        for name in expected.dtype.names:
            np.testing.assert_array_equal(np.asarray(got[name]), expected[name])
        assert f["Header"]["Particle"].array(library="np")[0] == 100


@pytest.mark.parametrize(
    "output,expected",
    [("nss.fits", "conex_nss.root"), ("out/run.fits", "out/conex_run.root")],
)
def test_conex_path(output, expected):
    assert conex_path(output) == Path(expected)
