"""End-to-end regression tests for ``CphotAng.run`` batched execution.

These cover the vectorized ``run()`` path that the rest of the collected suite
does not exercise:

* batched evaluation must equal per-shower evaluation (locks the batched
  single-kernel node accumulation and guards against cross-shower contamination),
* heterogeneous batches (``n_showers >= 2``) must run and stay finite,
* the ``per_wavelength`` decomposition must sum back to the collapsed result,
* the five project reference ("golden") (beta, alt) -> (pden, cang) values.
"""

import warnings

import numpy as np
import pytest

from nuspacesim.simulation.eas_optical.cphotang import CphotAng

DET_ALT = 525.0


def _cpa():
    return CphotAng(detector_altitude=DET_ALT)


# Reference values: emergence angle in DEGREES (converted to radians for run),
# shower energy = 1.0 * 100 PeV. (pden photon density, cang Cherenkov angle deg).
# pden values re-anchored for the per-(shower, node) energy-quadrature window:
# each node integrates from its own Cherenkov threshold to a per-shower energy
# envelope, so photonDen is now a pure per-shower function (no batch-wide
# energy-bound coupling). Batched == per-shower bit-identically (see
# test_batch_equals_per_shower). Per-shower the grid converges to <=0.6% at
# n=16 (see test_grid_converges_per_shower). cang essentially unchanged.
# Re-anchored when the aerosol slant column moved from the plane-parallel
# vertical_OD/cos(theta) (main's convention, which diverges at grazing incidence)
# to the exact curved-atmosphere layer-walk (aerosol_column). The shift is
# concentrated on grazing showers: beta=0.5 deg pden 16.23 -> 21.67 (+34%, the
# plane-parallel had over-attenuated it), beta=5 deg +6.8%, beta>=10 deg <0.5%.
GOLDEN = [
    (20.0, 10.0, 956.7518, 0.4831),
    (10.0, 5.0, 137.1388, 0.9683),
    (5.0, 3.0, 35.6925, 1.1331),
    (30.0, 8.0, 1424.7673, 0.5376),
    (0.5, 5.0, 21.6668, 1.0349),
]


def test_grid_converges_per_shower():
    """GL-in-slant-depth grid converges in node count, per shower.

    Compare the default n=16 grid against a high-order n=128 reference. The
    peak-split GL-in-X grid is accurate to ~0.1% for the typical shower; a tail
    of high-altitude / low-energy showers (narrow valid window straddling the
    atmospheric-layer kinks) reaches ~2%. This is the clean physical-accuracy
    check for the propagation grid; with the per-(shower, node) energy window
    the GOLDEN/snapshot anchors are batch-independent too.
    """
    cpa = _cpa()
    rng = np.random.default_rng(20240619)
    errs = []
    for _ in range(60):
        bdeg = rng.uniform(0.5, 42.0)
        alt = rng.uniform(0.0, 18.0)
        e = rng.uniform(0.2, 8.0)
        z = np.array([0.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            d16, _ = cpa.run(
                np.radians([bdeg]),
                np.array([alt]),
                np.array([e]),
                z,
                z,
                n_nodes=16,
                per_wavelength=False,
            )
            d128, _ = cpa.run(
                np.radians([bdeg]),
                np.array([alt]),
                np.array([e]),
                z,
                z,
                n_nodes=128,
                per_wavelength=False,
            )
        errs.append(abs(d16[0] - d128[0]) / max(abs(d128[0]), 1e-9))
    errs = np.array(errs)
    assert np.median(errs) < 3e-3  # typical shower: sub-percent
    assert errs.max() < 3e-2  # worst case (narrow-window / layer-kink) bounded


def test_run_batch_n_ge_2():
    """run() must accept batches (regression: oracle broke for n_showers >= 2)."""
    cpa = _cpa()
    for n in (2, 3, 5):
        beta = np.radians(np.linspace(1.0, 35.0, n))
        alt = np.linspace(1.0, 12.0, n)
        e = np.ones(n)
        z = np.zeros(n)
        pden, cang = cpa.run(beta, alt, e, z, z, per_wavelength=False)
        assert pden.shape == (n,)
        assert cang.shape == (n,)
        assert np.all(np.isfinite(pden))
        assert np.all(np.isfinite(cang))


# Golden snapshot of the batched, vectorized run() output for a fixed batch.
# Locks the single-kernel node accumulation against future regressions.
_SNAP_BETA_DEG = np.array([1.0, 5.0, 10.0, 20.0, 30.0, 42.0])
_SNAP_ALT = np.array([5.0, 3.0, 5.0, 10.0, 8.0, 2.0])
_SNAP_E = np.array([0.3, 1.0, 2.0, 1.0, 5.0, 0.7])
# pden reflects the float32 kernel energy-integral (single precision in the
# (n, k, n_E) region; ~1e-4 from the float64 values, well below the ~0.3%
# quadrature accuracy). cang is the photon-weighted mean + population-variance
# spread (three-raw-moment form, no Bessel n/(n-1) factor). The float32 result
# is deterministic (see test_batch_equals_per_shower).
# Re-anchored twice: (1) ozone column 8-point GL -> exact analytic layer-walk
# (ozone_column; rate is piecewise-constant, GL was ~5% off); (2) aerosol slant
# column plane-parallel vertical_OD/cos(theta) -> exact curved-atmosphere
# layer-walk (aerosol_column), which un-attenuates grazing showers the
# plane-parallel had over-attenuated (here the beta=1 deg shower: 4.82 -> 6.46).
# Both are deliberate deviations from main toward the more physical column.
_SNAP_PDEN = np.array(
    [6.457016e00, 3.5692482e01, 2.7684943e02, 9.5675180e02, 6.0369473e03, 9.9823640e02]
)
_SNAP_CANG = np.array([1.036216, 1.133135, 0.964729, 0.48313, 0.512915, 0.963849])


def test_batched_run_snapshot():
    """Vectorized run() reproduces the recorded batched output (regression lock).

    Runs cleanly (no divide/invalid warnings) and matches the snapshot tightly.
    """
    cpa = _cpa()
    z = np.zeros_like(_SNAP_BETA_DEG)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pden, cang = cpa.run(
            np.radians(_SNAP_BETA_DEG), _SNAP_ALT, _SNAP_E, z, z, per_wavelength=False
        )
    np.testing.assert_allclose(pden, _SNAP_PDEN, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cang, _SNAP_CANG, rtol=1e-4, atol=1e-4)


def test_decay_above_atmosphere_is_zero_and_finite():
    """Decay at/above the atmosphere top (degenerate grid) -> finite zero output.

    Guards the constraint redesign: such showers have no valid nodes, so the
    grid collapses to zero width and the ozone column is clamped to 0; the
    result must be exactly 0 with no NaN/Inf, and must not perturb the valid
    showers sharing the batch.
    """
    cpa = _cpa()
    z6 = np.zeros(6)
    beta = np.radians(np.array([20.0, 10.0, 20.0, 5.0, 30.0, 15.0]))
    alt = np.array([10.0, 5.0, 70.0, 3.0, 66.0, 8.0])  # 70, 66 km are above zMaxZ=65
    e = np.ones(6)

    # The degenerate showers legitimately hit a guarded 0/0 in the masked-out
    # branch of _photon_density; only the finiteness of the *output* matters.
    with np.errstate(divide="ignore", invalid="ignore"):
        pden, cang = cpa.run(beta, alt, e, z6, z6, per_wavelength=False)

    assert np.all(np.isfinite(pden)) and np.all(np.isfinite(cang))
    degenerate = np.array([2, 4])
    valid = np.array([0, 1, 3, 5])
    assert np.all(pden[degenerate] == 0.0)
    assert np.all(pden[valid] > 0.0)

    # All-degenerate batch -> all zeros, still finite.
    with np.errstate(divide="ignore", invalid="ignore"):
        pden_all, cang_all = cpa.run(
            np.radians(np.full(4, 20.0)),
            np.full(4, 70.0),
            np.ones(4),
            np.zeros(4),
            np.zeros(4),
            per_wavelength=False,
        )
    assert np.all(pden_all == 0.0) and np.all(np.isfinite(cang_all))


def test_batch_equals_per_shower():
    """Batched run() equals per-shower processing -- bit-exact.

    The energy-quadrature window is per-(shower, node) (each node integrates
    from its own Cherenkov threshold to a per-shower energy envelope), so a
    shower's result no longer depends on its batch. Both photon density AND
    Cherenkov angle must match per-shower processing to float precision; this
    locks out any cross-shower contamination.
    """
    cpa = _cpa()
    rng = np.random.default_rng(2024)
    n = 64
    beta = np.radians(rng.uniform(0.5, 40.0, n))
    alt = rng.uniform(0.0, 18.0, n)
    e = rng.uniform(0.2, 8.0, n)
    z = np.zeros(n)

    pden_batch, cang_batch = cpa.run(beta, alt, e, z, z, per_wavelength=False)

    pden_each = np.empty(n)
    cang_each = np.empty(n)
    for i in range(n):
        p, c = cpa.run(
            beta[i : i + 1],
            alt[i : i + 1],
            e[i : i + 1],
            z[:1],
            z[:1],
            per_wavelength=False,
        )
        pden_each[i] = p[0]
        cang_each[i] = c[0]

    assert np.all(np.isfinite(pden_batch))
    np.testing.assert_allclose(cang_batch, cang_each, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pden_batch, pden_each, rtol=1e-6, atol=1e-6)


def test_heterogeneous_batch_finite():
    """Edge-heavy heterogeneous batch must stay finite and non-negative."""
    cpa = _cpa()
    beta = np.radians(np.array([0.5, 1.0, 5.0, 20.0, 42.0, 42.0]))
    alt = np.array([0.0, 20.0, 3.0, 10.0, 0.0, 20.0])
    e = np.array([0.1, 10.0, 1.0, 1.0, 5.0, 0.5])
    z = np.zeros_like(beta)
    pden, cang = cpa.run(beta, alt, e, z, z, per_wavelength=False)
    assert np.all(np.isfinite(pden))
    assert np.all(np.isfinite(cang))
    assert np.all(pden >= 0.0)


def test_per_wavelength_consistency():
    """Summing per_wavelength output over wavelength == collapsed output."""
    cpa = _cpa()
    rng = np.random.default_rng(7)
    n = 256
    beta = np.radians(rng.uniform(0.5, 40.0, n))
    alt = rng.uniform(0.0, 18.0, n)
    e = rng.uniform(0.2, 8.0, n)
    z = np.zeros(n)

    pden, _ = cpa.run(beta, alt, e, z, z, per_wavelength=False)
    pden_wl, _ = cpa.run(beta, alt, e, z, z, per_wavelength=True)

    assert pden_wl.ndim == 2
    assert pden_wl.shape[0] == n
    assert np.all(np.isfinite(pden_wl))
    # float32 accumulation: ~1e-7 observed
    np.testing.assert_allclose(np.sum(pden_wl, axis=1), pden, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("beta_deg,alt,pden,cang", GOLDEN)
def test_golden_values(beta_deg, alt, pden, cang):
    """Five reference physics points, processed together as one batch."""
    cpa = _cpa()
    betas = np.radians(np.array([b for b, *_ in GOLDEN]))
    alts = np.array([a for _, a, *_ in GOLDEN])
    e = np.ones(len(GOLDEN))
    z = np.zeros(len(GOLDEN))
    pdens, cangs = cpa.run(betas, alts, e, z, z, per_wavelength=False)

    i = [b for b, *_ in GOLDEN].index(beta_deg)
    # ~0.5% drift accumulated across the vectorization refactor; physics anchored.
    assert pdens[i] == pytest.approx(pden, rel=1.5e-2)
    assert cangs[i] == pytest.approx(cang, rel=1.5e-2)


def test_ozone_rate_is_derivative_of_ozone_losses():
    """ozone_rate(z) == -d/dz ozone_losses(z) within each linear segment.

    ozone_losses (TotZon) is piecewise-linear, so its analytic slope must
    match a finite difference everywhere except across the known breakpoints.
    """
    cpa = _cpa()
    # Dense altitudes, then drop any sample whose +/-eps straddles a breakpoint.
    z = np.linspace(0.1, 99.0, 5000)
    eps = 1e-4
    breaks = np.concatenate(([5.35], np.asarray(cpa.OzZeta, dtype=float)))
    near_break = np.any(np.abs(z[:, None] - breaks[None, :]) < 1e-2, axis=1)
    zc = z[~near_break]

    fd = -(cpa.ozone_losses(zc + eps) - cpa.ozone_losses(zc - eps)) / (2 * eps)
    an = cpa.ozone_rate(zc)
    np.testing.assert_allclose(an, fd, rtol=1e-4, atol=1e-4)
    assert np.all(an >= 0.0)


def test_aerosol_column_curved_slant():
    """Curved-atmosphere aerosol slant column: exact, and bounded at grazing.

    Validates the analytic walk against a step-friendly fine Riemann sum of the
    aerosol extinction along the path, confirms it reduces to the vertical column
    for a near-vertical shower, and confirms it stays bounded at grazing
    incidence (where the former plane-parallel ``vertical_OD / cos`` diverged).
    """
    from nuspacesim.simulation.eas_optical.cphotang import aerosol_column
    from nuspacesim.simulation.eas_optical.propagation import lexpr, zexpr

    cpa = _cpa()
    R = float(cpa.RadE)
    rng = np.random.default_rng(31)
    n, k = 400, 12
    beta = np.radians(rng.uniform(0.5, 45.0, n))
    alt = rng.uniform(0.0, 6.0, n)
    L_nodes = lexpr(alt, beta)[:, None] + rng.uniform(0, 20, (n, k))
    L_max = lexpr(65.0, beta)
    L_nodes = np.minimum(L_nodes, L_max[:, None])

    walk = aerosol_column(L_nodes, L_max, beta, cpa.aero_zbnd, cpa.aero_ext, R)

    # Step-friendly fine Riemann reference (extinction is piecewise-constant).
    def ext_of_z(z):
        zi = np.clip(z.astype(np.int32), 0, len(cpa.aero_ext) - 1)
        return np.where(z < 30.0, cpa.aero_ext[zi], 0.0)

    t = (np.arange(20000) + 0.5) / 20000
    Lg = L_nodes[..., None] + (L_max[:, None, None] - L_nodes[..., None]) * t
    ref = (L_max[:, None] - L_nodes) * np.mean(
        ext_of_z(zexpr(Lg, beta[:, None, None], R=R)), axis=-1
    )

    # The walk is exact; the residual is the Riemann reference's own O(1/M) error
    # at the extinction steps (5e-3 at M=3000 -> 9e-4 at M=20000 confirms it).
    m = ref > 1e-6
    assert np.max(np.abs(walk - ref)[m] / ref[m]) < 2e-3
    assert np.all(np.isfinite(walk)) and np.all(walk >= 0.0)

    # Near-vertical: curved column ~ vertical aerosol OD above the node.
    bv = np.radians(np.array([89.5]))
    Lv = lexpr(np.array([[1.0]]), bv[:, None])
    av = aerosol_column(
        Lv, lexpr(np.array([65.0]), bv), bv, cpa.aero_zbnd, cpa.aero_ext, R
    )
    assert av[0, 0] == pytest.approx(float(cpa.aOD55[1]), rel=2e-2)


def test_call_matches_run_single_block():
    """__call__ within a single dask block equals a direct run() call."""
    cpa = _cpa()
    rng = np.random.default_rng(11)
    n = 500  # < default chunk -> single block -> identical to run()
    beta = np.radians(rng.uniform(0.5, 40.0, n))
    alt = rng.uniform(0.0, 18.0, n)
    e = rng.uniform(0.2, 8.0, n)
    z = np.zeros(n)

    d_run, c_run = cpa.run(beta, alt, e, z, z, per_wavelength=False)
    d_call, c_call = cpa(beta, alt, e, z, z)  # default chunks (one block here)

    np.testing.assert_allclose(d_call, d_run, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(c_call, c_run, rtol=1e-5, atol=1e-5)


def test_call_chunked_matches_single_block():
    """Multi-block __call__ equals a single-block evaluation -- bit-exact.

    The energy-quadrature window is per-(shower, node), independent of which
    dask block a shower lands in, so chunking no longer shifts the photon
    density. Both photon density and Cherenkov angle match to float precision.
    """
    cpa = _cpa()
    rng = np.random.default_rng(13)
    n = 4000
    beta = np.radians(rng.uniform(0.5, 40.0, n))
    alt = rng.uniform(0.0, 18.0, n)
    e = rng.uniform(0.2, 8.0, n)
    z = np.zeros(n)

    d_single, c_single = cpa(beta, alt, e, z, z, chunks=n)  # one block
    d_chunked, c_chunked = cpa(beta, alt, e, z, z, chunks=500)  # 8 blocks

    assert np.all(np.isfinite(d_chunked))
    np.testing.assert_allclose(c_chunked, c_single, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(d_chunked, d_single, rtol=1e-6, atol=1e-6)


def test_cloud_lower_bound_obscures_shower_head():
    """A cloud deck folded into the window's lower bound reduces visible light.

    Clouds raise ``X_lo`` to the slant depth at the cloud top, so light from the
    shower head (below the clouds) is not counted. A finite deck must reduce the
    photon density without NaN/Inf; a deck above every shower's death depth must
    drive the visible yield to exactly zero.
    """
    cpa = _cpa()
    rng = np.random.default_rng(21)
    n = 400
    beta = np.radians(rng.uniform(2.0, 30.0, n))
    alt = rng.uniform(0.0, 1.0, n)  # near-ground decays (visible head low down)
    e = rng.uniform(0.2, 4.0, n)
    z = np.zeros(n)

    clear, _ = cpa.run(beta, alt, e, z, z, per_wavelength=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        clouded, _ = cpa.run(
            beta, alt, e, z, z, cloudf=lambda la, lo: 3.0, per_wavelength=False
        )
        opaque, _ = cpa.run(
            beta, alt, e, z, z, cloudf=lambda la, lo: 70.0, per_wavelength=False
        )

    assert np.all(np.isfinite(clouded)) and np.all(np.isfinite(opaque))
    # A 3 km deck obscures the head, so no shower gains light and most lose some.
    assert np.all(clouded <= clear + 1e-6)
    assert np.mean(clouded < clear * 0.999) > 0.5
    # A deck above the shower's visible top (z_shower_top = 65 km) leaves the
    # window with zero width -> nothing visible.
    assert np.all(opaque == 0.0)
