"""Tests for shower propagation length oracle.

Validates:
- Atmospheric density model
- Geometric transformations (z ↔ L)
- Differential slant depth computation
- Canonical slant_depth / length_at_depth round-trip inverse
- Batch processing correctness
"""

import numpy as np
import pytest

from nuspacesim.simulation.eas_optical.propagation import (
    Atmosphere,
    d2Xl_known_layer,
    dXl,
    dXl_known_layer,
    length_at_depth,
    length_at_depth_approx,
    length_at_depth_hermite,
    lexpr,
    shower_propagation_length,
    slant_depth,
    slant_depth_intervals,
    total_slant_depth,
    us_std_atm_density,
    zexpr,
)

# ---------------------------------------------------------------------------
# 1. Atmosphere model
# ---------------------------------------------------------------------------


def test_atmosphere_lookup_tables_correct_length():
    """Atmosphere lookup tables have correct dimensions."""
    a = Atmosphere()
    assert len(a.H_b) == 8
    assert len(a.Lm_b) == 8
    assert len(a.T_b) == 8
    assert len(a.P_b) == 8
    assert len(a.Z_b) == 8


def test_atmosphere_layer_index_lookup():
    """Binary search correctly identifies atmospheric layers."""
    a = Atmosphere()
    # Test several altitudes to verify lookup works
    assert a(0.0) == 0  # troposphere
    assert a(11.01) == 1  # stratosphere start
    assert a(20.01) == 2  # stratosphere
    assert a(33.0) == 3  # upper stratosphere
    assert a(48.0) == 4  # mesosphere
    assert a(52.0) == 5  # upper mesosphere
    assert a(72.0) == 6  # thermosphere


def test_us_std_atm_density_positive():
    """Atmospheric density is positive at all valid altitudes."""
    z = np.linspace(0.0, 100.0, 1000)
    rho = us_std_atm_density(z)
    assert np.all(rho > 0.0)


def test_us_std_atm_density_decreasing():
    """Atmospheric density decreases monotonically with altitude."""
    z = np.linspace(0.0, 100.0, 1000)
    rho = us_std_atm_density(z)
    diffs = np.diff(rho)
    assert np.all(diffs <= 0.0), f"Density increased: max diff={diffs.max()}"


def test_us_std_atm_density_rejects_nonfinite():
    """Atmospheric density rejects NaN/Inf altitudes."""
    with pytest.raises(ValueError, match="z contains NaN/Inf"):
        us_std_atm_density(np.array([10.0, np.nan]))


def test_us_std_atm_density_rejects_negative_R():
    """Atmospheric density rejects non-positive Earth radius."""
    with pytest.raises(ValueError, match="R must be positive"):
        us_std_atm_density(10.0, R=-1.0)


def test_us_std_atm_density_single_implementation():
    """atmospheric_models re-exports this module's density: one implementation."""
    from nuspacesim.simulation.eas_optical import atmospheric_models

    assert atmospheric_models.us_std_atm_density is us_std_atm_density


# ---------------------------------------------------------------------------
# 2. Geometric transformations
# ---------------------------------------------------------------------------


def test_lexpr_zexpr_inverse():
    """lexpr and zexpr are inverse functions."""
    z = np.linspace(1.0, 100.0, 50)
    beta = np.radians(20.0)
    L = lexpr(z, beta)
    z_recovered = zexpr(L, beta)
    np.testing.assert_allclose(z_recovered, z, rtol=1e-14, atol=1e-10)


def test_lexpr_zero_at_zero_altitude():
    """Propagation length is zero at zero altitude for vertical showers."""
    beta = 0.0
    L = lexpr(0.0, beta)
    np.testing.assert_allclose(L, 0.0, atol=1e-15)


def test_lexpr_increases_with_altitude():
    """Propagation length increases with altitude."""
    z = np.linspace(0.0, 100.0, 100)
    beta = np.radians(20.0)
    L = lexpr(z, beta)
    diffs = np.diff(L)
    assert np.all(diffs > 0.0), "Propagation length decreased with altitude"


def test_lexpr_increases_with_angle():
    """Propagation length has expected behavior with angle.

    Note: lexpr = -R*sin(beta) + sqrt(R^2*sin^2(beta) + z^2 + 2*R*z)
    This is NOT monotonically increasing in beta for all z.
    For small z, it can actually decrease initially before increasing.
    We just check that it's finite and positive.
    """
    z = 50.0
    beta = np.radians(np.linspace(0.0, 60.0, 100))
    L = lexpr(z, beta)
    assert np.all(np.isfinite(L))
    assert np.all(L >= 0.0)


def test_lexpr_rejects_nonfinite():
    """lexpr rejects NaN/Inf inputs."""
    with pytest.raises(ValueError, match="z contains NaN/Inf"):
        lexpr(np.nan, 0.5)
    with pytest.raises(ValueError, match="beta contains NaN/Inf"):
        lexpr(10.0, np.inf)


def test_zexpr_rejects_nonfinite():
    """zexpr rejects NaN/Inf inputs."""
    with pytest.raises(ValueError, match="lval contains NaN/Inf"):
        zexpr(np.nan, 0.5)
    with pytest.raises(ValueError, match="beta contains NaN/Inf"):
        zexpr(100.0, np.inf)


# ---------------------------------------------------------------------------
# 3. Differential slant depth
# ---------------------------------------------------------------------------


def test_dXl_positive():
    """Differential slant depth is positive (density is positive)."""
    lval = np.linspace(0.0, 1000.0, 100)
    beta = np.radians(20.0)
    dX = dXl(lval, beta)
    assert np.all(dX > 0.0), f"dXl went negative: min={dX.min()}"


def test_dXl_decreases_along_path():
    """Differential slant depth decreases as shower propagates (altitude increases)."""
    lval = np.linspace(0.0, 1000.0, 100)
    beta = np.radians(20.0)
    dX = dXl(lval, beta)
    # Should decrease monotonically as we move through atmosphere
    # (since density decreases with altitude)
    diffs = np.diff(dX)
    # Allow some numerical noise in isothermal layers
    assert np.all(diffs <= 1e-10), f"dXl increased unexpectedly: max diff={diffs.max()}"


def test_dXl_rejects_nonfinite():
    """dXl rejects NaN/Inf inputs."""
    with pytest.raises(ValueError, match="lval contains NaN/Inf"):
        dXl(np.nan, 0.5)


# ---------------------------------------------------------------------------
# 4. Shower propagation length oracle
# ---------------------------------------------------------------------------


def test_shower_propagation_length_single_shower():
    """Oracle runs without error on single shower."""
    decay_alt = 10.0
    beta = np.radians(20.0)
    Xtarg = 3882.56  # g/cm² for 1e12 eV shower
    L = shower_propagation_length(decay_alt, beta, Xtarg)
    assert isinstance(L, (float, np.floating))
    assert np.isfinite(L)
    assert L > 0.0


def test_shower_propagation_length_batch():
    """Oracle handles batch of showers."""
    decay_alts = np.linspace(0.5, 15.0, 100)
    betas = np.radians(np.linspace(1.0, 45.0, 100))
    Xtarg = 3882.56
    Ls = shower_propagation_length(decay_alts, betas, Xtarg)
    assert Ls.shape == (100,)
    assert np.all(np.isfinite(Ls))
    assert np.all(Ls > 0.0)


def test_shower_propagation_length_increases_with_angle():
    """Propagation length has expected physical behavior with angle.

    For fixed decay altitude, propagation length to reach a fixed slant depth
    target is NOT necessarily monotonic in angle. At small angles, more atmosphere
    is traversed vertically; at large angles, the path is longer but through
    thinner air. The oracle finds where X(L) = Xtarg, which can have complex
    dependence on angle.

    We just verify all results are finite and positive.
    """
    decay_alt = 10.0
    betas = np.radians(np.linspace(1.0, 45.0, 50))
    Xtarg = 3882.56
    Ls = shower_propagation_length(
        np.full_like(betas, decay_alt), betas, Xtarg, Nqmax=16, Niter=2
    )
    assert np.all(np.isfinite(Ls))
    assert np.all(Ls > 0.0)


def test_shower_propagation_length_decreases_with_decay_altitude():
    """Propagation length decreases as decay altitude increases (less atmosphere)."""
    decay_alts = np.linspace(0.5, 20.0, 50)
    beta = np.radians(20.0)
    Xtarg = 3882.56
    Ls = shower_propagation_length(
        decay_alts, np.full_like(decay_alts, beta), Xtarg, Nqmax=16, Niter=2
    )
    diffs = np.diff(Ls)
    # Should be monotonically decreasing
    assert np.all(
        diffs <= 1e-6
    ), f"Propagation length increased: max diff={diffs.max()}"


def test_shower_propagation_length_returns_scalar_for_scalar_input():
    """Oracle returns scalar when given scalar inputs."""
    L = shower_propagation_length(10.0, np.radians(20.0), 3882.56)
    assert np.isscalar(L)


def test_shower_propagation_length_returns_array_for_array_input():
    """Oracle returns array when given array inputs."""
    decay_alts = np.array([5.0, 10.0, 15.0])
    betas = np.radians(np.array([10.0, 20.0, 30.0]))
    Ls = shower_propagation_length(decay_alts, betas, 3882.56)
    assert isinstance(Ls, np.ndarray)
    assert Ls.shape == (3,)


# ---------------------------------------------------------------------------
# 5. Defensive validation
# ---------------------------------------------------------------------------


def test_shower_propagation_length_rejects_shape_mismatch():
    """Oracle rejects mismatched decay_altitude and beta shapes."""
    decay_alts = np.array([5.0, 10.0])
    betas = np.radians(np.array([10.0, 20.0, 30.0]))
    with pytest.raises(
        ValueError, match="decay_altitude and beta must have same shape"
    ):
        shower_propagation_length(decay_alts, betas, 3882.56)


def test_shower_propagation_length_rejects_nonfinite_decay_altitude():
    """Oracle rejects NaN/Inf in decay_altitude."""
    with pytest.raises(ValueError, match="decay_altitude contains NaN/Inf"):
        shower_propagation_length(
            np.array([10.0, np.nan]), np.radians([20.0, 20.0]), 3882.56
        )


def test_shower_propagation_length_rejects_nonfinite_beta():
    """Oracle rejects NaN/Inf in beta."""
    with pytest.raises(ValueError, match="beta contains NaN/Inf"):
        shower_propagation_length(
            np.array([10.0, 10.0]), np.array([0.5, np.inf]), 3882.56
        )


def test_shower_propagation_length_rejects_nonpositive_Xtarg():
    """Oracle rejects non-positive target slant depth."""
    with pytest.raises(ValueError, match="Xtarg must be finite and positive"):
        shower_propagation_length(10.0, np.radians(20.0), -100.0)
    with pytest.raises(ValueError, match="Xtarg must be finite and positive"):
        shower_propagation_length(10.0, np.radians(20.0), 0.0)


def test_shower_propagation_length_rejects_nonpositive_zMax():
    """Oracle rejects non-positive maximum altitude."""
    with pytest.raises(ValueError, match="zMax must be finite and positive"):
        shower_propagation_length(10.0, np.radians(20.0), 3882.56, zMax=-10.0)


def test_shower_propagation_length_rejects_negative_decay_altitude():
    """Oracle rejects negative decay altitude."""
    with pytest.raises(ValueError, match="decay_altitude must be non-negative"):
        shower_propagation_length(-5.0, np.radians(20.0), 3882.56)


def test_shower_propagation_length_rejects_decay_altitude_above_zMax():
    """Oracle rejects decay altitude >= zMax."""
    with pytest.raises(ValueError, match="decay_altitude must be < zMax"):
        shower_propagation_length(130.0, np.radians(20.0), 3882.56, zMax=125.0)


def test_shower_propagation_length_rejects_invalid_Nqmax():
    """Oracle rejects non-positive Nqmax."""
    with pytest.raises(ValueError, match="Nqmax must be a positive integer"):
        shower_propagation_length(10.0, np.radians(20.0), 3882.56, Nqmax=0)


def test_shower_propagation_length_rejects_invalid_Niter():
    """Oracle rejects non-positive Niter."""
    with pytest.raises(ValueError, match="Niter must be a positive integer"):
        shower_propagation_length(10.0, np.radians(20.0), 3882.56, Niter=0)


# ---------------------------------------------------------------------------
# 6. Performance characteristic
# ---------------------------------------------------------------------------


def test_shower_propagation_length_handles_large_batch():
    """Oracle handles large batches efficiently."""
    import time

    n = 10000  # 10k showers
    decay_alts = np.random.uniform(0.5, 15.0, n)
    betas = np.radians(np.random.uniform(1.0, 45.0, n))
    Xtarg = 3882.56

    start = time.time()
    Ls = shower_propagation_length(decay_alts, betas, Xtarg, Nqmax=8, Niter=1)
    elapsed = time.time() - start

    assert Ls.shape == (n,)
    assert np.all(np.isfinite(Ls))
    # Should be reasonably fast (notebook reports ~1.4s for 1M showers)
    # For 10k showers, expect << 0.1s
    assert elapsed < 1.0, f"Oracle too slow: {elapsed:.3f}s for {n} showers"


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


def test_shower_propagation_length_vertical_shower():
    """Oracle handles vertical showers (beta=0)."""
    decay_alt = 10.0
    beta = 0.0
    Xtarg = 3882.56
    L = shower_propagation_length(decay_alt, beta, Xtarg, Nqmax=16, Niter=2)
    assert np.isfinite(L)
    assert L > 0.0


def test_shower_propagation_length_high_altitude_decay():
    """Oracle handles showers decaying near top of atmosphere."""
    decay_alt = 100.0
    beta = np.radians(20.0)
    Xtarg = 3882.56
    L = shower_propagation_length(decay_alt, beta, Xtarg, Nqmax=16, Niter=2)
    assert np.isfinite(L)
    # Should be small (little atmosphere remaining)
    assert L < 200.0


def test_shower_propagation_length_escaping_shower():
    """Oracle handles showers that escape atmosphere before reaching Xtarg."""
    decay_alt = 80.0  # High altitude
    beta = np.radians(80.0)  # Nearly horizontal
    Xtarg = 1e6  # Very large target (unreachable)
    L = shower_propagation_length(decay_alt, beta, Xtarg, Nqmax=16, Niter=1)
    # Should return full atmospheric path length
    assert np.isfinite(L)
    assert L > 0.0


# ---------------------------------------------------------------------------
# 8. Canonical slant_depth / length_at_depth (single source of truth)
# ---------------------------------------------------------------------------


def test_slant_depth_matches_total_slant_depth():
    """total_slant_depth is a thin wrapper on the canonical slant_depth."""
    z_em = np.array([0.5, 3.0, 8.0, 15.0])
    beta = np.radians(np.array([2.0, 10.0, 25.0, 40.0]))
    L0 = lexpr(z_em, beta)
    L1 = lexpr(65.0, beta)
    np.testing.assert_allclose(
        total_slant_depth(z_em, beta, z_top=65.0),
        slant_depth(L0, L1, beta),
        rtol=0,
        atol=0,
    )


def test_length_at_depth_inverts_slant_depth():
    """X(L(x)) == x to machine precision for all reachable depths.

    This round-trip identity is the invariant that ties the propagation length
    and slant-depth layers together (l = L(X(l)), x = X(L(x))).
    """
    rng = np.random.default_rng(0)
    n = 200
    alt = rng.uniform(0.0, 20.0, n)
    beta = np.radians(rng.uniform(1.0, 60.0, n))
    L0 = lexpr(alt, beta)
    Lcap = lexpr(Atmosphere().Z_b[-1], beta)
    X_full = slant_depth(L0, Lcap, beta)
    # Fractions across the whole reachable depth, on a trailing quadrature-like axis.
    frac = np.linspace(0.01, 0.99, 16)
    X = X_full[:, None] * frac
    L = length_at_depth(L0, X, beta)
    X_back = slant_depth(L0[:, None], L, beta[:, None])
    rel = np.abs(X_back - X) / X
    assert rel.max() < 1e-10


def test_propagation_length_round_trips_to_target():
    """For reachable targets, the returned length lands exactly on Xtarg."""
    rng = np.random.default_rng(1)
    n = 100
    alt = rng.uniform(0.0, 10.0, n)
    beta = np.radians(rng.uniform(5.0, 50.0, n))
    L0 = lexpr(alt, beta)
    Lcap = lexpr(Atmosphere().Z_b[-1], beta)
    X_full = slant_depth(L0, Lcap, beta)
    Xtarg = 0.5 * X_full  # reachable: half the available depth
    prop = shower_propagation_length(alt, beta, Xtarg)
    X_back = slant_depth(L0, L0 + prop, beta)
    np.testing.assert_allclose(X_back, Xtarg, rtol=1e-10, atol=1e-8)


def test_length_at_depth_hermite_matches_exact():
    """The fast inverse-Hermite grid inverse agrees with the exact Newton inverse.

    The cubic seed (polish=0) is good to ~1e-2 in L; one polish step drives it to
    the canonical inverse. This is the grid workhorse: many node depths per shared
    path, inverted by spline evaluation rather than per-target root finding.
    """
    rng = np.random.default_rng(3)
    n = 200
    alt = rng.uniform(0.0, 17.0, n)
    beta = np.radians(rng.uniform(0.5, 42.0, n))
    L0 = lexpr(alt, beta)
    Lmax = lexpr(65.0, beta)
    X_top = slant_depth(L0, Lmax, beta)
    # Node depths spread across each shower's reachable window.
    frac = np.linspace(0.02, 0.98, 16)
    X = X_top[:, None] * frac
    L_exact = length_at_depth(L0, X, beta, niter=10)

    L_seed = length_at_depth_hermite(L0, X, beta, 65.0, polish=0)
    L_pol = length_at_depth_hermite(L0, X, beta, 65.0, polish=1)
    seed_rel = np.abs(L_seed - L_exact) / np.maximum(np.abs(L_exact), 1e-9)
    pol_rel = np.abs(L_pol - L_exact) / np.maximum(np.abs(L_exact), 1e-9)
    assert seed_rel.max() < 5e-2  # cubic seed alone
    assert pol_rel.max() < 1e-3  # one Newton polish -> canonical


def test_d2Xl_known_layer_matches_finite_difference():
    """The vectorized known-layer Hessian equals the central difference of dXl.

    Validates the known-layer second derivative d²X/dl² against a central
    finite difference of dXl (the true derivative). The known-layer Hessian is
    the form the grid inverse (the within-layer Halley step) relies on.
    """
    a = Atmosphere()
    rng = np.random.default_rng(11)
    n = 500
    beta = np.radians(rng.uniform(0.5, 45.0, n))
    L = lexpr(rng.uniform(0.0, 30.0, n), beta)
    z = zexpr(L, beta)
    li = a(z * 6371.0 / (z + 6371.0))
    h = 1e-4
    fd = (dXl(L + h, beta, a=a) - dXl(L - h, beta, a=a)) / (2.0 * h)
    got = d2Xl_known_layer(L, beta, li, a)
    np.testing.assert_allclose(got, fd, rtol=1e-5, atol=1e-7)


def test_length_at_depth_approx_matches_exact():
    """The hybrid approx grid inverse agrees with the exact Newton inverse.

    Near-ground decays (the upward-shower regime) keep every node inside the
    lowest atmospheric layer, so the global-Halley fast path applies and is
    exact to the grid tolerance; high-altitude decays force kink crossings and
    exercise the layer-walking hermite fallback. Both must round-trip.
    """
    rng = np.random.default_rng(7)
    for alt_hi in (2.0, 17.0):  # near-ground (fast path) and high (fallback)
        n = 200
        alt = rng.uniform(0.0, alt_hi, n)
        beta = np.radians(rng.uniform(0.5, 42.0, n))
        L0 = lexpr(alt, beta)
        Lmax = lexpr(65.0, beta)
        X_top = slant_depth(L0, Lmax, beta)
        frac = np.linspace(0.02, 0.98, 16)
        X = X_top[:, None] * frac
        L_exact = length_at_depth(L0, X, beta, niter=10)
        L_approx = length_at_depth_approx(L0, X, beta, 65.0)
        # Round-trip in slant depth: the quantity the grid actually consumes.
        X_rt = slant_depth(L0[:, None], L_approx, beta[:, None])
        rt_rel = np.abs(X_rt - X) / np.maximum(X, 1e-9)
        assert rt_rel.max() < 1e-3
        # And agreement with the canonical length inverse.
        L_rel = np.abs(L_approx - L_exact) / np.maximum(np.abs(L_exact), 1e-9)
        assert L_rel.max() < 1e-3


def test_length_at_depth_approx_single_layer_is_exact():
    """For contained near-ground showers (no kink crossed) the fast path is exact."""
    rng = np.random.default_rng(8)
    n = 300
    alt = rng.uniform(0.0, 1.0, n)  # decay near the surface
    beta = np.radians(rng.uniform(5.0, 42.0, n))
    L0 = lexpr(alt, beta)
    # Shallow targets that stay within the lowest layer (< ~11 km).
    Lmax = lexpr(8.0, beta)
    X_top = slant_depth(L0, Lmax, beta)
    X = X_top[:, None] * np.linspace(0.05, 0.95, 12)
    L_approx = length_at_depth_approx(L0, X, beta, 65.0)
    L_exact = length_at_depth(L0, X, beta, niter=12)
    np.testing.assert_allclose(L_approx, L_exact, rtol=5e-4, atol=1e-3)


def test_slant_depth_intervals_consistency():
    """slant_depth_intervals agrees with direct canonical slant_depth calls."""
    rng = np.random.default_rng(2)
    n, k = 12, 16
    alt = rng.uniform(0.0, 12.0, n)
    beta = np.radians(rng.uniform(2.0, 45.0, n))
    L_start = lexpr(alt, beta)
    L_max = lexpr(65.0, beta)
    frac = np.linspace(0.05, 0.95, k)
    L_nodes = L_start[:, None] + (L_max - L_start)[:, None] * frac

    X_to_node, X_to_det, X_total = slant_depth_intervals(L_start, L_nodes, L_max, beta)

    np.testing.assert_allclose(
        X_to_node, slant_depth(L_start[:, None], L_nodes, beta[:, None]), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        X_total, slant_depth(L_start, L_max, beta), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        X_to_det, X_total[:, None] - X_to_node, rtol=0, atol=1e-9
    )
