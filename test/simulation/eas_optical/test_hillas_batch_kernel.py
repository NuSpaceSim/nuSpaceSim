"""Tests for batch-oriented single-integral Hillas kernel.

Validates:
- Energy quadrature correctness
- Hillas regime constants and CDF computation
- Batch kernel correctness and invariants
- Defensive input validation
"""

import numpy as np
import pytest

from nuspacesim.simulation.eas_optical.hillas_batch_kernel import (
    delta_nphots_single_integral_NE16,
    energy_quadrature_split_at_Eswitch,
    g_theta_exact,
    gauss_legendre_interval,
    hillas_F_u_inplace,
    hillas_regime_constants,
    precompute_kernel_NE16,
)

# ---------------------------------------------------------------------------
# 1. Gauss-Legendre interval mapping
# ---------------------------------------------------------------------------


def test_gauss_legendre_interval_integrates_constant():
    """GL weights integrate f(x)=1 over [a, b]."""
    a, b = 2.0, 10.0
    _, weights = gauss_legendre_interval(n=16, a=a, b=b)
    np.testing.assert_allclose(np.sum(weights), b - a, rtol=1e-14, atol=0.0)


def test_gauss_legendre_interval_rejects_invalid_bounds():
    """Invalid bounds raise ValueError."""
    with pytest.raises(ValueError, match="Require b > a"):
        gauss_legendre_interval(n=8, a=10.0, b=10.0)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        gauss_legendre_interval(n=0, a=1.0, b=2.0)


# ---------------------------------------------------------------------------
# 2. Energy quadrature split at Eswitch
# ---------------------------------------------------------------------------


def test_energy_quadrature_split_integrates_constant():
    """Two-panel weights integrate f(E)=1 over [Emin, Emax]."""
    Emin = 25.0
    Emax = 1e5
    Eswitch = 1e3
    _, W_E = energy_quadrature_split_at_Eswitch(Emin, Eswitch, Emax, n_per_panel=8)
    np.testing.assert_allclose(np.sum(W_E), Emax - Emin, rtol=1e-12, atol=0.0)


def test_energy_quadrature_split_has_16_nodes():
    """Default n_per_panel=8 produces 16 total nodes."""
    E_nodes, W_E = energy_quadrature_split_at_Eswitch(10.0, 100.0, 1e4, n_per_panel=8)
    assert E_nodes.shape == (16,)
    assert W_E.shape == (16,)


def test_energy_quadrature_split_rejects_invalid_energy_bounds():
    """Invalid energy bounds raise ValueError."""
    with pytest.raises(ValueError, match="Require Emax > Emin"):
        energy_quadrature_split_at_Eswitch(100.0, 50.0, 100.0)
    with pytest.raises(ValueError, match="Emin, Eswitch, Emax must be finite"):
        energy_quadrature_split_at_Eswitch(np.nan, 50.0, 100.0)


# ---------------------------------------------------------------------------
# 3. Angular factor g(theta)
# ---------------------------------------------------------------------------


def test_g_theta_exact_matches_2_1_minus_cos():
    """g(θ) = 2(1 - cos θ) at representative angles."""
    theta = np.array([0.0, 0.01, 0.1, 0.5, 1.0])
    expected = 2.0 * (1.0 - np.cos(theta))
    actual = g_theta_exact(theta)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


# ---------------------------------------------------------------------------
# 4. Hillas regime constants
# ---------------------------------------------------------------------------


def test_hillas_regime_constants_validates_positive_parameters():
    """Regime constants reject non-positive A, lam1, lam2."""
    with pytest.raises(ValueError, match="A, lam1, lam2 must be strictly positive"):
        hillas_regime_constants(A=0.0, z0=0.5, lam1=0.4, lam2=0.3)
    with pytest.raises(ValueError, match="A, lam1, lam2 must be strictly positive"):
        hillas_regime_constants(A=1.0, z0=0.5, lam1=-0.4, lam2=0.3)


def test_hillas_regime_constants_validates_finite_parameters():
    """Regime constants reject NaN/Inf."""
    with pytest.raises(ValueError, match="Hillas parameters must be finite"):
        hillas_regime_constants(A=np.nan, z0=0.5, lam1=0.4, lam2=0.3)


# ---------------------------------------------------------------------------
# 5. Hillas CDF in-place computation
# ---------------------------------------------------------------------------


def test_hillas_F_u_inplace_non_negative():
    """F(u) is non-negative for all u >= 0."""
    const = hillas_regime_constants(A=0.777, z0=0.59, lam1=0.478, lam2=0.380)
    u = np.linspace(0.0, 5.0, 100)
    out = np.empty_like(u)
    hillas_F_u_inplace(u, const, out)
    assert np.all(out >= 0.0), f"F(u) went negative: min={out.min()}"


def test_hillas_F_u_inplace_monotonic():
    """F(u) is non-decreasing."""
    const = hillas_regime_constants(A=0.777, z0=0.59, lam1=0.478, lam2=0.380)
    u = np.linspace(0.0, 5.0, 1000)
    out = np.empty_like(u)
    hillas_F_u_inplace(u, const, out)
    diffs = np.diff(out)
    assert np.all(diffs >= -1e-15), f"F(u) decreased: min diff={diffs.min()}"


def test_hillas_F_u_inplace_rejects_shape_mismatch():
    """F_u_inplace requires u and out to have same shape."""
    const = hillas_regime_constants(A=0.777, z0=0.59, lam1=0.478, lam2=0.380)
    u = np.linspace(0.0, 1.0, 10)
    out = np.empty(5)
    with pytest.raises(ValueError, match="u and out must have same shape"):
        hillas_F_u_inplace(u, const, out)


# ---------------------------------------------------------------------------
# 6. Precompute kernel
# ---------------------------------------------------------------------------


def test_precompute_kernel_NE16_produces_16_nodes():
    """Precompute produces nL+nH energy nodes (default 3+8=11)."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    n_E = pre["nL"] + pre["nH"]
    assert pre["E_nodes"].shape == (n_E,)
    assert pre["W_E"].shape == (n_E,)
    assert pre["hi_mask"].shape == (n_E,)
    assert pre["hi_mask"].sum() == pre["nH"]
    # (E/21)^2 is derived in the kernel per slab, not stored in pre.
    assert "Efac" not in pre
    # explicit symmetric case still works
    pre16 = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3, nL=8, nH=8)
    assert pre16["E_nodes"].shape == (16,)


def test_precompute_kernel_regime_split_correct():
    """High-energy mask correctly identifies E >= Eswitch."""
    Eswitch = 1e3
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=Eswitch)
    E_nodes = pre["E_nodes"]
    hi_mask = pre["hi_mask"]
    # First 8 nodes should be < Eswitch, last 8 should be >= Eswitch
    assert np.all(E_nodes[~hi_mask] < Eswitch)
    assert np.all(E_nodes[hi_mask] >= Eswitch)


# ---------------------------------------------------------------------------
# 7. Batch kernel correctness
# ---------------------------------------------------------------------------


def test_delta_nphots_single_integral_runs_without_error():
    """Batch kernel runs without error on valid inputs."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    n_E = pre["nL"] + pre["nH"]
    p = 100
    thetaC = np.full(p, 0.08)
    sigsum = np.ones(p)
    mean_w = np.ones((p, n_E)) * 0.1
    W_extra = np.ones((p, n_E))
    out = delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)
    assert out.shape == (p,)
    assert np.all(np.isfinite(out))


def test_delta_nphots_single_integral_non_negative():
    """Batch kernel produces non-negative outputs."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    n_E = pre["nL"] + pre["nH"]
    p = 50
    thetaC = np.linspace(0.01, 0.1, p)
    sigsum = np.ones(p)
    mean_w = np.ones((p, n_E)) * 0.1
    W_extra = np.ones((p, n_E))
    out = delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)
    assert np.all(out >= 0.0), f"Kernel produced negative output: min={out.min()}"


def test_delta_nphots_single_integral_monotonic_in_thetaC():
    """Output non-decreasing as thetaC increases (fixed other params)."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    n_E = pre["nL"] + pre["nH"]
    thetaC = np.linspace(0.01, 0.1, 50)
    sigsum = np.ones_like(thetaC)
    mean_w = np.ones(n_E) * 0.1
    W_extra = np.ones(n_E)
    out = delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)
    # Allowing small numerical noise
    diffs = np.diff(out)
    assert np.all(
        diffs >= -1e-10
    ), f"Output decreased with thetaC: min diff={diffs.min()}"


def test_delta_nphots_single_integral_zero_for_zero_sigsum():
    """Output is zero when sigsum=0."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    n_E = pre["nL"] + pre["nH"]
    p = 10
    thetaC = np.full(p, 0.08)
    sigsum = np.zeros(p)
    mean_w = np.ones((p, n_E)) * 0.1
    W_extra = np.ones((p, n_E))
    out = delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)
    np.testing.assert_allclose(out, 0.0, atol=1e-15)


def test_delta_nphots_single_integral_broadcastable_mean_w():
    """Kernel handles broadcastable mean_w (n_E,) instead of (p, n_E)."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    n_E = pre["nL"] + pre["nH"]
    p = 20
    thetaC = np.full(p, 0.08)
    sigsum = np.ones(p)
    mean_w = np.ones(n_E) * 0.1  # (n_E,) broadcastable
    W_extra = np.ones(n_E)  # (n_E,) broadcastable
    out = delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)
    assert out.shape == (p,)
    assert np.all(np.isfinite(out))


def test_delta_nphots_single_integral_handles_empty_batch():
    """Kernel handles p=0 gracefully."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    n_E = pre["nL"] + pre["nH"]
    thetaC = np.array([])
    sigsum = np.array([])
    mean_w = np.ones(n_E) * 0.1
    W_extra = np.ones(n_E)
    out = delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# 8. Defensive validation
# ---------------------------------------------------------------------------


def test_delta_nphots_single_integral_rejects_shape_mismatch():
    """Kernel rejects mismatched thetaC and sigsum shapes."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    thetaC = np.full(10, 0.08)
    sigsum = np.ones(5)
    mean_w = np.ones(16) * 0.1
    W_extra = np.ones(16)
    with pytest.raises(ValueError, match="thetaC and sigsum must have same length"):
        delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)


def test_delta_nphots_single_integral_rejects_nonfinite_thetaC():
    """Kernel rejects NaN/Inf in thetaC."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    thetaC = np.array([0.08, np.nan, 0.09])
    sigsum = np.ones(3)
    mean_w = np.ones(16) * 0.1
    W_extra = np.ones(16)
    with pytest.raises(ValueError, match="thetaC contains NaN/Inf"):
        delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)


def test_delta_nphots_single_integral_rejects_nonfinite_sigsum():
    """Kernel rejects NaN/Inf in sigsum."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    thetaC = np.full(3, 0.08)
    sigsum = np.array([1.0, np.inf, 1.0])
    mean_w = np.ones(16) * 0.1
    W_extra = np.ones(16)
    with pytest.raises(ValueError, match="sigsum contains NaN/Inf"):
        delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre)


def test_delta_nphots_single_integral_rejects_invalid_p_sub():
    """Kernel rejects non-positive p_sub."""
    pre = precompute_kernel_NE16(eCthres=25.0, Eshow=1e8, Eswitch=1e3)
    thetaC = np.full(10, 0.08)
    sigsum = np.ones(10)
    mean_w = np.ones(16) * 0.1
    W_extra = np.ones(16)
    with pytest.raises(ValueError, match="p_sub must be a positive integer"):
        delta_nphots_single_integral_NE16(thetaC, sigsum, mean_w, W_extra, pre, p_sub=0)
