"""Phase A gate tests for the Hillas angular-distribution kernel.

These 8 tests validate the math kernel correctness before proceeding
to the Phase B drop-in replacement.
"""

import numpy as np
import pytest

from nuspacesim.simulation.eas_optical.hillas_kernel import (
    hillas_cdf,
    hillas_delta_n_analytic,
    hillas_delta_n_cdf_diff,
    hillas_delta_n_numerical,
    hillas_dndu,
    pade11,
)
from nuspacesim.simulation.eas_optical.hillas_params import HILLAS_DEFAULT


# -----------------------------------------------------------------------
# 1. pade11 matches exact at small theta
# -----------------------------------------------------------------------
def test_pade11_matches_exact_at_small_theta():
    """Relative error < 1e-6 for θ < 0.1 rad.

    At very small θ (< 1e-3), 2(1-cos θ) suffers catastrophic cancellation,
    so we compare the Padé against 2·sin²(θ/2) which is numerically stable.
    """
    theta = np.linspace(1e-6, 0.1, 1000)
    exact = 4.0 * np.sin(theta / 2.0) ** 2  # stable form of 2(1-cos θ)
    approx = pade11(theta)
    rel_err = np.abs(approx - exact) / exact
    assert np.all(rel_err < 1e-6), f"Max rel err = {rel_err.max():.2e}"


# -----------------------------------------------------------------------
# 2. pade11 vs 2(1-cos) full range
# -----------------------------------------------------------------------
def test_pade11_vs_2_1_cos_full_range():
    """Relative error < 0.3% up to θ = 0.5 rad."""
    theta = np.linspace(0.01, 0.5, 1000)
    exact = 2.0 * (1.0 - np.cos(theta))
    approx = pade11(theta)
    rel_err = np.abs(approx - exact) / exact
    assert np.all(rel_err < 3e-3), f"Max rel err = {rel_err.max():.4e}"


# -----------------------------------------------------------------------
# 3. V1 equals V2 to roundoff
# -----------------------------------------------------------------------
def test_v1_equals_v2_roundoff():
    """V1 (analytic) ≡ V2 (CDF-diff) to rtol=1e-14."""
    p = HILLAS_DEFAULT
    u0 = np.array([0.0, 0.01, 0.1, 0.2, 0.3, p.u_star - 0.01, p.u_star + 0.01])
    u1 = u0 + 0.05

    v1 = hillas_delta_n_analytic(u0, u1, p)
    v2 = hillas_delta_n_cdf_diff(u0, u1, p)

    np.testing.assert_allclose(
        v1, v2, rtol=1e-14, atol=0, err_msg="V1 and V2 differ beyond roundoff"
    )


# -----------------------------------------------------------------------
# 4. V1 matches V5 gold
# -----------------------------------------------------------------------
def test_v1_matches_v5_gold():
    """V1 vs V5 to rtol=1e-13, including straddle cases."""
    p = HILLAS_DEFAULT
    # Include cases below, above, and straddling z_star
    u0 = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, p.u_star - 0.05, 0.5, 1.0])
    u1 = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, p.u_star + 0.05, 1.0, 2.0])

    v1 = hillas_delta_n_analytic(u0, u1, p)
    v5 = hillas_delta_n_numerical(u0, u1, p, n_quad=64)

    np.testing.assert_allclose(
        v1, v5, rtol=1e-13, atol=0, err_msg="V1 and V5 (gold) differ"
    )


# -----------------------------------------------------------------------
# 5. Straddling z_star: split == sum of two non-straddling pieces
# -----------------------------------------------------------------------
def test_straddling_z_star():
    """Split at u_star == sum of two non-straddling pieces."""
    p = HILLAS_DEFAULT
    u_lo = np.array([0.1])
    u_hi = np.array([0.6])

    combined = hillas_delta_n_analytic(u_lo, u_hi, p)
    part1 = hillas_delta_n_analytic(u_lo, np.array([p.u_star]), p)
    part2 = hillas_delta_n_analytic(np.array([p.u_star]), u_hi, p)

    np.testing.assert_allclose(
        combined, part1 + part2, rtol=1e-15, err_msg="Straddle split failed"
    )


# -----------------------------------------------------------------------
# 6. Small theta regime: dn/du(u→0) ≈ A·exp(z★/λ₁)
# -----------------------------------------------------------------------
def test_small_theta_regime():
    """dn/du at u=0 should equal A·exp(z★/λ₁)."""
    p = HILLAS_DEFAULT
    expected = p.A * np.exp(p.z_star / p.lam1)
    actual = hillas_dndu(0.0, p)
    np.testing.assert_allclose(actual, expected, rtol=1e-15)


# -----------------------------------------------------------------------
# 7. Large u underflow: graceful underflow to 0
# -----------------------------------------------------------------------
def test_large_u_underflow():
    """dn/du should underflow gracefully to 0 for very large u."""
    p = HILLAS_DEFAULT
    u_large = np.array([1e4, 1e6, 1e8])
    result = hillas_dndu(u_large, p)
    assert np.all(result >= 0.0), "dn/du went negative"
    assert np.all(np.isfinite(result)), "dn/du produced inf/nan"
    assert np.all(result < 1e-100), f"dn/du not small enough: {result}"


# -----------------------------------------------------------------------
# 8. CDF monotonic: F(u) non-decreasing
# -----------------------------------------------------------------------
def test_cdf_monotonic():
    """F(u) = Δn(0, u) must be non-decreasing."""
    u_vals = np.linspace(0.0, 5.0, 500)
    cdf_vals = hillas_cdf(u_vals)
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-15), f"CDF decreased: min diff = {diffs.min():.2e}"
