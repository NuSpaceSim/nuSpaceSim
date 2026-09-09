"""Tests for CphotAng shower propagation length integration.

Validates:
- Shower propagation length method wrapper correctness
- Defensive input validation
"""

import numpy as np
import pytest

from nuspacesim.simulation.eas_optical.cphotang import CphotAng

# ---------------------------------------------------------------------------
# 1. Shower propagation length method wrapper
# ---------------------------------------------------------------------------


def test_shower_propagation_length_scalar():
    """Shower propagation length method handles scalar inputs."""
    cphotang = CphotAng(detector_altitude=525.0)
    decay_alt = 10.0
    beta = np.radians(20.0)
    Xtarg = 3882.56  # g/cm² for 1e12 eV shower

    L = cphotang.shower_propagation_length(decay_alt, beta, Xtarg)

    assert isinstance(L, (float, np.floating))
    assert np.isfinite(L)
    assert L > 0.0


def test_shower_propagation_length_uses_cphotang_earth_radius():
    """Shower propagation length method uses CphotAng's RadE parameter."""
    cphotang = CphotAng(detector_altitude=525.0)
    decay_alt = 10.0
    beta = np.radians(20.0)
    Xtarg = 3882.56

    # Oracle should use cphotang.RadE (6378.14 km)
    L = cphotang.shower_propagation_length(decay_alt, beta, Xtarg)

    # Verify by calling with explicit R parameter
    from nuspacesim.simulation.eas_optical.propagation import (
        shower_propagation_length,
    )

    L_explicit = shower_propagation_length(decay_alt, beta, Xtarg, R=cphotang.RadE)

    np.testing.assert_allclose(L, L_explicit, rtol=1e-14)


def test_shower_propagation_length_passes_kwargs():
    """Shower propagation length method passes additional kwargs to underlying function."""
    cphotang = CphotAng(detector_altitude=525.0)
    decay_alt = 10.0
    beta = np.radians(20.0)
    Xtarg = 3882.56

    # Should accept Nqmax, Niter, etc.
    L = cphotang.shower_propagation_length(decay_alt, beta, Xtarg, Nqmax=16, Niter=2)

    assert np.isfinite(L)
    assert L > 0.0


# ---------------------------------------------------------------------------
# 3. Defensive validation
# ---------------------------------------------------------------------------


def test_shower_propagation_length_rejects_nonfinite_decay_altitude():
    """Shower propagation length method rejects NaN/Inf decay altitude."""
    cphotang = CphotAng(detector_altitude=525.0)

    with pytest.raises(ValueError, match="decay_altitude must be finite"):
        cphotang.shower_propagation_length(np.nan, np.radians(20.0), 3882.56)

    with pytest.raises(ValueError, match="decay_altitude must be finite"):
        cphotang.shower_propagation_length(np.inf, np.radians(20.0), 3882.56)


def test_shower_propagation_length_rejects_nonfinite_beta():
    """Shower propagation length method rejects NaN/Inf beta."""
    cphotang = CphotAng(detector_altitude=525.0)

    with pytest.raises(ValueError, match="beta must be finite"):
        cphotang.shower_propagation_length(10.0, np.nan, 3882.56)

    with pytest.raises(ValueError, match="beta must be finite"):
        cphotang.shower_propagation_length(10.0, np.inf, 3882.56)


def test_shower_propagation_length_rejects_nonpositive_xtarg():
    """Shower propagation length method rejects non-positive target slant depth."""
    cphotang = CphotAng(detector_altitude=525.0)

    with pytest.raises(ValueError, match="Xtarg must be finite and positive"):
        cphotang.shower_propagation_length(10.0, np.radians(20.0), -100.0)

    with pytest.raises(ValueError, match="Xtarg must be finite and positive"):
        cphotang.shower_propagation_length(10.0, np.radians(20.0), 0.0)

    with pytest.raises(ValueError, match="Xtarg must be finite and positive"):
        cphotang.shower_propagation_length(10.0, np.radians(20.0), np.nan)


# ---------------------------------------------------------------------------
# 4. Physical behavior
# ---------------------------------------------------------------------------


def test_shower_propagation_length_decreases_with_altitude():
    """Propagation length decreases as decay altitude increases."""
    cphotang = CphotAng(detector_altitude=525.0)
    beta = np.radians(20.0)
    Xtarg = 3882.56

    decay_alts = np.array([5.0, 10.0, 15.0, 20.0])
    Ls = np.array(
        [cphotang.shower_propagation_length(alt, beta, Xtarg) for alt in decay_alts]
    )

    # Should be monotonically decreasing
    diffs = np.diff(Ls)
    assert np.all(diffs <= 1e-6), f"Propagation length increased: diffs={diffs}"


def test_shower_propagation_length_vertical_shower():
    """Oracle handles vertical showers (beta=0)."""
    cphotang = CphotAng(detector_altitude=525.0)
    decay_alt = 10.0
    beta = 0.0
    Xtarg = 3882.56

    L = cphotang.shower_propagation_length(decay_alt, beta, Xtarg, Nqmax=16, Niter=2)

    assert np.isfinite(L)
    assert L > 0.0


def test_shower_propagation_length_high_altitude():
    """Oracle handles high-altitude showers."""
    cphotang = CphotAng(detector_altitude=525.0)
    decay_alt = 100.0
    beta = np.radians(20.0)
    Xtarg = 3882.56

    L = cphotang.shower_propagation_length(decay_alt, beta, Xtarg, Nqmax=16, Niter=2)

    assert np.isfinite(L)
    # Should be small (little atmosphere remaining)
    assert L < 200.0


# ---------------------------------------------------------------------------
# 5. Batch processing
# ---------------------------------------------------------------------------


def test_shower_propagation_length_batch():
    """Shower propagation length method handles batch of showers."""
    cphotang = CphotAng(detector_altitude=525.0)

    decay_alts = np.array([5.0, 10.0, 15.0, 20.0])
    betas = np.radians(np.array([10.0, 20.0, 30.0, 40.0]))
    Xtarg = 3882.56

    Ls = cphotang.shower_propagation_length(decay_alts, betas, Xtarg)

    assert Ls.shape == (4,)
    assert np.all(np.isfinite(Ls))
    assert np.all(Ls > 0.0)


# ---------------------------------------------------------------------------
# 6. Performance characteristic
# ---------------------------------------------------------------------------


def test_shower_propagation_length_performance():
    """Shower propagation length method handles large batches efficiently."""
    import time

    cphotang = CphotAng(detector_altitude=525.0)

    n = 10000
    decay_alts = np.random.uniform(0.5, 15.0, n)
    betas = np.radians(np.random.uniform(1.0, 45.0, n))
    Xtarg = 3882.56

    start = time.time()
    Ls = cphotang.shower_propagation_length(decay_alts, betas, Xtarg)
    elapsed = time.time() - start

    assert Ls.shape == (n,)
    assert np.all(np.isfinite(Ls))
    # Should be fast (< 1s for 10k showers)
    assert elapsed < 1.0, f"Oracle too slow: {elapsed:.3f}s for {n} showers"
