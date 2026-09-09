"""Defensive and correctness tests for log-energy GL quadrature helpers."""

import numpy as np
import pytest

from nuspacesim.simulation.eas_optical.quadrature import (
    gauss_legendre_logE_nodes_weights,
    gauss_legendre_logE_two_panel,
)


def test_gauss_legendre_loge_integrates_constant():
    """Weights integrate f(E)=1 over [E_min, E_max]."""
    E_min = 10.0
    E_max = 1.0e5
    _, weights = gauss_legendre_logE_nodes_weights(E_min, E_max, n=16)
    np.testing.assert_allclose(np.sum(weights), E_max - E_min, rtol=1e-14, atol=0.0)


def test_gauss_legendre_two_panel_integrates_constant():
    """Two-panel weights integrate f(E)=1 over [E_min, E_max]."""
    E_min = 15.0
    E_max = 2.0e4
    _, weights = gauss_legendre_logE_two_panel(
        E_min, E_max, delta=np.log(100.0), n1=8, n2=8
    )
    np.testing.assert_allclose(np.sum(weights), E_max - E_min, rtol=1e-13, atol=0.0)


def test_gauss_legendre_loge_rejects_invalid_bounds():
    """Invalid bounds raise ValueError with explicit message."""
    with pytest.raises(ValueError, match="E_max must be > E_min"):
        gauss_legendre_logE_nodes_weights(100.0, 100.0, n=8)
    with pytest.raises(ValueError, match="E_min must be a finite positive scalar"):
        gauss_legendre_logE_nodes_weights(0.0, 100.0, n=8)


def test_gauss_legendre_two_panel_rejects_invalid_orders_and_delta():
    """Order and panel-width guards reject invalid inputs."""
    with pytest.raises(ValueError, match="n1 must be > 0"):
        gauss_legendre_logE_two_panel(10.0, 100.0, delta=1.0, n1=0, n2=8)
    with pytest.raises(ValueError, match="delta must be a finite positive scalar"):
        gauss_legendre_logE_two_panel(10.0, 100.0, delta=0.0, n1=8, n2=8)
