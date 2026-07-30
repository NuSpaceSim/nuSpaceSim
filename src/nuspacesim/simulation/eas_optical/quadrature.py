"""Gauss-Legendre quadrature helpers for log-energy integration."""

from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

__all__ = [
    "gauss_legendre_logE_nodes_weights",
    "gauss_legendre_logE_two_panel",
]


def _require_finite_positive_scalar(name, value):
    """Return validated positive scalar as float64."""
    try:
        value_f64 = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a finite positive scalar; got {value!r}"
        ) from None
    if not np.isfinite(value_f64) or value_f64 <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar; got {value!r}")
    return value_f64


def _require_positive_int(name, value):
    """Return validated positive integer."""
    if not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be a positive integer; got {type(value).__name__}"
        )
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0; got {value}")
    return value


def gauss_legendre_logE_nodes_weights(E_min, E_max, n):
    r"""GL nodes and weights mapped to y = ln(E / E_min).

    Returns energy nodes E_i and integration weights w_i such that

        ∫_{E_min}^{E_max} f(E) dE  ≈  Σ_i  w_i · f(E_i)

    where the quadrature is performed in the variable y = ln(E/E_min).
    The Jacobian dE/dy = E is absorbed into the weights.

    Parameters
    ----------
    E_min, E_max : float
        Energy bounds in MeV.
    n : int
        Number of quadrature points.

    Returns
    -------
    E_nodes : ndarray, shape (n,)
        Energy nodes in MeV.
    weights : ndarray, shape (n,)
        Integration weights (include Jacobian E·dy).
    """
    E_min = _require_finite_positive_scalar("E_min", E_min)
    E_max = _require_finite_positive_scalar("E_max", E_max)
    if E_max <= E_min:
        raise ValueError(f"E_max must be > E_min; got E_min={E_min}, E_max={E_max}")
    n = _require_positive_int("n", n)

    y_max = np.log(E_max / E_min)
    nodes, w = roots_legendre(n)
    # Map [-1, 1] -> [0, y_max]
    y = 0.5 * y_max * (nodes + 1.0)
    E_nodes = E_min * np.exp(y)
    weights = 0.5 * y_max * w * E_nodes
    return E_nodes, weights


def gauss_legendre_logE_two_panel(E_min, E_max, delta, n1, n2):
    r"""Two-panel GL quadrature split near threshold.

    Panel A: [E_min, E_split]  with n1 points
    Panel B: [E_split, E_max]  with n2 points

    where E_split = E_min · exp(delta).

    If E_split >= E_max the second panel is empty and only panel A is used.

    Parameters
    ----------
    E_min, E_max : float
        Energy bounds in MeV.
    delta : float
        Log-energy width of the near-threshold panel (in natural-log units).
    n1, n2 : int
        Number of GL points for panels A and B.

    Returns
    -------
    E_nodes : ndarray
        Concatenated energy nodes.
    weights : ndarray
        Concatenated weights.
    """
    E_min = _require_finite_positive_scalar("E_min", E_min)
    E_max = _require_finite_positive_scalar("E_max", E_max)
    if E_max <= E_min:
        raise ValueError(f"E_max must be > E_min; got E_min={E_min}, E_max={E_max}")
    delta = _require_finite_positive_scalar("delta", delta)
    n1 = _require_positive_int("n1", n1)
    n2 = _require_positive_int("n2", n2)

    E_split = E_min * np.exp(delta)

    if E_split >= E_max:
        return gauss_legendre_logE_nodes_weights(E_min, E_max, n1)

    E1, w1 = gauss_legendre_logE_nodes_weights(E_min, E_split, n1)
    E2, w2 = gauss_legendre_logE_nodes_weights(E_split, E_max, n2)

    return np.concatenate([E1, E2]), np.concatenate([w1, w2])


@lru_cache(maxsize=32)
def cached_leggauss(n):
    """Cached Gauss-Legendre reference nodes/weights on [-1,1].

    The eigenvalue decomposition in ``np.polynomial.legendre.leggauss``
    is expensive.  Since the nodes/weights on [-1,1] depend only on *n*,
    caching avoids redundant recomputation across thousands of calls.
    """
    return np.polynomial.legendre.leggauss(int(n))
