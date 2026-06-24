"""Pure-math Hillas angular-distribution kernel functions.

No simulation dependencies. All functions operate in float64 by default.

Implements V1 (analytic primitive), V2 (CDF-diff), and V5 (GL quadrature)
variants of the Hillas bin integral per AGENTS.md specification.
"""

import numpy as np
from scipy.special import roots_legendre

from .hillas_params import HILLAS_DEFAULT, HillasParams

__all__ = [
    "pade11",
    "hillas_dndu",
    "hillas_delta_n_analytic",
    "hillas_cdf",
    "hillas_delta_n_cdf_diff",
    "hillas_delta_n_numerical",
    "hillas_w",
    "hillas_w_mean",
    "hillas_u",
]


# ---------------------------------------------------------------------------
# Angular mapping
# ---------------------------------------------------------------------------


def pade11(theta):
    r"""Padé [1,1] approximation to 2(1 - cos θ).

    g₁₁(θ) = θ² / (1 + θ²/12)

    Relative error < 1e-6 for θ < 0.1 rad, < 0.3 % up to θ ≈ 0.5 rad.
    """
    theta = np.asarray(theta, dtype=np.float64)
    t2 = theta * theta
    return t2 / (1.0 + t2 / 12.0)


# ---------------------------------------------------------------------------
# Point evaluation
# ---------------------------------------------------------------------------


def hillas_dndu(u, p=HILLAS_DEFAULT):
    """Hillas dn/du at a single u value (or array).

    dn/du = A · exp(-(√u - z★) / λ_i)
    """
    u = np.asarray(u, dtype=np.float64)
    z = np.sqrt(u)
    lam = p.lam(z)
    return p.A * np.exp(-(z - p.z_star) / lam)


# ---------------------------------------------------------------------------
# V1 — Analytic primitive (closed-form bin integral)
# ---------------------------------------------------------------------------


def _primitive_segment(za, zb, lam, z_star, A):
    """Analytic integral of A·exp(-(z - z★)/λ)·2z dz from za to zb.

    Result = 2Aλ [(za + λ)·exp(-(za - z★)/λ) - (zb + λ)·exp(-(zb - z★)/λ)]
    """
    term_a = (za + lam) * np.exp(-(za - z_star) / lam)
    term_b = (zb + lam) * np.exp(-(zb - z_star) / lam)
    return 2.0 * A * lam * (term_a - term_b)


def hillas_delta_n_analytic(u0, u1, p=HILLAS_DEFAULT):
    r"""V1: Closed-form Δn(u0, u1) with split at z★.

    Integrates dn/du from u0 to u1 using the analytic primitive.
    If the interval straddles z★, splits at u★ = z★² and sums both pieces.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    u1 = np.asarray(u1, dtype=np.float64)

    z0 = np.sqrt(u0)
    z1 = np.sqrt(u1)

    u_star = p.z_star**2

    straddles = (u0 < u_star) & (u1 > u_star)
    below = u1 <= u_star
    above = u0 >= u_star

    result = np.zeros_like(u0, dtype=np.float64)

    # Entirely below z_star
    mask_below = below & ~straddles
    if np.any(mask_below):
        result[mask_below] = _primitive_segment(
            z0[mask_below], z1[mask_below], p.lam1, p.z_star, p.A
        )

    # Entirely above z_star
    mask_above = above & ~straddles
    if np.any(mask_above):
        result[mask_above] = _primitive_segment(
            z0[mask_above], z1[mask_above], p.lam2, p.z_star, p.A
        )

    # Straddling z_star: split
    if np.any(straddles):
        part1 = _primitive_segment(z0[straddles], p.z_star, p.lam1, p.z_star, p.A)
        part2 = _primitive_segment(p.z_star, z1[straddles], p.lam2, p.z_star, p.A)
        result[straddles] = part1 + part2

    return result


# ---------------------------------------------------------------------------
# V2 — CDF-diff form
# ---------------------------------------------------------------------------


def hillas_cdf(u, p=HILLAS_DEFAULT):
    """Hillas CDF: F(u) = Δn(0, u).

    The cumulative distribution from u = 0 to u.
    """
    u = np.asarray(u, dtype=np.float64)
    u = np.maximum(u, 0.0)
    u0 = np.zeros_like(u)
    return hillas_delta_n_analytic(u0, u, p)


def hillas_delta_n_cdf_diff(u0, u1, p=HILLAS_DEFAULT):
    """V2: Δn via CDF difference, F(u1) - F(u0)."""
    return hillas_cdf(u1, p) - hillas_cdf(u0, p)


# ---------------------------------------------------------------------------
# V5 — Numerical GL quadrature (gold reference)
# ---------------------------------------------------------------------------


def hillas_delta_n_numerical(u0, u1, p=HILLAS_DEFAULT, n_quad=64):
    r"""V5: Gold-reference GL quadrature of dn/du in z-space.

    Integrates A·exp(-(z - z★)/λ)·2z dz over [z0, z1] with z = √u.
    Splits at z★ if the interval straddles it.
    Uses Gauss-Legendre quadrature with n_quad points per sub-interval.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    u1 = np.asarray(u1, dtype=np.float64)

    scalar_input = u0.ndim == 0 and u1.ndim == 0
    u0 = np.atleast_1d(u0)
    u1 = np.atleast_1d(u1)

    z0 = np.sqrt(u0)
    z1 = np.sqrt(u1)
    u_star = p.z_star**2

    nodes, weights = roots_legendre(n_quad)

    result = np.zeros_like(u0, dtype=np.float64)

    for i in range(len(u0)):
        za, zb = z0[i], z1[i]
        if za >= zb:
            continue

        # Determine sub-intervals
        if u0[i] < u_star and u1[i] > u_star:
            intervals = [(za, p.z_star, p.lam1), (p.z_star, zb, p.lam2)]
        elif u1[i] <= u_star:
            intervals = [(za, zb, p.lam1)]
        else:
            intervals = [(za, zb, p.lam2)]

        total = 0.0
        for a, b, lam in intervals:
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a)
            z_pts = mid + half * nodes
            integrand = p.A * np.exp(-(z_pts - p.z_star) / lam) * 2.0 * z_pts
            total += half * np.dot(weights, integrand)

        result[i] = total

    if scalar_input:
        return result[0]
    return result


# ---------------------------------------------------------------------------
# w, <w>, and u helpers
# ---------------------------------------------------------------------------


def hillas_w(theta, E):
    r"""w(θ, E) = g₁₁(θ) · (E / 21)².

    Parameters
    ----------
    theta : array_like
        Angle in radians.
    E : array_like
        Energy in MeV.
    """
    theta = np.asarray(theta, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    return pade11(theta) * (E / 21.0) ** 2


def hillas_w_mean(E, e2hill):
    r"""Mean angular spread ⟨w⟩ = 0.0054·E·(1+v) / (1 + 13v + 8.3v²).

    Parameters
    ----------
    E : array_like
        Energy in MeV.
    e2hill : array_like
        Hillas e2 parameter (1150 + 454·ln(s)).
    """
    # Preserve the input dtype (float64 callers unchanged; the kernel path drives
    # float32 here for throughput). Python-scalar constants are weak under NEP 50,
    # so a float32 input stays float32 throughout.
    E = np.asarray(E)
    e2hill = np.asarray(e2hill)
    # e2hill <= 0 is outside the parameterization (shower age below its floor);
    # keep ⟨w⟩ finite there so dead/early nodes can't inject NaN via E/0. Such
    # nodes carry zero photon yield, so the substituted value never reaches output.
    e2hill = np.where(e2hill > 0.0, e2hill, 1.0)
    v = E / e2hill
    return 0.0054 * E * (1.0 + v) / (1.0 + 13.0 * v + 8.3 * v**2)


def hillas_u(theta, E, e2hill):
    r"""Hillas u-variable: u = w(θ, E) / ⟨w⟩(E, e2hill).

    Parameters
    ----------
    theta : array_like
        Angle in radians.
    E : array_like
        Energy in MeV.
    e2hill : array_like
        Hillas e2 parameter.
    """
    w = hillas_w(theta, E)
    w_avg = hillas_w_mean(E, e2hill)
    return w / w_avg
