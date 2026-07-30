"""Batch-oriented single-integral Hillas kernel (θ-collapsed).

Reference implementation from docs/HILLAS_SINGLE_INTEGRAL_KERNEL.md.

This module implements the θ-collapsed formulation with:
- Energy-regime-aware Hillas parameters (low/high split at E_switch)
- Asymmetric two-panel Gauss-Legendre quadrature split exactly at E_switch
  (default nL=3, nH=8: low panel converges at ~4 nodes; high panel is the limiter)
- Sub-batching for bandwidth optimization
- In-place CDF computation to minimize allocations
"""

from functools import lru_cache

import numpy as np

from .hillas_kernel import hillas_w_mean
from .hillas_params import HILLAS_HIGH_ENERGY, HILLAS_LOW_ENERGY


@lru_cache(maxsize=32)
def _cached_leggauss(n):
    """Cached Gauss-Legendre reference nodes/weights on [-1,1].

    The eigenvalue decomposition in ``np.polynomial.legendre.leggauss``
    is expensive.  Since the nodes/weights on [-1,1] depend only on *n*,
    caching avoids redundant recomputation across thousands of calls.
    """
    return np.polynomial.legendre.leggauss(int(n))


__all__ = [
    "gauss_legendre_interval",
    "energy_quadrature_split_at_Eswitch",
    "g_theta_exact",
    "hillas_regime_constants",
    "hillas_F_u_inplace",
    "precompute_kernel_NE16",
    "delta_nphots_single_integral_NE16",
]


# ---------------------------------------------------------------------------
# 1) Energy quadrature
# ---------------------------------------------------------------------------


def gauss_legendre_interval(n, a, b, dtype=np.float64):
    """Gauss-Legendre nodes/weights mapped from [-1,1] to [a,b].

    Parameters
    ----------
    n : int
        Number of quadrature points.
    a, b : float
        Interval bounds.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    y_nodes : ndarray, shape (n,)
        Quadrature nodes in [a, b].
    y_weights : ndarray, shape (n,)
        Quadrature weights.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"n must be a positive integer; got {n}")
    try:
        a = dtype(a)
        b = dtype(b)
    except (TypeError, ValueError):
        raise ValueError(f"a and b must be convertible to {dtype}") from None
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError(f"a and b must be finite; got a={a}, b={b}")
    if b <= a:
        raise ValueError(f"Require b > a; got a={a}, b={b}")

    x, w = _cached_leggauss(n)
    xm = dtype(0.5) * (a + b)
    xr = dtype(0.5) * (b - a)
    return (xm + xr * x).astype(dtype, copy=False), (xr * w).astype(dtype, copy=False)


def energy_quadrature_split_at_Eswitch(
    Emin, Eswitch, Emax, n_per_panel=8, dtype=np.float64
):
    """Two-panel Gauss-Legendre in y = ln(E/Emin), split exactly at Eswitch.

    Weights include Jacobian dE = E dy.

    Parameters
    ----------
    Emin : float
        Minimum energy (MeV).
    Eswitch : float
        Energy regime transition point (MeV).
    Emax : float
        Maximum energy (MeV).
    n_per_panel : int, optional
        Number of quadrature points per panel (default 8).
    dtype : dtype, optional
        Output dtype (default float64).

    Returns
    -------
    E_nodes : ndarray, shape (2*n_per_panel,)
        Energy nodes (MeV).
    W_E : ndarray, shape (2*n_per_panel,)
        Quadrature weights including Jacobian (dE = E dy).
    """
    if not isinstance(n_per_panel, (int, np.integer)) or n_per_panel <= 0:
        raise ValueError(f"n_per_panel must be a positive integer; got {n_per_panel}")
    try:
        Emin = dtype(Emin)
        Esw = dtype(Eswitch)
        Emax = dtype(Emax)
    except (TypeError, ValueError):
        raise ValueError(
            f"Emin, Eswitch, Emax must be convertible to {dtype}"
        ) from None
    if not (np.isfinite(Emin) and np.isfinite(Esw) and np.isfinite(Emax)):
        raise ValueError("Emin, Eswitch, Emax must be finite")
    if not (Emax > Emin):
        raise ValueError(f"Require Emax > Emin; got Emin={Emin}, Emax={Emax}")

    # Clamp Eswitch into bounds
    Esw = min(max(Esw, Emin), Emax)

    y_max = np.log(Emax / Emin, dtype=dtype)
    y_sw = np.log(Esw / Emin, dtype=dtype)

    # Panel L: [0, y_sw], Panel H: [y_sw, y_max]
    yL, wL = gauss_legendre_interval(n_per_panel, dtype(0.0), y_sw, dtype=dtype)
    yH, wH = gauss_legendre_interval(n_per_panel, y_sw, y_max, dtype=dtype)

    y = np.concatenate([yL, yH])
    wy = np.concatenate([wL, wH])

    E = Emin * np.exp(y, dtype=dtype)
    W = wy * E  # Jacobian: dE = E dy

    if E.size != 2 * n_per_panel:
        raise RuntimeError(
            f"Unexpected node count; expected {2 * n_per_panel}, got {E.size}"
        )
    return E, W


def _energy_quadrature_batched(Emin, Emax, Eswitch, nL=3, nH=8, dtype=np.float64):
    """Per-row asymmetric two-panel Gauss-Legendre in y = ln(E/Emin).

    Vectorized form of :func:`energy_quadrature_split_at_Eswitch`: ``Emin`` and
    ``Emax`` are ``(P,)`` arrays giving each row its own integration window.
    Returns ``(P, nL+nH)`` nodes and Jacobian-weighted weights.

    ``Eswitch`` is clamped per row into ``[Emin, Emax]``. When the window lies
    entirely on one side of ``Eswitch`` the off-side panel collapses to zero
    width and therefore zero weight, so the first ``nL`` columns are always the
    low-regime panel and the last ``nH`` columns the high-regime panel (the
    kernel's column-mask accumulation relies on this structural split).

    The default ``nL=3, nH=8`` reflects the empirical finding that the low panel
    converges at ~4 nodes while the high panel (spanning ~16 log-decades up to
    the primary energy) is the sole accuracy limiter.
    """
    Emin = np.asarray(Emin, dtype=dtype)
    Emax = np.asarray(Emax, dtype=dtype)
    if Emin.shape != Emax.shape:
        Emin, Emax = np.broadcast_arrays(Emin, Emax)
    if not (np.all(np.isfinite(Emin)) and np.all(np.isfinite(Emax))):
        raise ValueError("Emin, Emax must be finite")
    if not np.all(Emax > Emin):
        raise ValueError("Require Emax > Emin for every row")

    xL, wL = _cached_leggauss(nL)
    xH, wH = _cached_leggauss(nH)
    xL = xL.astype(dtype, copy=False)
    wL = wL.astype(dtype, copy=False)
    xH = xH.astype(dtype, copy=False)
    wH = wH.astype(dtype, copy=False)

    Esw = np.clip(dtype(Eswitch), Emin, Emax)  # (P,)
    y_max = np.log(Emax / Emin)  # (P,)
    y_sw = np.log(Esw / Emin)  # (P,)

    # Panel L: [0, y_sw]; Panel H: [y_sw, y_max]. Map reference x in [-1,1].
    yL_g = (0.5 * y_sw)[:, None] * (dtype(1.0) + xL[None, :])  # (P, nL)
    wyL = (0.5 * y_sw)[:, None] * wL[None, :]
    midH = 0.5 * (y_sw + y_max)
    halfH = 0.5 * (y_max - y_sw)
    yH_g = midH[:, None] + halfH[:, None] * xH[None, :]  # (P, nH)
    wyH = halfH[:, None] * wH[None, :]

    y = np.concatenate([yL_g, yH_g], axis=1)  # (P, nL+nH)
    wy = np.concatenate([wyL, wyH], axis=1)

    E = Emin[:, None] * np.exp(y)  # (P, nL+nH)
    W = wy * E  # Jacobian: dE = E dy
    return E.astype(dtype, copy=False), W.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# 2) Original angular factor (no Padé)
# ---------------------------------------------------------------------------


def g_theta_exact(theta, dtype=np.float64):
    """Original angular factor g(θ) = 2(1 - cos θ).

    Parameters
    ----------
    theta : array_like
        Angle in radians.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    g : ndarray
        Angular factor.
    """
    theta = np.asarray(theta, dtype=dtype)
    return dtype(2.0) * (dtype(1.0) - np.cos(theta, dtype=dtype))


# ---------------------------------------------------------------------------
# 3) Hillas CDF precomputation per regime
# ---------------------------------------------------------------------------


def hillas_regime_constants(A, z0, lam1, lam2, dtype=np.float64):
    """Precompute constants for fast F(u) evaluation.

    Define:
      I(z;lam) = 2*A*lam * [(z0+lam) - (z+lam) * exp(-(z - z0)/lam)]
    Then:
      F(u) = I(z;lam1) - I(0;lam1)                                if z<=z0
           = [I(z0;lam1)-I(0;lam1)] + [I(z;lam2)-I(z0;lam2)]      if z>z0
    where z = sqrt(u).

    Parameters
    ----------
    A, z0, lam1, lam2 : float
        Hillas parameters.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    dict
        Precomputed constants for this regime.
    """
    try:
        A = dtype(A)
        z0 = dtype(z0)
        lam1 = dtype(lam1)
        lam2 = dtype(lam2)
    except (TypeError, ValueError):
        raise ValueError("A, z0, lam1, lam2 must be convertible to dtype") from None
    if not np.all(np.isfinite([A, z0, lam1, lam2])):
        raise ValueError("Hillas parameters must be finite")
    if A <= 0.0 or lam1 <= 0.0 or lam2 <= 0.0:
        raise ValueError("A, lam1, lam2 must be strictly positive")
    if z0 < 0.0:
        raise ValueError("z0 must be non-negative")

    K1 = dtype(2.0) * A * lam1
    K2 = dtype(2.0) * A * lam2
    C1 = z0 + lam1
    C2 = z0 + lam2
    inv1 = dtype(1.0) / lam1
    inv2 = dtype(1.0) / lam2

    # I(0;lam1)
    I0 = K1 * (
        C1 - (dtype(0.0) + lam1) * np.exp(-(dtype(0.0) - z0) * inv1, dtype=dtype)
    )

    # I(z0;lam1)=0 analytically, so F(z0) = -I0
    Fz0 = -I0

    # Reduced constants for hillas_F_u_inplace's per-element evaluation:
    #   F = K*(C - (z+lam)*ex) + offset = D - (2A*lam)*(z+lam)*ex
    # since K = 2A*lam and (K*C + offset) is a per-sub-regime constant D. This
    # lets the kernel blend only {lam, inv, D} (3 arrays) instead of
    # {lam, K, C, inv, offset} (5), and drop a term from the final expression.
    twoA = dtype(2.0) * A
    D1 = K1 * C1 - I0  # z <= z0 sub-regime (offset = -I0)
    D2 = K2 * C2 + Fz0  # z >  z0 sub-regime (offset = Fz0)

    return dict(
        A=A,
        z0=z0,
        lam1=lam1,
        lam2=lam2,
        K1=K1,
        K2=K2,
        C1=C1,
        C2=C2,
        inv1=inv1,
        inv2=inv2,
        I0=I0,
        Fz0=Fz0,
        twoA=twoA,
        D1=D1,
        D2=D2,
    )


def hillas_F_u_inplace(u, const, out, dtype=np.float64):
    """Compute F(u) for one regime into 'out'.

    u and out must have the same shape. Uses out as scratch (stores z=sqrt(u) temporarily).
    This is vectorized and uses only an O(shape(u)) temporary boolean mask.

    The piecewise CDF reduces to ``F = D - (2A*lam)*(z+lam)*exp(-(z-z0)/lam)``
    with ``z = sqrt(u)`` and the per-sub-regime (z<=z0 vs z>z0) constants
    ``lam, inv=1/lam, D`` selected by mask. ``K = 2A*lam`` and ``D = K*C + offset``
    are folded into the precomputed constants, so only three per-element blends
    remain (was five) -- ~12% faster F-eval, the kernel's dominant cost.

    Parameters
    ----------
    u : ndarray
        Input u values.
    const : dict
        Constants from hillas_regime_constants.
    out : ndarray
        Output array (same shape as u). Will be overwritten.
    dtype : dtype, optional
        Output dtype.
    """
    if u.shape != out.shape:
        raise ValueError(
            f"u and out must have same shape; got {u.shape} vs {out.shape}"
        )

    # clamp u>=0 into out
    np.maximum(u, dtype(0.0), out=out)

    # out := z = sqrt(u)
    np.sqrt(out, out=out)

    z = out
    z0 = const["z0"]
    mf = (z <= z0).astype(dtype)  # 1 where z<=z0, 0 otherwise
    nf = dtype(1.0) - mf  # 0 where z<=z0, 1 otherwise

    # Per-element sub-regime selection via mask arithmetic (avoids np.where).
    # Only lam, inv, and the merged constant D need blending now.
    lam = mf * const["lam1"] + nf * const["lam2"]
    inv = mf * const["inv1"] + nf * const["inv2"]
    D = mf * const["D1"] + nf * const["D2"]

    # Single exp, fused expression: F = D - (2A*lam)*(z+lam)*exp(-(z-z0)/lam).
    ex = np.exp(-(z - z0) * inv)
    out[:] = D - (const["twoA"] * lam) * (z + lam) * ex


# ---------------------------------------------------------------------------
# 4) Precompute kernel constants (call once)
# ---------------------------------------------------------------------------


def precompute_kernel_NE16(eCthres, Eshow, Eswitch, nL=3, nH=8, dtype=np.float64):
    """Precompute everything independent of per-layer shower state.

    Parameters
    ----------
    eCthres : float or ndarray
        Minimum energy (MeV). Scalar or ``(P,)`` array for per-row windows.
    Eshow : float or ndarray
        Maximum energy (MeV). Scalar or ``(P,)`` array.
    Eswitch : float
        Energy regime transition (MeV).
    nL : int, optional
        GL nodes on the low-energy panel ``[eCthres, Eswitch]`` (default 3).
        The low panel converges at ~4 nodes; fewer than 3 is not recommended.
    nH : int, optional
        GL nodes on the high-energy panel ``[Eswitch, Eshow]`` (default 8).
        The high panel spans ~16 log-decades and is the sole accuracy limiter.
    dtype : dtype, optional
        Output dtype.

    ``eCthres`` and ``Eshow`` may be scalars (one shared ``nL+nH``-node grid) or
    ``(P,)`` arrays (a per-row window for each of the P kernel rows).

    Returns
    -------
    dict
        Precomputed kernel constants:
        - ``E_nodes``, ``W_E``: shape ``(nL+nH,)`` or ``(P, nL+nH)``
        - ``hi_mask``: bool ``(nL+nH,)`` — True for the ``nH`` high-panel columns
        - ``nL``, ``nH``: stored for the kernel's buffer sizing
        - ``low``, ``high``: Hillas regime constant dicts

    The angular factor ``(E/21)^2`` is not materialized here; the kernel
    recomputes it cheaply per sub-batch slab.
    """
    emin = np.asarray(eCthres, dtype=dtype)
    emax = np.asarray(Eshow, dtype=dtype)
    if emin.ndim == 0 and emax.ndim == 0:
        # Scalar path: build the two panels directly with asymmetric counts.
        try:
            Emin_s, Esw_s, Emax_s = dtype(eCthres), dtype(Eswitch), dtype(Eshow)
        except (TypeError, ValueError):
            raise ValueError("eCthres, Eswitch, Eshow must be finite scalars") from None
        Esw_s = min(max(Esw_s, Emin_s), Emax_s)
        y_max = np.log(Emax_s / Emin_s, dtype=dtype)
        y_sw = np.log(Esw_s / Emin_s, dtype=dtype)
        yL, wL = gauss_legendre_interval(nL, dtype(0.0), y_sw, dtype=dtype)
        yH, wH = gauss_legendre_interval(nH, y_sw, y_max, dtype=dtype)
        y = np.concatenate([yL, yH])
        wy = np.concatenate([wL, wH])
        E_nodes = Emin_s * np.exp(y, dtype=dtype)
        W_E = wy * E_nodes
        hi_mask = np.zeros(nL + nH, dtype=bool)
        hi_mask[nL:] = True
    else:
        E_nodes, W_E = _energy_quadrature_batched(
            emin, emax, Eswitch, nL=nL, nH=nH, dtype=dtype
        )
        # Structural split: columns 0..nL-1 are panel L, nL..nL+nH-1 are panel H.
        # A collapsed panel carries zero weight, so this holds for every row.
        hi_mask = np.zeros(nL + nH, dtype=bool)
        hi_mask[nL:] = True

    low = hillas_regime_constants(
        A=HILLAS_LOW_ENERGY.A,
        z0=HILLAS_LOW_ENERGY.z_star,
        lam1=HILLAS_LOW_ENERGY.lam1,
        lam2=HILLAS_LOW_ENERGY.lam2,
        dtype=dtype,
    )
    high = hillas_regime_constants(
        A=HILLAS_HIGH_ENERGY.A,
        z0=HILLAS_HIGH_ENERGY.z_star,
        lam1=HILLAS_HIGH_ENERGY.lam1,
        lam2=HILLAS_HIGH_ENERGY.lam2,
        dtype=dtype,
    )

    return dict(
        dtype=dtype,
        E_nodes=E_nodes,
        W_E=W_E,
        hi_mask=hi_mask,
        nL=nL,
        nH=nH,
        low=low,
        high=high,
    )


# ---------------------------------------------------------------------------
# 5) Single-integral per-layer kernel with sub-batching
# ---------------------------------------------------------------------------


def delta_nphots_single_integral_NE16(
    thetaC,
    sigsum,
    mean_w,
    W_extra,
    pre,
    p_sub=8192,
):
    """Compute per-shower photon contribution using θ-collapsed single-integral formulation.

    out[i] = ∫_{Emin}^{Emax} F(u_max(E,s_i)) * W(E,s_i) dE

    where:
      u_max(E,s_i) = g(thetaC_i) * (E/21)^2 / <w>(E,s_i)
      W(E,s_i)     = sigsum_i * W_extra(E,s_i) * W_E(E)

    Implementation:
      - Processes showers in slabs of size p_sub to limit temporaries to (p_sub,16)
      - Evaluates both Hillas regimes and accumulates by energy-column masks (8 low + 8 high)
      - Reuses buffers to reduce allocation churn

    Parameters
    ----------
    thetaC : ndarray, shape (p,)
        Per-shower Cherenkov angle (rad).
    sigsum : ndarray, shape (p,)
        Per-shower scalar prefactor (e.g., SPYield scaling consolidated).
    mean_w : ndarray, shape (p,16) or (16,)
        Mean angular spread <w>(E,s). Broadcastable.
    W_extra : ndarray, shape (p,16) or (16,)
        Additional weights excluding quadrature W_E. Broadcastable.
    pre : dict
        Precomputed kernel constants from precompute_kernel_NE16.
    p_sub : int, optional
        Sub-batch size (tune to bandwidth knee, default 8192).

    Returns
    -------
    out : ndarray, shape (p,)
        Per-shower photon contribution.
    """
    if not isinstance(p_sub, (int, np.integer)) or p_sub <= 0:
        raise ValueError(f"p_sub must be a positive integer; got {p_sub}")

    dtype = pre["dtype"]
    thetaC = np.asarray(thetaC, dtype=dtype)
    sigsum = np.asarray(sigsum, dtype=dtype)

    if thetaC.ndim != 1:
        raise ValueError(f"thetaC must be 1D; got ndim={thetaC.ndim}")
    if sigsum.ndim != 1:
        raise ValueError(f"sigsum must be 1D; got ndim={sigsum.ndim}")
    if thetaC.shape[0] != sigsum.shape[0]:
        raise ValueError(
            f"thetaC and sigsum must have same length; got {thetaC.shape[0]} vs {sigsum.shape[0]}"
        )
    if not np.all(np.isfinite(thetaC)):
        raise ValueError("thetaC contains NaN/Inf")
    if not np.all(np.isfinite(sigsum)):
        raise ValueError("sigsum contains NaN/Inf")

    E_nodes_all = pre["E_nodes"]  # (n_E,) or (p, n_E)
    W_E_all = pre["W_E"]  # (n_E,) or (p, n_E)
    hi = pre["hi_mask"]  # (n_E,)
    n_E = hi.shape[0]  # nL + nH
    per_row_E = E_nodes_all.ndim == 2

    p = thetaC.shape[0]
    out = np.zeros(p, dtype=dtype)

    if per_row_E and E_nodes_all.shape[0] != p:
        raise ValueError(
            f"per-row E_nodes/W_E first dim {E_nodes_all.shape[0]} != n rows {p}"
        )

    if p == 0:
        return out

    # The regime columns are contiguous (panel L = low regime in columns
    # 0..nlo-1, panel H = high regime after), by construction of the two-panel
    # split at Eswitch. So each regime's F(u) is evaluated only on its own
    # column block and the buffer holding u is reused in place to hold F(u).
    nlo = int(np.count_nonzero(~hi))

    # Reusable buffers sized to (slab, nL+nH)
    slab = min(p_sub, p)
    u_buf = np.empty((slab, n_E), dtype=dtype)
    scale = np.empty_like(u_buf)

    for i0 in range(0, p, p_sub):
        i1 = min(i0 + p_sub, p)
        n = i1 - i0

        th = thetaC[i0:i1]  # (n,)
        sig = sigsum[i0:i1]  # (n,)
        gC = g_theta_exact(th, dtype=dtype)  # (n,)

        # Resolve slab views
        u_view = u_buf[:n, :]
        scale_view = scale[:n, :]

        # Resolve mean_w and W_extra for this slab with minimal copying
        mw = mean_w[i0:i1, :] if np.ndim(mean_w) == 2 else mean_w  # (n,16) or (16,)
        wx = W_extra[i0:i1, :] if np.ndim(W_extra) == 2 else W_extra

        # Resolve the energy grid for this slab (per-row windows or shared grid)
        E_nodes = E_nodes_all[i0:i1, :] if per_row_E else E_nodes_all[None, :]
        W_E = W_E_all[i0:i1, :] if per_row_E else W_E_all  # (n,16) or (16,)

        # scale[n,16] = sig[:,None] * wx * W_E
        np.multiply(wx, W_E, out=scale_view)
        scale_view *= sig[:, None]

        # u[n,16] = gC[:,None] * (E/21)^2 / mw. (E/21)^2 is recomputed per slab
        # (cheap square) rather than carried as a full (p,16) array in `pre`.
        np.divide(E_nodes, dtype(21.0), out=u_view)
        u_view *= u_view  # (E/21)^2
        u_view *= gC[:, None]
        u_view /= mw

        # Transform u -> F(u) in place, each regime on its own column block.
        hillas_F_u_inplace(u_view[:, :nlo], pre["low"], u_view[:, :nlo], dtype=dtype)
        hillas_F_u_inplace(u_view[:, nlo:], pre["high"], u_view[:, nlo:], dtype=dtype)

        # F(u) now lives in u_view (low|high already in place) -> one reduction.
        out[i0:i1] += np.einsum("nk,nk->n", u_view, scale_view)

    return out


def secondary_track_fraction(E0, eCthres, s):
    r"""Integral track-length spectrum T(E) of Hillas (1982), eqn (8).

    The fraction of charged-particle track length carried by particles with
    kinetic energy above ``E`` (= ``eCthres``), at shower age ``s``::

                  / 0.89*E0 - 1.2 \ s
        T(E)  =  ( --------------  )   * (1 + 1e-4 * s * E) ** -2
                  \    E0 + E     /

    Defined (Hillas p. 1466) as the total track length of charged particles of
    kinetic energy > E divided by the total vertical track-length component, so
    ``N_e * T(E) * dx`` is the track length above E in a vertical slab ``dx``.
    This is the Cherenkov-yield weight: electrons above the Cherenkov threshold,
    weighted by the track they lay down. ``E0 = hillas_scale_energy(s)`` is the age-dependent
    scale energy (MeV). All energies are kinetic, in MeV.

    Reference: A. M. Hillas, "Angular and energy distributions of charged
    particles in electron-photon cascades in air," J. Phys. G: Nucl. Phys. 8
    (1982) 1461-1473, eqn (8).
    """
    v1 = ((0.89 * E0 - 1.2) / (E0 + eCthres)) ** s
    # (1 + 1e-4*s*E)**-2 is just 1/c^2 -- a multiply + reciprocal, ~15x faster
    # than the general `** -2` pow on the (n,k,n_E) grid (and dtype-preserving).
    c = 1.0 + 1e-4 * s * eCthres
    c *= c
    return v1 / c


def hillas_scale_energy(shape, s):
    """Age-dependent scale energy E0(s) (MeV) of the track-length spectrum.

    The ``E0`` appearing in :func:`tracklen` (Hillas 1982, eqn (8), p. 1466)::

        E0(s) = 44 - 17*(s - 1.46)**2   if s >= 0.4
              = 26                        if s <  0.4
    """
    E0 = np.full(shape, 26.0, dtype=np.float64)
    E0[s >= 0.4] = 44.0 - 17.0 * (s[(s >= 0.4)] - 1.46) ** 2
    return E0


def make_hillas_photon_yield(n_energy_low=3, n_energy_high=8, dtype=np.float32):
    """Default photon-yield model: Hillas (1982) theta-collapsed single integral.

    Returns a callable ``model(inputs: PhotonYieldInputs) -> node_contribs`` of
    shape ``(n_showers, n_nodes)`` -- the high-performance vectorized CPU kernel
    used by :meth:`CphotAng.run` when no custom model is supplied. It evaluates,
    per node, the energy integral ``int F(u_max(E,s)) * w_x(E,s) dE`` over a
    two-panel Gauss-Legendre grid in ``y = ln(E/eCthres)`` split at the
    low/high-energy regime boundary (see docs/HILLAS_SINGLE_INTEGRAL_KERNEL.md).

    Parameters
    ----------
    n_energy_low : int, optional
        GL nodes on the low-energy panel ``[eCthres, Eswitch=1 GeV]`` (default 3).
    n_energy_high : int, optional
        GL nodes on the high-energy panel ``[Eswitch, Eshow]`` (default 8). This
        panel is the sole accuracy limiter of the energy integral.
    dtype : numpy dtype, optional
        Working precision for the energy-node region (default float32). The
        energy quadrature is only ~0.3% accurate, so single precision (~1e-7)
        is well within budget while roughly doubling transcendental throughput
        and halving the ``(n, k, n_E)`` bandwidth -- the run() working-set peak.

    Notes
    -----
    The returned closure is a plain function (no per-call object churn beyond a
    single dataclass read), so the default path is bit-identical and equal in
    cost to the former inlined kernel.
    """

    def hillas_photon_yield(inputs):
        eCthres = inputs.eCthres
        Eshow = inputs.Eshow
        thetaC = inputs.thetaC
        sig_per_node = inputs.sig_per_node
        e2hill = inputs.e2hill
        E0 = inputs.E0
        s = inputs.s
        n_nodes = inputs.n_nodes

        n_showers = sig_per_node.shape[0]
        # Radiating nodes carry sig > 0 (every in-window node, minus any zeroed
        # by clouds or by a degenerate zero-weight grid).
        radiating = sig_per_node > 0.0
        if not np.any(radiating):
            return np.zeros((n_showers, n_nodes))

        # Per-(shower, node) energy-quadrature window. The integral over electron
        # energy runs from the node's Cherenkov threshold eCthres(shower, node)
        # up to a per-shower envelope above the primary energy. Both bounds are
        # LOCAL (no batch-wide min/max reduction), so photonDen is a pure
        # per-shower function -- independent of batch composition. Non-radiating
        # rows get a dummy finite window; their sig == 0 zeros the contribution.
        Ieang = np.floor(np.log10(Eshow)).astype(np.int64) + 1  # (n_showers,)
        E_max_node = np.broadcast_to((10.0 ** (Ieang + 1))[:, None], eCthres.shape)
        Emin = np.where(radiating, eCthres, 1.0)  # (n_showers, n_nodes)
        Emax = np.where(radiating, E_max_node, 10.0)
        P = n_showers * n_nodes

        pre = precompute_kernel_NE16(
            Emin.reshape(P),
            Emax.reshape(P),
            Eswitch=1e3,
            nL=n_energy_low,
            nH=n_energy_high,
            dtype=dtype,
        )
        E_grid = pre["E_nodes"].reshape(n_showers, n_nodes, -1)  # f32 (n, k, n_E)
        n_E = E_grid.shape[-1]

        # mw_all uses raw e2hill; hillas_w_mean is total, and any non-radiating
        # node (sig_all == 0) contributes exactly 0 (finite F(u) * 0).
        sig_all = sig_per_node.astype(dtype)  # (n_showers, n_nodes)
        thetaC = thetaC.astype(dtype)
        mw_all = hillas_w_mean(E_grid, e2hill[:, :, None].astype(dtype))

        # wx_all = max(-dT/dE, 0) for T = secondary_track_fraction(E0, E, s). Closed-form
        # derivative avoids a second (n_showers, n_nodes, n_E) tracklen call:
        #   -dT/dE = s * T * (1/(E0 + E) + 2e-4 / (1 + 1e-4*s*E))
        # Built in place to avoid spinning up transient (n, k, n_E) arrays for
        # each sub-expression (this block is the run() working-set peak).
        E0_b = E0[:, :, None].astype(dtype)
        s_b = s[:, :, None].astype(dtype)
        T = secondary_track_fraction(E0_b, E_grid, s_b)  # (n_showers, n_nodes, n_E)
        c = 1.0 + 1e-4 * s_b * E_grid  # 1 + 1e-4*s*E
        np.reciprocal(c, out=c)
        c *= 2e-4  # c := 2e-4 / (1 + 1e-4*s*E)
        wx_all = E0_b + E_grid
        np.reciprocal(wx_all, out=wx_all)  # wx_all := 1/(E0 + E)
        wx_all += c  # 1/(E0+E) + 2e-4/(1+1e-4*s*E)
        wx_all *= T
        wx_all *= s_b
        np.maximum(wx_all, 0.0, out=wx_all)
        # T and c are dead now; free them so they don't sit resident
        # (2x (n,k,n_E)) through the kernel call, the working-set peak.
        del T, c

        # Fold (n_showers, n_nodes) into a single batch axis -> one kernel call.
        # Reshapes are views (C-contiguous); non-radiating nodes carry
        # sig_all == 0, contributing exactly 0 (finite F(u) * 0). The per-row
        # energy grid in `pre` is flattened in the same C-order, so rows align.
        contrib_flat = delta_nphots_single_integral_NE16(
            thetaC.reshape(P),
            sig_all.reshape(P),
            mw_all.reshape(P, n_E),
            wx_all.reshape(P, n_E),
            pre,
        )
        return contrib_flat.reshape(n_showers, n_nodes)

    return hillas_photon_yield
