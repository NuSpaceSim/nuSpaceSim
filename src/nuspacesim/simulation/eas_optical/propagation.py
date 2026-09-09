"""Shower propagation length oracle using Halley's method and Gauss-Legendre quadrature.

This module provides fast computation of shower propagation lengths through
Earth's atmosphere using a combination of root-finding (Halley's method) and
numerical integration (Gauss-Legendre quadrature).

Reference implementation: docs/Compact Shower Length.ipynb
"""

import numpy as np

from .quadrature import cached_leggauss

__all__ = [
    "Atmosphere",
    "us_std_atm_density",
    "lexpr",
    "zexpr",
    "viewing_angle",
    "dXl",
    "dXl_known_layer",
    "d2Xl_known_layer",
    "total_slant_depth",
    "slant_depth",
    "length_at_depth",
    "length_at_depth_hermite",
    "length_at_depth_approx",
    "slant_depth_intervals",
    "shower_propagation_length",
]


class Atmosphere:
    """US 1976 Standard Atmosphere lookup tables and constants.

    Provides atmospheric model parameters for density calculations.
    """

    def __init__(self):
        # Geopotential altitude breakpoints (km)
        self.H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852])

        # Temperature lapse rates (K/km)
        self.Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])

        # Base temperatures (K)
        self.T_b = np.array(
            [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
        )

        # Base pressures (Pa)
        self.P_b = np.array(
            [
                1.01325000e05,
                2.26320635e04,
                5.47488866e03,
                8.68018689e02,
                1.10906302e02,
                6.69388728e01,
                3.95642046e00,
                3.73383638e-01,
            ]
        )

        # Gas constant and molecular mass
        self.Rstar = 8.31432e-3  # kJ/(kmol·K)
        self.M0 = 28.9644  # kg/kmol
        self.gmr = 34.163195  # Combined constant

        # Earth radius (km)
        self.R = 6371.0
        self.earth_radius = self.R

        # Geometric altitude breakpoints (km)
        self.Z_b = self.H_b * self.R / (self.R - self.H_b)

    def __call__(self, h):
        """Fast binary search for atmospheric layer index.

        Parameters
        ----------
        h : float or ndarray
            Geopotential altitude (km).

        Returns
        -------
        i : int or ndarray
            Layer index for lookup tables.
        """
        i = (h > self.H_b[4]) << 2
        i |= (h > self.H_b[i + 2]) << 1
        i |= h > self.H_b[i + 1]
        return i


def us_std_atm_density(z, R=6371.0, a=None):
    """Compute atmospheric density using US 1976 Standard Atmosphere model.

    Parameters
    ----------
    z : float or ndarray
        Geometric altitude (km).
    R : float, optional
        Earth radius (km), default 6371.0.
    a : Atmosphere, optional
        Atmosphere instance. If None, creates a new one.

    Returns
    -------
    rho : float or ndarray
        Atmospheric density (g/cm³).

    Raises
    ------
    ValueError
        If z contains NaN/Inf values or R is non-positive.
    """
    if a is None:
        a = Atmosphere()

    z = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains NaN/Inf")
    if R <= 0.0:
        raise ValueError(f"R must be positive; got {R}")

    # Convert geometric to geopotential altitude
    h = z * R / (z + R)
    i = a(h)

    Hb = a.H_b[i]
    Tb = a.T_b[i]
    Lmb = a.Lm_b[i]
    Pb = a.P_b[i]
    deltah = h - Hb
    temperature = Tb + Lmb * deltah

    # Pressure: isothermal (Lm==0) vs gradient layers. Compute both branches
    # and select with np.where -- avoids boolean-mask scatter/gather on large
    # arrays (~1.8x faster, bit-identical to the masked form).
    iso = Lmb == 0.0
    Lmb_safe = np.where(iso, 1.0, Lmb)  # avoid 0-division in unused branch
    p_gradient = Pb * (Tb / temperature) ** (a.gmr / Lmb_safe)
    p_isothermal = Pb * np.exp(-a.gmr * deltah / Tb)
    pressure = np.where(iso, p_isothermal, p_gradient)

    # Density from ideal gas law
    density = pressure / temperature
    rho = 1e-9 * a.M0 / a.Rstar * density  # g/cm³
    return rho


def lexpr(z, beta, R=6371.0):
    """Compute propagation length from altitude and angle.

    For a shower at altitude z emerging at angle beta from nadir,
    compute the propagation length along the shower axis.

    Parameters
    ----------
    z : float or ndarray
        Altitude (km).
    beta : float or ndarray
        Earth emergence angle from nadir (radians).
    R : float, optional
        Earth radius (km), default 6371.0.

    Returns
    -------
    L : float or ndarray
        Propagation length (km).

    Raises
    ------
    ValueError
        If inputs contain NaN/Inf or R is non-positive.
    """
    z = np.asarray(z, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    if not np.all(np.isfinite(z)):
        raise ValueError("z contains NaN/Inf")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains NaN/Inf")
    if R <= 0.0:
        raise ValueError(f"R must be positive; got {R}")

    Rsinbeta = R * np.sin(beta)
    return -Rsinbeta + np.sqrt(Rsinbeta**2 + z**2 + 2.0 * R * z)


def zexpr(lval, beta, R=6371.0):
    """Compute altitude from propagation length and angle.

    Inverse of lexpr: given propagation length L and angle beta,
    compute the altitude z.

    Parameters
    ----------
    lval : float or ndarray
        Propagation length (km).
    beta : float or ndarray
        Earth emergence angle from nadir (radians).
    R : float, optional
        Earth radius (km), default 6371.0.

    Returns
    -------
    z : float or ndarray
        Altitude (km).

    Raises
    ------
    ValueError
        If inputs contain NaN/Inf or R is non-positive.
    """
    lval = np.asarray(lval, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    if not np.all(np.isfinite(lval)):
        raise ValueError("lval contains NaN/Inf")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains NaN/Inf")
    if R <= 0.0:
        raise ValueError(f"R must be positive; got {R}")

    Rsinbeta = R * np.sin(beta)
    return -R + np.sqrt(R**2 + lval**2 + 2.0 * lval * Rsinbeta)


def viewing_angle(beta_tr, Zdet, Re):
    """Viewing angle from detector to shower emergence point.

    Parameters
    ----------
    beta_tr : float or ndarray
        Earth emergence angle (radians).
    Zdet : float or ndarray
        Detector altitude (km).
    Re : float
        Earth radius (km).

    Returns
    -------
    float or ndarray
        Viewing angle (radians).
    """
    return np.arcsin((Re / (Re + Zdet)) * np.cos(beta_tr))


def dXl(lval, beta, a=None, R=6371.0):
    """Differential slant depth dX/dl at propagation length l.

    Parameters
    ----------
    lval : float or ndarray
        Propagation length (km).
    beta : float or ndarray
        Earth emergence angle from nadir (radians).
    a : Atmosphere, optional
        Atmosphere instance. If None, creates a new one.
    R : float, optional
        Earth radius (km), default 6371.0.

    Returns
    -------
    dXdl : float or ndarray
        Differential slant depth (g/cm²/km).

    Raises
    ------
    ValueError
        If inputs contain NaN/Inf or R is non-positive.
    """
    if a is None:
        a = Atmosphere()

    z = zexpr(lval, beta, R=R)
    return 1e5 * us_std_atm_density(z, R=R, a=a)


def dXl_known_layer(lval, beta, li, a, R=6371.0):
    """``dXl`` when the US-1976 atmospheric layer index ``li`` is already known.

    Identical result to :func:`dXl`, but the caller supplies ``li`` (broadcastable
    to ``lval``) instead of paying the per-element binary-search layer lookup
    ``a(h)`` and the per-element table gathers. This is the form
    :func:`length_at_depth_hermite` wants: its support grid is boundary-aligned,
    so every Gauss-Legendre node within a segment sits in one layer -- the layer
    is resolved once per *segment* (shape ``(nb-1, n_showers)``) and broadcast
    over the node axis, rather than reclassified per node. Bit-identical to
    ``dXl`` whenever ``li`` is the true layer of each point (verified: GL nodes
    are strictly interior to their layer, so per-node == per-segment lookup).

    ``li`` must index the US-1976 tables (0..len(H_b)-1); the caller owns its
    correctness (typically ``a(h)`` of the segment midpoint altitude).
    """
    z = zexpr(lval, beta, R=R)
    h = z * R / (z + R)
    Hb = a.H_b[li]
    Tb = a.T_b[li]
    Lmb = a.Lm_b[li]
    Pb = a.P_b[li]
    deltah = h - Hb
    temperature = Tb + Lmb * deltah
    # Same two-branch pressure as us_std_atm_density (isothermal vs gradient).
    iso = Lmb == 0.0
    Lmb_safe = np.where(iso, 1.0, Lmb)
    p_gradient = Pb * (Tb / temperature) ** (a.gmr / Lmb_safe)
    p_isothermal = Pb * np.exp(-a.gmr * deltah / Tb)
    pressure = np.where(iso, p_isothermal, p_gradient)
    # Match us_std_atm_density's exact op order so the result is bit-identical
    # to ``dXl`` (rho in g/cm^3, then the dXl 1e5 factor).
    density = pressure / temperature
    rho = 1e-9 * a.M0 / a.Rstar * density
    return 1e5 * rho


def d2Xl_known_layer(lval, beta, li, a, R=6371.0):
    """Second derivative ``d²X/dl²`` of slant depth with the layer index ``li`` known.

    The Halley analogue of :func:`dXl_known_layer`: a fully vectorized second
    derivative ``d²X/dl²`` with the caller supplying ``li`` (broadcastable to
    ``lval``), so it pays neither the per-element layer binary-search ``a(h)``
    nor a boolean-mask scatter over atmospheric layers. This is the
    form :func:`length_at_depth_approx` needs for its within-layer Halley step.

    The chain rule gives ``d²X/dl² = (d/dh)(dX/dl) * dh/dl``. With ``dX/dl`` the
    value :func:`dXl_known_layer` returns, the in-layer altitude derivative is
    ``-(dX/dl) * (gmr + Lm_b) / T`` (the ``Lm_b`` term is the lapse-rate
    correction, present only in gradient layers and vanishing in isothermal
    ones), and the geometric factor ``dh/dl = R²(R sinβ + l)/u³`` follows from
    the law-of-cosines ``z(l)``. Validated against a central finite difference
    of :func:`dXl`, the true ``d²X/dl²`` (GL nodes are strictly interior to
    their layer, so the supplied ``li`` is always correct).
    """
    Rsinbeta = R * np.sin(beta)
    u = np.sqrt(R * R + lval * lval + 2.0 * lval * Rsinbeta)
    z = -R + u
    h = z * R / (z + R)
    Hb = a.H_b[li]
    Tb = a.T_b[li]
    Lmb = a.Lm_b[li]
    Pb = a.P_b[li]
    deltah = h - Hb
    temperature = Tb + Lmb * deltah
    iso = Lmb == 0.0
    Lmb_safe = np.where(iso, 1.0, Lmb)
    p_gradient = Pb * (Tb / temperature) ** (a.gmr / Lmb_safe)
    p_isothermal = Pb * np.exp(-a.gmr * deltah / Tb)
    pressure = np.where(iso, p_isothermal, p_gradient)
    dXdl = 1e5 * (1e-9 * a.M0 / a.Rstar * (pressure / temperature))  # = dXl
    # d(dX/dl)/dh = -(dX/dl)(gmr + Lm_b)/T; the Lm_b term is gradient-layer only.
    ddXdl_dh = np.where(
        iso, -dXdl * a.gmr / temperature, -dXdl * (a.gmr + Lmb) / temperature
    )
    dh_dl = R * R * (Rsinbeta + lval) / u**3
    return ddXdl_dh * dh_dl


def total_slant_depth(z_emission, beta, z_top=65.0, R=6371.0, Nq=8, a=None):
    """Compute total atmospheric slant depth from emission altitude to top of atmosphere.

    Parameters
    ----------
    z_emission : float or ndarray
        Emission altitude (km).
    beta : float or ndarray
        Earth emergence angle from nadir (radians).
    z_top : float, optional
        Top of atmosphere altitude (km), default 65.0.
    R : float, optional
        Earth radius (km), default 6371.0.
    Nq : int, optional
        Number of GL quadrature points, default 8.
    a : Atmosphere, optional
        Atmosphere instance. If None, creates a new one.

    Returns
    -------
    X_total : float or ndarray
        Total slant depth from z_emission to z_top (g/cm²).

    Raises
    ------
    ValueError
        If inputs contain NaN/Inf or invalid values.
    """
    if a is None:
        a = Atmosphere()

    z_emission = np.atleast_1d(np.asarray(z_emission, dtype=np.float64))
    beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))

    if z_emission.shape != beta.shape:
        raise ValueError(
            f"z_emission and beta must have same shape; got {z_emission.shape} vs {beta.shape}"
        )
    if not np.all(np.isfinite(z_emission)):
        raise ValueError("z_emission contains NaN/Inf")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains NaN/Inf")
    if not np.isfinite(z_top) or z_top <= 0.0:
        raise ValueError(f"z_top must be finite and positive; got {z_top}")
    if R <= 0.0:
        raise ValueError(f"R must be positive; got {R}")
    if not isinstance(Nq, (int, np.integer)) or Nq <= 0:
        raise ValueError(f"Nq must be a positive integer; got {Nq}")

    # Canonical layer-split slant depth from emission to top of atmosphere.
    L_start = lexpr(z_emission, beta, R=R)
    L_end = lexpr(z_top, beta, R=R)
    X_total = slant_depth(L_start, L_end, beta, R=R, a=a, n=Nq)

    return X_total if X_total.shape != (1,) else float(X_total[0])


def _slant_depth_segment(L0, L1, beta, x, w, a, R, li=None):
    """Slant depth over a single layer-interior interval [L0, L1] by GL.

    Both endpoints must lie within one smooth atmospheric layer (no Z_b
    crossing) so the Gauss-Legendre rule on dXl is exact. ``beta`` must already
    carry the trailing axes needed to broadcast against [L0, L1].

    ``li`` is the US-1976 layer index of the interval, if the caller already
    knows it (e.g. :func:`slant_depth`, whose loop bounds each segment by two
    consecutive ``Z_b``). Supplying it routes the integrand through
    :func:`dXl_known_layer`, skipping the per-node binary-search layer lookup and
    table gather inside ``dXl`` -- bit-identical, since the GL nodes are strictly
    interior to layer ``li`` (a degenerate clamped segment has ``half == 0`` and
    contributes nothing regardless of ``li``). ``li=None`` falls back to ``dXl``.
    """
    half = 0.5 * (L1 - L0)
    mid = 0.5 * (L1 + L0)
    Ln = mid[..., None] + half[..., None] * x
    bb = np.asarray(beta)[..., None]
    # A degenerate clamped segment (half == 0 -- e.g. a shower decaying above the
    # modeled atmosphere collapses every segment to a layer boundary) places its
    # nodes outside layer ``li``, where the known-layer barometric formula can go
    # non-finite. Such a segment contributes nothing, so zero the GL sum where
    # half == 0 before the multiply (avoiding 0 * inf = NaN), and suppress the
    # transient over/invalid from those discarded evaluations. Bit-identical to
    # the plain-dXl path on real segments (half > 0).
    with np.errstate(over="ignore", invalid="ignore"):
        dseg = (
            dXl(Ln, bb, a=a, R=R) if li is None else dXl_known_layer(Ln, bb, li, a, R=R)
        )
        seg = np.where(half > 0, np.sum(w * dseg, axis=-1), 0.0)
    return half * seg


def slant_depth(L0, L1, beta, R=6371.0, a=None, n=8):
    """Canonical slant depth X (g/cm²) along the path from propagation length
    L0 to L1 at emergence angle ``beta``.

    The path is split at the US-1976 atmospheric layer boundaries (Z_b), where
    the density derivative is discontinuous, and each smooth segment is
    integrated with Gauss-Legendre on the exact differential slant depth
    ``dXl``. Exact to machine precision at ``n=8``. This is the single canonical
    slant-depth primitive; it is the exact forward of :func:`length_at_depth`.
    Vectorized over broadcastable ``(L0, L1, beta)``.
    """
    if a is None:
        a = Atmosphere()
    x, w = cached_leggauss(n)
    Zb = a.Z_b
    L0 = np.asarray(L0, dtype=np.float64)
    L1 = np.asarray(L1, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    bshape = np.broadcast_shapes(L0.shape, L1.shape, beta.shape)
    bb = np.broadcast_to(beta, bshape)
    total = np.zeros(bshape)
    for i in range(len(Zb) - 1):
        # Segment i lies between consecutive Z_b boundaries, so its layer index
        # is exactly i -- pass it through to skip dXl's per-node layer lookup.
        lo = np.minimum(np.maximum(lexpr(Zb[i], beta, R=R), L0), L1)
        hi = np.minimum(np.maximum(lexpr(Zb[i + 1], beta, R=R), L0), L1)
        total = total + _slant_depth_segment(lo, hi, bb, x, w, a, R, li=i)
    return total


def length_at_depth(L0, X, beta, R=6371.0, a=None, n=8, niter=8):
    """Canonical inverse of :func:`slant_depth`: the propagation length L where
    the slant depth accumulated from L0 equals X.

    Round-trips :func:`slant_depth` to machine precision. The target's
    atmospheric layer is located from the cumulative slant depth at the layer
    boundaries (layer-walking), then solved within that single smooth layer by
    Newton's method from the layer's lower edge -- ``slant_depth`` is concave in
    L (density falls with altitude), so Newton from below converges
    monotonically without a bracketing safeguard.

    ``L0, beta`` have shape ``S``; ``X`` has shape ``S + (k,)`` (one trailing
    axis of targets, e.g. quadrature nodes). Returns L of shape ``S + (k,)``.
    """
    if a is None:
        a = Atmosphere()
    x, w = cached_leggauss(n)
    Zb = a.Z_b
    L0 = np.asarray(L0, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Lcap = lexpr(Zb[-1], beta, R=R)
    # Layer boundaries in L (clamped to [L0, Lcap]) and cumulative depth at each.
    B = np.stack([np.minimum(np.maximum(lexpr(z, beta, R=R), L0), Lcap) for z in Zb])
    Xc = np.zeros_like(B)
    for i in range(len(Zb) - 1):
        Xc[i + 1] = Xc[i] + _slant_depth_segment(B[i], B[i + 1], beta, x, w, a, R)
    Bm = np.moveaxis(B, 0, -1)  # S + (nb,)
    Xcm = np.moveaxis(Xc, 0, -1)
    # Locate each target's layer, then Newton within it.
    le = Xcm[..., None, :] <= X[..., :, None]  # S + (k, nb)
    idx = np.clip(le.sum(-1) - 1, 0, len(Zb) - 2)  # S + (k,)
    Blo = np.take_along_axis(Bm, idx, axis=-1)
    Bhi = np.take_along_axis(Bm, idx + 1, axis=-1)
    dX = X - np.take_along_axis(Xcm, idx, axis=-1)
    bb = beta[..., None]
    L = Blo.copy()
    for _ in range(niter):
        f = _slant_depth_segment(Blo, L, bb, x, w, a, R) - dX
        L = np.clip(L - f / dXl(L, bb, a=a, R=R), Blo, Bhi)
    return L


def _hermite_support_altitudes(a, z_top, dens):
    """Support altitudes for the inverse-Hermite grid: the US-1976 layer
    boundaries below ``z_top`` plus ``z_top`` itself (so every interval lies
    inside one smooth layer), with ``dens`` interior points inserted per layer
    to resolve the thick low layers where ``L(X)`` is most curved.
    """
    zb = [z for z in a.Z_b if z < z_top] + [float(z_top)]
    if dens <= 0:
        return np.asarray(zb, dtype=np.float64)
    out = []
    for lo, hi in zip(zb[:-1], zb[1:]):
        out.extend(np.linspace(lo, hi, dens + 2)[:-1].tolist())
    out.append(zb[-1])
    return np.asarray(out, dtype=np.float64)


def length_at_depth_hermite(
    L0, X, beta, z_top, R=6371.0, a=None, n=3, dens=1, polish=1, n_polish=2
):
    """Fast inverse of :func:`slant_depth` for a batch of targets sharing each
    path -- the propagation-grid workhorse.

    Builds the inverse function ``L(X)`` once per shower as a cubic Hermite
    spline on a boundary-aligned support grid (so no piece straddles an
    atmospheric-layer kink) and *evaluates* it at every target ``X`` -- no
    per-target root finding. The Hermite uses only the support depths, lengths
    and the slope ``dL/dX = 1/dXl`` (all continuous across layer boundaries), so
    the curvature kinks fall harmlessly on the spline seams.

    Atmosphere-model work is ``O(#support points)`` per shower and independent of
    the number of targets ``k``, which is what makes it fast for the 16-node
    grid. The cubic seed is accurate to ~1e-2 in ``L``; ``polish`` Newton steps
    (reusing the spline's own bracket, so they add only one slant-depth segment
    each) then drive it to the canonical inverse -- ``polish=1`` reaches ~1e-4,
    enough that the result is indistinguishable from :func:`length_at_depth` in
    the grid. ``slant_depth`` is concave in ``L``, so the bracketed Newton step
    is monotone and safe.

    Quadrature orders are deliberately split. The support-grid integration (``n``)
    spans whole atmospheric layers to seed the cubic, but those layers are smooth
    and well-resolved -- ``n=3`` matches ``n=4`` to machine precision. The polish
    residual (``n_polish``) integrates only the tiny ``[y0, L]`` sub-interval
    inside one smooth layer, so ``n_polish=2`` lands the Newton step at ~2e-4 (vs
    the ~1e-3 grid tolerance). Together they roughly halve the atmosphere-density
    evaluations of the former ``n=4``-everywhere scheme.

    ``L0, beta`` have shape ``(n_showers,)``; ``X`` has shape
    ``(n_showers, k)``. Returns ``L`` of shape ``(n_showers, k)``.
    """
    if a is None:
        a = Atmosphere()
    x, w = cached_leggauss(n)
    L0 = np.asarray(L0, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    ZS = _hermite_support_altitudes(a, z_top, dens)
    Lmax = lexpr(z_top, beta, R=R)
    nb = len(ZS)
    # Support lengths (clamped into [L0, Lmax]); cumulative depth and slope there.
    B = np.stack([np.minimum(np.maximum(lexpr(z, beta, R=R), L0), Lmax) for z in ZS])
    # Cumulative slant depth at the supports: GL each [B[i], B[i+1]] segment, but
    # evaluate the atmosphere for ALL segments in one vectorized call rather than
    # a per-support Python loop -- each loop iteration otherwise re-paid the
    # Atmosphere layer-lookup / zexpr / finiteness-scan overhead.
    lo = B[:-1]  # (nb-1, n_showers)
    hi = B[1:]
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    Ln = mid[..., None] + half[..., None] * x  # (nb-1, n_showers, n)
    # Each segment lies in one atmospheric layer, so resolve the layer ONCE per
    # (segment, shower) from the midpoint and reuse it for all n nodes -- skips
    # the per-node binary-search lookup + table gathers inside dXl (bit-identical,
    # since GL nodes are strictly interior to their segment's layer).
    z_mid = zexpr(mid, beta[None, :], R=R)
    # int16 keeps the broadcast-materialized layer-index temporaries small (8
    # layers fit easily); the table gather casts as needed.
    seg_layer = a(z_mid * R / (z_mid + R)).astype(np.int16)  # (nb-1, n_showers)
    # Pass li at FULL node shape (materialized contiguous): the table gather then
    # yields contiguous (nb-1, n_showers, n) params, so the pow/exp run on
    # contiguous memory. A broadcast *view* index gathers slowly (stride-0 read)
    # and (.,.,1) params force a broadcast into the transcendentals -- both lose
    # the win; the small int copy pays for itself.
    li_seg = np.ascontiguousarray(np.broadcast_to(seg_layer[..., None], Ln.shape))
    dseg = dXl_known_layer(Ln, beta[None, :, None], li_seg, a, R=R)
    seg = half * np.sum(w * dseg, axis=-1)
    Xc = np.empty_like(B)
    Xc[0] = 0.0
    Xc[1:] = np.cumsum(seg, axis=0)
    M = 1.0 / dXl(B, beta[None, :], a=a, R=R)  # dL/dX at each support
    Bm = np.moveaxis(B, 0, -1)  # (n_showers, nb)
    Xcm = np.moveaxis(Xc, 0, -1)
    Mm = np.moveaxis(M, 0, -1)
    # Locate each target's support interval, gather its endpoints.
    le = Xcm[:, None, :] <= X[:, :, None]  # (n_showers, k, nb)
    idx = np.clip(le.sum(-1) - 1, 0, nb - 2)  # (n_showers, k)
    x0 = np.take_along_axis(Xcm, idx, -1)
    x1 = np.take_along_axis(Xcm, idx + 1, -1)
    y0 = np.take_along_axis(Bm, idx, -1)
    y1 = np.take_along_axis(Bm, idx + 1, -1)
    m0 = np.take_along_axis(Mm, idx, -1)
    m1 = np.take_along_axis(Mm, idx + 1, -1)
    # Cubic Hermite of L(X) on [x0, x1] (guard degenerate clamped intervals).
    h = x1 - x0
    ok = h > 0
    hsafe = np.where(ok, h, 1.0)
    t = np.where(ok, (X - x0) / hsafe, 0.0)
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    L = np.where(ok, h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1, y0)
    # Newton polish to the canonical inverse, reusing the spline's bracket:
    # solve slant_depth(y0, L) = X - x0 within [y0, y1]. Each step is one extra
    # slant-depth segment (no new support setup). The residual integrates only a
    # tiny in-layer sub-interval, so a lower-order rule (n_polish) suffices. Each
    # target stays inside its bracket's layer, so the layer index is gathered once
    # from seg_layer via the bracket -- no per-iteration binary-search in dXl.
    xp, wp = cached_leggauss(n_polish)
    li_pol = np.take_along_axis(seg_layer.T, idx, axis=1)  # (n_showers, k)
    bb = beta[:, None]
    dX = X - x0
    for _ in range(polish):
        ph = 0.5 * (L - y0)
        pm = 0.5 * (L + y0)
        Lp = pm[..., None] + ph[..., None] * xp  # (n_showers, k, n_polish)
        li_p = np.ascontiguousarray(np.broadcast_to(li_pol[..., None], Lp.shape))
        dpol = dXl_known_layer(Lp, bb[..., None], li_p, a, R=R)
        f = ph * np.sum(wp * dpol, axis=-1) - dX
        L = np.clip(L - f / dXl_known_layer(L, bb, li_pol, a, R=R), y0, y1)
    return L


def length_at_depth_approx(L0, X, beta, z_top, R=6371.0, a=None, Nq=3, niter=1):
    """Fast approximate inverse of :func:`slant_depth` for the propagation grid.

    A faster sibling of :func:`length_at_depth_hermite` exploiting that, for the
    near-ground decays of upward air showers, the vast majority of showers
    accumulate their entire development depth *within the single lowest
    atmospheric layer* (the targets never cross a US-1976 density kink). For
    those showers the slant depth is a smooth integral with no interior kink, so
    each target ``X`` is inverted directly by a single global Gauss-Legendre rule
    plus one cubic Halley root step -- no per-layer split, no per-element layer
    lookup (the layer is the decay layer, resolved once per shower). Halley's
    cubic convergence reaches the grid tolerance from the analytic seed in
    ``niter=1`` step at ``Nq=3``.

    Showers whose deepest target *does* cross into a higher layer (a small
    minority; an un-split global rule loses accuracy across the kink) are
    detected up front -- the depth to the top of the decay layer is one extra
    GL segment -- and delegated to :func:`length_at_depth_hermite`, which
    layer-walks them exactly. The split is by a single boolean mask, so each
    shower is solved by exactly one method.

    Round-trips :func:`slant_depth` to well within the grid's quadrature floor
    (max ~3e-4 relative on the fast path; the fallback is hermite-exact); the
    residual propagation-length error is sub-metre, entering only the weak
    ``z_nodes``/``L_weights`` dependence (the to-node slant depth stays ``X``
    exactly). ``Nq``/``niter`` configure the fast-path quadrature/root order.

    ``L0, beta`` have shape ``(n_showers,)``; ``X`` has shape
    ``(n_showers, k)``. Returns ``L`` of shape ``(n_showers, k)``.
    """
    if a is None:
        a = Atmosphere()
    L0 = np.asarray(L0, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    qx, qw = cached_leggauss(Nq)

    # Decay-point layer (the candidate single layer for every target).
    z_dec = zexpr(L0, beta, R=R)
    h_dec = z_dec * R / (z_dec + R)
    li = a(h_dec).astype(np.int16)  # (n_showers,)
    # Single-layer test: slant depth from L0 to the top of the decay layer
    # (capped at z_top) -- if it covers the deepest target, no kink is crossed.
    z_up = np.minimum(a.Z_b[li + 1], z_top)
    L_up = lexpr(z_up, beta, R=R)
    half = 0.5 * (L_up - L0)
    mid = 0.5 * (L_up + L0)
    q = mid[:, None] + half[:, None] * qx  # (n_showers, Nq)
    li_q = np.broadcast_to(li[:, None], q.shape)
    X_up = half * np.sum(qw * dXl_known_layer(q, beta[:, None], li_q, a, R=R), axis=-1)
    single = X.max(axis=1) <= X_up  # (n_showers,) bool

    L = np.empty_like(X)

    # Fast path: within-layer global Halley on the single-layer showers.
    s = single
    if s.any():
        bs = beta[s][:, None]  # (m, 1)
        x0 = L0[s][:, None]
        y0 = -X[s]  # (m, k)  f(L0) = -X
        lis = li[s][:, None]
        df = dXl_known_layer(x0, bs, lis, a, R=R)
        d2 = d2Xl_known_layer(x0, bs, lis, a, R=R)
        xn = x0 - 2.0 * y0 * df / (2.0 * df * df - y0 * d2)
        for _ in range(niter):
            hh = 0.5 * (xn - x0)
            mm = 0.5 * (xn + x0)
            qq = mm[..., None] + hh[..., None] * qx  # (m, k, Nq)
            li_qq = np.broadcast_to(lis[..., None], qq.shape)
            fn = y0 + np.sum(
                hh[..., None] * qw * dXl_known_layer(qq, bs[..., None], li_qq, a, R=R),
                axis=-1,
            )
            df = dXl_known_layer(xn, bs, lis, a, R=R)
            d2 = d2Xl_known_layer(xn, bs, lis, a, R=R)
            xn = xn - 2.0 * fn * df / (2.0 * df * df - fn * d2)
        L[s] = xn

    # Fallback: layer-walking hermite for the kink-crossing minority.
    if not s.all():
        L[~s] = length_at_depth_hermite(L0[~s], X[~s], beta[~s], z_top, R=R, a=a)

    return L


def slant_depth_intervals(L_start, L_nodes, L_max, beta, Nq=8, a=None, R=6371.0):
    """Compute slant depth integrals for shower simulation nodes.

    For each node, computes:
    - X_to_node: cumulative slant depth from L_start to node
    - X_to_detector: remaining slant depth from node to L_max (detector altitude)
    - X_total: full atmospheric slant depth L_start to L_max

    Parameters
    ----------
    L_start : ndarray
        Starting propagation lengths, shape (n_showers,) or (n_showers, 1).
    L_nodes : ndarray
        Node propagation lengths, shape (n_showers, n_nodes).
    L_max : ndarray
        Maximum propagation lengths (top of atmosphere), shape (n_showers,) or (n_showers, 1).
    beta : ndarray
        Viewing angles, shape (n_showers,) or (n_showers, 1).
    Nq : int, optional
        GL quadrature order for sub-intervals, default 8.
    a : Atmosphere, optional
        Atmosphere instance.
    R : float, optional
        Earth radius (km), default 6371.0.

    Returns
    -------
    X_to_node : ndarray
        Cumulative slant depth to each node, shape (n_showers, n_nodes).
    X_to_detector : ndarray
        Remaining slant depth from each node, shape (n_showers, n_nodes).
    X_total : ndarray
        Total slant depth across full atmosphere, shape (n_showers,).
    """
    if a is None:
        a = Atmosphere()

    # Normalize to (n_showers, 1) so beta/L_start/L_max broadcast against the
    # (n_showers, n_nodes) node axis.
    L_start = np.atleast_1d(L_start)
    L_max = np.atleast_1d(L_max)
    beta = np.atleast_1d(beta)
    L_start = L_start[:, None] if L_start.ndim == 1 else L_start
    L_max = L_max[:, None] if L_max.ndim == 1 else L_max
    beta_2d = beta[:, None] if beta.ndim == 1 else beta

    # Canonical layer-split slant depth (exact across the US-1976 kinks the
    # old single-segment GL straddled).
    X_to_node = slant_depth(L_start, L_nodes, beta_2d, R=R, a=a, n=Nq)
    X_total = slant_depth(L_start[:, 0], L_max[:, 0], beta_2d[:, 0], R=R, a=a, n=Nq)
    X_to_detector = np.maximum(X_total[:, None] - X_to_node, 0.0)

    return X_to_node, X_to_detector, X_total


def shower_propagation_length(
    decay_altitude,
    beta,
    Xtarg,
    zMax=125.0,
    R=6371.0,
    Nqmax=8,
    Nqroot=3,
    Niter=1,
    a=None,
):
    """Calculate shower propagation length to a target slant depth.

    Determines the propagation length of an extensive air shower by finding
    where the accumulated slant depth reaches the target value Xtarg. This is a
    thin wrapper over the canonical :func:`length_at_depth` (exact inverse of the
    layer-split :func:`slant_depth`); showers whose target exceeds the available
    atmospheric depth escape and return the full path length to ``zMax``.

    ``Nqroot``/``Niter`` are retained for backward compatibility but no longer
    affect the result (the former Halley iteration is replaced by the exact,
    machine-precision layer-walking Newton inverse).

    Parameters
    ----------
    decay_altitude : float or ndarray
        Altitude where shower decay occurs (km).
    beta : float or ndarray
        Earth emergence angle from nadir (radians).
    Xtarg : float
        Target slant depth (g/cm²). Shower terminates when slant depth
        reaches this value.
    zMax : float, optional
        Maximum altitude at top of atmosphere (km), default 125.0.
    R : float, optional
        Earth radius (km), default 6371.0.
    Nqmax : int, optional
        GL quadrature order per atmospheric layer, default 8 (exact at 8).
    Nqroot : int, optional
        Deprecated/ignored (former Halley sub-quadrature order).
    Niter : int, optional
        Deprecated/ignored (former Halley iteration count).
    a : Atmosphere, optional
        Atmosphere instance. If None, creates a new one.

    Returns
    -------
    propagation_len : float or ndarray
        Propagation length (km). If shower reaches target slant depth
        before exiting atmosphere, returns the length to target.
        Otherwise returns full atmospheric path length.

    Raises
    ------
    ValueError
        If decay_altitude and beta have mismatched shapes, or if inputs
        contain NaN/Inf values.

    Notes
    -----
    - For showers that escape the atmosphere before reaching Xtarg,
      returns the full atmospheric path length to ``zMax``.
    - For showers that reach Xtarg within the atmosphere, returns the exact
      length to target (round-trips :func:`slant_depth` to machine precision).

    Examples
    --------
    >>> # Single shower at 10 km altitude, 20 deg from nadir
    >>> L = shower_propagation_length(10.0, np.radians(20.0), 3882.56)
    >>> print(f"Propagation length: {L:.2f} km")

    >>> # Batch of showers
    >>> decay_alts = np.linspace(0.5, 15, 100)
    >>> betas = np.radians(np.linspace(1, 45, 100))
    >>> Ls = shower_propagation_length(decay_alts, betas, 3882.56)
    """
    if a is None:
        a = Atmosphere()

    # Validate inputs
    return_as_scalar = np.isscalar(decay_altitude) and np.isscalar(beta)
    decay_altitude = np.atleast_1d(decay_altitude)
    beta = np.atleast_1d(beta)

    if decay_altitude.shape != beta.shape:
        raise ValueError(
            f"decay_altitude and beta must have same shape; "
            f"got {decay_altitude.shape} vs {beta.shape}"
        )
    if not np.all(np.isfinite(decay_altitude)):
        raise ValueError("decay_altitude contains NaN/Inf")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains NaN/Inf")
    Xtarg = np.atleast_1d(np.asarray(Xtarg, dtype=np.float64))
    if not np.all(np.isfinite(Xtarg) & (Xtarg > 0.0)):
        raise ValueError("Xtarg must be finite and positive")
    if (
        Xtarg.shape != ()
        and Xtarg.shape != (1,)
        and Xtarg.shape != decay_altitude.shape
    ):
        raise ValueError(
            f"Xtarg shape {Xtarg.shape} must be scalar or match "
            f"decay_altitude shape {decay_altitude.shape}"
        )
    # Broadcast Xtarg to match shower shape
    Xtarg = np.broadcast_to(Xtarg, decay_altitude.shape)
    if not np.isfinite(zMax) or zMax <= 0.0:
        raise ValueError(f"zMax must be finite and positive; got {zMax}")
    if R <= 0.0:
        raise ValueError(f"R must be positive; got {R}")
    if not isinstance(Nqmax, (int, np.integer)) or Nqmax <= 0:
        raise ValueError(f"Nqmax must be a positive integer; got {Nqmax}")
    if not isinstance(Nqroot, (int, np.integer)) or Nqroot <= 0:
        raise ValueError(f"Nqroot must be a positive integer; got {Nqroot}")
    if not isinstance(Niter, (int, np.integer)) or Niter <= 0:
        raise ValueError(f"Niter must be a positive integer; got {Niter}")
    if np.any(decay_altitude < 0.0):
        raise ValueError("decay_altitude must be non-negative")
    if np.any(decay_altitude >= zMax):
        raise ValueError(f"decay_altitude must be < zMax={zMax}")

    decay_proplen = lexpr(decay_altitude, beta, R=R)
    atmos_proplen = lexpr(zMax, beta, R=R)

    # Total slant depth available decay -> top of atmosphere (canonical
    # layer-split forward). Showers whose target exceeds this escape and return
    # the full atmospheric path length; the rest are inverted exactly.
    X_total = slant_depth(decay_proplen, atmos_proplen, beta, R=R, a=a, n=Nqmax)
    escape = X_total <= Xtarg

    # length_at_depth wants the target depth on a trailing axis; one target each.
    L_reached = length_at_depth(
        decay_proplen, Xtarg[..., None], beta, R=R, a=a, n=Nqmax
    )[..., 0]

    propagation_len = np.where(
        escape, atmos_proplen - decay_proplen, L_reached - decay_proplen
    )

    return propagation_len[0] if return_as_scalar else propagation_len


def propagation_grid(
    alt,
    betaE,
    n_nodes,
    n_slant_sub,
    atm,
    R,
    z_shower_top,
    X_lo_floor,
    X_death,
    X_peak_depth,
    cloud_top=None,
):
    """GL-in-slant-depth propagation grid along each shower axis.

    Three concerns, in order: the valid slant-depth window, the node grid
    within it, and the geometry of those nodes. Because the grid spans
    exactly the valid window, every node is physically valid by construction
    -- no downstream masking. Returns ``L_max, X_top``, the slant depth to
    each node ``X_nodes``, the per-node ``L_nodes, L_weights, z_nodes``, and
    the altitude ``z_peak`` of the shower maximum (for the detector geometry).

    ``cloud_top`` (``(N,)`` altitudes or ``None``) raises the window's lower
    bound to where the shower clears the clouds (see :meth:`_valid_window`).

    Shapes: ``alt, betaE, Eshow, gb`` are ``(N,)``. Returns ``L_max (N,)``,
    ``X_top (N,)``, ``X_nodes (N, k)``, ``L_nodes (N, k)``,
    ``L_weights (N, k)``, ``z_nodes (N, k)``, ``z_peak (N,)``.
    """
    L_start = lexpr(alt, betaE, R=R)
    L_max = lexpr(float(z_shower_top), betaE, R=R)
    X_lo, X_hi, X_top = visibility_window(
        L_start, L_max, betaE, X_lo_floor, X_death, n_slant_sub, atm, R, cloud_top
    )
    X_nodes, wX, X_peak = gl_node_grid(X_lo, X_hi, X_peak_depth, n_nodes)
    L_nodes, L_weights, z_nodes, z_peak = node_geometry(
        L_start, X_nodes, X_peak, wX, betaE, atm, R, z_shower_top
    )
    return L_max, X_top, X_nodes, L_nodes, L_weights, z_nodes, z_peak


def visibility_window(
    L_start, L_max, betaE, X_lo_floor, X_death, n_slant_sub, atm, R, cloud_top=None
):
    """Slant-depth window ``[X_lo, X_hi]`` over which every node is valid.

    The endpoints are the shower's *visible* slant-depth extremes:

    * ``X_lo = max(_X_LO_COEFF * gb, X_cloud)`` -- the lower of the two
      obscurations of the shower head. ``_X_LO_COEFF * gb`` is the depth
      where the shower age first clears the e2hill floor (below it the Hillas
      angular parameterization is undefined); ``X_cloud`` is the slant depth
      to the cloud top (light generated below the clouds is obscured). With
      no clouds (``cloud_top is None``) only the age floor applies.
    * ``X_hi = min(Xtarg, X_top)`` -- the Greisen-death depth (particle count
      back to 1), capped at ``X_top``, the slant depth to the shower's
      visible top ``z_shower_top = min(detector_altitude, zMaxZ)``. Light
      generated above the detector is beamed behind it (lost); light above
      ``zMaxZ`` is from air too rarified to radiate. For an orbital detector
      the radiation cap binds; for a low detector (balloon) the detector
      altitude truncates the visible tail.

    Folding the cloud cutoff into ``X_lo`` (rather than zeroing sub-cloud
    nodes after the fact) keeps every Gauss-Legendre node inside the visible
    window, so none of the quadrature resolution is wasted below the clouds.

    A degenerate shower (visible window of zero or negative width -- decay
    above the visible top, or clouds above the death depth) has
    ``X_hi`` clamped up to ``X_lo``, so it yields exactly zero without
    masking. Also returns ``X_top`` (the to-detector column reference).

    Shapes: ``L_start, L_max, betaE, gb, Eshow`` are ``(N,)``; ``cloud_top``
    is ``(N,)`` altitudes or ``None``. Returns ``X_lo, X_hi, X_top`` each
    ``(N,)``.
    """
    X_top = slant_depth(L_start, L_max, betaE, R=R, a=atm, n=n_slant_sub)
    X_lo = np.atleast_1d(X_lo_floor)
    if cloud_top is not None:
        # Depth from decay to where the shower rises above the cloud top.
        # Clamp the cloud length to >= L_start so a decay already above the
        # clouds contributes no extra lower bound (X_cloud = 0).
        L_cloud = np.maximum(lexpr(cloud_top, betaE, R=R), L_start)
        X_cloud = slant_depth(L_start, L_cloud, betaE, R=R, a=atm, n=n_slant_sub)
        X_lo = np.maximum(X_lo, X_cloud)
    X_hi = np.maximum(np.minimum(np.atleast_1d(X_death), X_top), X_lo)
    return X_lo, X_hi, X_top


def gl_node_grid(X_lo, X_hi, X_peak_depth, n_nodes):
    """GL nodes in slant depth across ``[X_lo, X_hi]``, split at shower max.

    Slant depth is the shower's natural development variable (shower age is a
    function of X), so the peaked Greisen yield is smooth in X and
    Gauss-Legendre resolves it well -- markedly better than placing nodes in
    path length. The split at the shower-maximum depth (Greisen peak, s = 1
    at ``X_peak = 36.66 * gb``) puts half the nodes on the rising edge and
    half on the falling edge, doubling node density where the shower radiates
    and recovering accuracy at n=16. Returns the node depths ``X_nodes``,
    their Gauss-Legendre weights ``wX`` in X, and the shower-maximum depth
    ``X_peak`` (clamped into the window) for the detector geometry.

    Shapes: ``X_lo, X_hi, gb`` are ``(N,)``; returns ``X_nodes (N, k)``,
    ``wX (N, k)``, ``X_peak (N,)``. The k nodes are two stacked panels of
    ``k//2`` (rising edge) and ``k - k//2`` (falling edge).
    """
    X_peak = np.clip(np.atleast_1d(X_peak_depth), X_lo, X_hi)
    n1 = n_nodes // 2
    ref_x1, ref_w1 = cached_leggauss(n1)
    ref_x2, ref_w2 = cached_leggauss(n_nodes - n1)
    h1 = 0.5 * (X_peak - X_lo)
    h2 = 0.5 * (X_hi - X_peak)
    Xn1 = (0.5 * (X_peak + X_lo))[:, None] + h1[:, None] * ref_x1
    Xn2 = (0.5 * (X_hi + X_peak))[:, None] + h2[:, None] * ref_x2
    X_nodes = np.concatenate([Xn1, Xn2], axis=1)  # (n_showers, n_nodes)
    wX = np.concatenate([h1[:, None] * ref_w1, h2[:, None] * ref_w2], axis=1)
    return X_nodes, wX, X_peak


def node_geometry(L_start, X_nodes, X_peak, wX, betaE, atm, R, z_shower_top):
    """Map node depths to propagation length, altitude, and path weight.

    Inverts slant depth for all node depths -- plus the shower-maximum depth
    ``X_peak`` -- at once with :func:`length_at_depth_approx` (single global
    Halley step per target for the near-ground-decay showers whose nodes stay
    within one atmospheric layer, exact layer-walking fallback for the
    kink-crossing minority). The to-node slant depth is ``X_nodes`` exactly
    regardless, so L enters only the weak ``z_nodes``/``L_weights``
    dependence -- far below the grid's quadrature floor. The path-length
    quadrature weight follows from the change of variable
    ``dL = dX / (dX/dL) = dX / dXl(L)``. Also returns the shower-maximum
    altitude ``z_peak``.

    Shapes: ``L_start, X_peak, betaE`` are ``(N,)``; ``X_nodes, wX`` are
    ``(N, k)``. The inverse is solved for ``(N, k+1)`` targets at once (the k
    nodes plus X_peak). Returns ``L_nodes (N, k)``, ``L_weights (N, k)``,
    ``z_nodes (N, k)``, ``z_peak (N,)``.
    """
    targets = np.concatenate([X_nodes, X_peak[:, None]], axis=1)
    L_all = length_at_depth_approx(
        L_start, targets, betaE, float(z_shower_top), R=R, a=atm
    )
    L_nodes = L_all[:, :-1]
    z_nodes = zexpr(L_nodes, betaE[:, None], R=R)
    z_peak = zexpr(L_all[:, -1], betaE, R=R)
    L_weights = wX / dXl(L_nodes, betaE[:, None], a=atm, R=R)
    return L_nodes, L_weights, z_nodes, z_peak
