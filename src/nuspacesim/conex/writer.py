r"""Map nuSpaceSim shower profiles onto the CONEX format and write it.

:class:`ConexWriter` is an ``on_shower_profile`` callable for
:func:`nuspacesim.compute.compute`::

    compute(config, on_shower_profile=ConexWriter("conex_run.root"))
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit, least_squares

from ..simulation.eas_optical.cphotang import ShowerProfile
from ..simulation.eas_optical.propagation import lexpr, slant_depth
from .format import empty_header, empty_showers
from .root_file import write_trees

__all__ = ["ConexWriter", "build_showers", "conex_path"]


def conex_path(output_file) -> Path:
    """CONEX file name for a results file: ``dir/conex_<stem>.root``."""
    p = Path(output_file)
    name = p.with_suffix(".root").name
    return p.with_name(name if name.startswith("conex") else f"conex_{name}")


def gaisser_hillas(X, X0, Xmax, Nmax, p3, p2, p1):
    lam = p3 * X**2 + p2 * X + p1
    return (
        Nmax
        * ((X - X0) / (Xmax - X0)) ** ((Xmax - X0) / lam)
        * np.exp((Xmax - X) / lam)
    )


def _fit_gh(X, N):
    """Six-parameter Gaisser-Hillas fit of one profile, CONEX conventions.

    Fits nodes up to twice the peak index, shifted to the first node and with N
    scaled by 1e-5. Returns ``(X0, Xmax, Nmax, p1, p2, p3, chi2)``; all NaN when
    there are no more points than parameters or no finite fit is found.
    """
    x0 = X[0]
    stop = 2 * int(np.argmax(N))
    x, y = X[:stop] - x0, N[:stop] / 1e5
    if x.size <= 6:
        return (np.nan,) * 7
    peak = int(np.argmax(y))
    init = [-0.30943336 * x[peak], x[peak], y[peak], 1e-7, 4e-4, 44.0]
    popt = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (OptimizeWarning, RuntimeWarning))
        try:
            popt, _ = curve_fit(gaisser_hillas, x, y, p0=init, maxfev=1_000_000)
        except (RuntimeError, ValueError):
            pass
        if popt is None or not np.all(np.isfinite(popt)):
            popt = least_squares(
                lambda p: gaisser_hillas(x, *p) - y, init, max_nfev=1_000_000
            ).x
        if not np.all(np.isfinite(popt)):
            return (np.nan,) * 7
        yfit = gaisser_hillas(x, *popt)
        pos = y > 0
        chi2 = np.sum((y[pos] - yfit[pos]) ** 2 / y[pos])
        chi2 = chi2 / (x.size - 6) / np.sqrt(popt[2] * 1e5) * 1e5
    g3, g2, g1 = popt[3], popt[4], popt[5]
    # Undo the shift X -> X - x0 (lambda's coefficients re-expand about 0).
    return (
        popt[0] + x0,
        popt[1] + x0,
        popt[2] * 1e5,
        g1 - g2 * x0 + g3 * x0**2,
        g2 - 2 * g3 * x0,
        g3,
        chi2,
    )


def build_showers(profile: ShowerProfile, rng: np.random.Generator) -> np.ndarray:
    """Fill CONEX ``Shower`` records from a profile batch.

    Keeps showers whose particle maximum lies inside the node window (first and
    last node below the peak). Depths are measured from ground along the axis:
    ``X`` adds the ground-to-decay slant depth ``Xfirst``.
    """
    N = profile.N
    peak_N = N.max(axis=1)
    keep = (N[:, 0] < peak_N) & (N[:, -1] < peak_N)
    beta, alt, N, z = profile.beta[keep], profile.altDec[keep], N[keep], profile.z[keep]
    n, n_nodes = N.shape

    L_ground = lexpr(0.0, beta)
    Xfirst = slant_depth(L_ground, lexpr(alt, beta), beta)
    X = profile.X[keep] + Xfirst[:, None]
    fit = np.array([_fit_gh(X[i], N[i]) for i in range(n)]).reshape(n, 7)

    rows = np.arange(n)
    peak = np.argmax(N, axis=1)
    Xmx = X[rows, peak]
    # Energy deposit: age about the fitted Xmax (peak node if the fit failed).
    xmax_ref = np.where(np.isnan(fit[:, 1]), Xmx, fit[:, 1])
    age = 3.0 * X / (X + 2.0 * xmax_ref[:, None])
    alpha = (47.9511 / (0.971315 + age) ** 9.23001 + 2.29587 + 0.285196 * age) * 1e-3
    dEdX = alpha * N

    out = empty_showers(n, n_nodes)
    out["lgE"] = np.log10(profile.showerEnergy[keep]) + 17.0  # 100 PeV = 1e17 eV
    out["zenith"] = 90.0 + np.degrees(beta)
    out["azimuth"] = 360.0 * rng.random(n)
    out["Xfirst"] = Xfirst
    out["Hfirst"] = alt * 1e3
    for j, name in enumerate(("X0", "Xmax", "Nmax", "p1", "p2", "p3", "chi2")):
        out[name] = fit[:, j]
    out["Xmx"] = out["XmxdEdX"] = Xmx
    out["Nmx"] = N[rows, peak]
    out["dEdXmx"] = dEdX[rows, peak]
    out["X"] = X
    out["N"] = out["Electrons"] = N
    out["H"] = z * 1e3
    out["D"] = (lexpr(z, beta[:, None]) - L_ground[:, None]) * 1e3
    out["dEdX"] = dEdX
    return out


class ConexWriter:
    """``on_shower_profile`` callable writing the profiles to a CONEX ROOT file."""

    def __init__(self, path, rng=None):
        self.path = Path(path)
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, profile: ShowerProfile) -> None:
        write_trees(
            self.path,
            {
                "Header": ("run header", empty_header()),
                "Shower": ("shower info", build_showers(profile, self.rng)),
            },
        )
