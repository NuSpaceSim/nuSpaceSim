# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
r"""Cherenkov photon density and angle determination class.


.. autosummary::
   :toctree:
   :recursive:

    CphotAng

"""

import os
import threading
import warnings
from dataclasses import dataclass

import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster, as_completed
from numpy.polynomial import Polynomial
from rich.progress import Progress

from .hillas_batch_kernel import (
    _cached_leggauss,
    delta_nphots_single_integral_NE16,
    precompute_kernel_NE16,
)
from .hillas_kernel import hillas_w_mean
from .propagation import (
    Atmosphere,
    dXl,
    length_at_depth_approx,
    lexpr,
    shower_propagation_length,
    slant_depth,
    viewing_angle,
    zexpr,
)
from .shower_properties import (
    greisen_particle_count,
    particle_count_fluctuated_gaisser_hillas,
    particle_count_parameterized_gaisser_hillas,
    propagation_angle,
    slant_depth_of_greisen_particle_count,
)

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files

__all__ = [
    "BackgroundCluster",
    "CphotAng",
    "PhotonYieldInputs",
    "hillas_single_integral_model",
]


@dataclass(frozen=True)
class PhotonYieldInputs:
    """Per-(shower, node) physical fields handed to a photon-yield model.

    A *photon-yield model* is any callable
    ``model(inputs: PhotonYieldInputs) -> node_contribs`` returning the per-node
    photon contribution as a ``(n_showers, n_nodes)`` float array; ``run()`` sums
    that over the node axis to get the per-shower photon count. This is the
    plug-in boundary that lets the default Gaisser/Hillas single-integral kernel
    be replaced by a different parameterization, a fast function approximation,
    or an entirely different physical model -- without touching the propagation
    grid, atmosphere, cloud, or Cherenkov-angle stages around it.

    Contract a model must honor:

    * Return shape ``(n_showers, n_nodes)`` (== ``sig_per_node.shape``).
    * A node with ``sig_per_node == 0`` does not radiate (it was clipped by the
      valid window or by clouds); the returned contribution there MUST be 0.
    * Results should be per-row independent (each node uses only its own fields)
      so batched evaluation equals per-shower evaluation.

    Attributes
    ----------
    eCthres : ndarray, shape (n_showers, n_nodes)
        Cherenkov threshold energy at each node (MeV).
    thetaC : ndarray, shape (n_showers, n_nodes)
        Cherenkov emission angle at each node (rad).
    sig_per_node : ndarray, shape (n_showers, n_nodes)
        Per-node yield scaling (sin^2(thetaC) * GL weight * particle count *
        atmospheric transmission). Zero marks a non-radiating node.
    e2hill : ndarray, shape (n_showers, n_nodes)
        Hillas E2 angular-distribution parameter (MeV).
    E0 : ndarray, shape (n_showers, n_nodes)
        Hillas E0 track-length-spectrum parameter.
    s : ndarray, shape (n_showers, n_nodes)
        Shower age at each node.
    Eshow : ndarray, shape (n_showers,)
        Primary shower energy (GeV).
    n_nodes : int
        Number of longitudinal nodes per shower (the second array axis).
    """

    eCthres: np.ndarray
    thetaC: np.ndarray
    sig_per_node: np.ndarray
    e2hill: np.ndarray
    E0: np.ndarray
    s: np.ndarray
    Eshow: np.ndarray
    n_nodes: int


# Lower edge of the shower's valid slant-depth window. The Hillas e2 parameter
# e2hill = 1150 + 454*ln(s) vanishes at the shower-age floor s_floor; below it
# the angular parameterization is undefined. Converting that age floor to slant
# depth via s = 3t/(t + 2*gb), t = X/36.66 gives X_lo = _X_LO_COEFF * gb, the
# exact depth at which every node first becomes physically valid.
_S_FLOOR = np.exp(-1150.0 / 454.0)  # shower age where e2hill == 0  (~0.07938)
_X_LO_COEFF = 36.66 * 2.0 * _S_FLOOR / (3.0 - _S_FLOOR)  # ~1.9929 g/cm^2 per gb


def tracklen(E0, eCthres, s):
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
    weighted by the track they lay down. ``E0 = e0(s)`` is the age-dependent
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


def hillas_single_integral_model(n_energy_low=3, n_energy_high=8, dtype=np.float32):
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

    def model(inputs):
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

        # wx_all = max(-dT/dE, 0) for T = tracklen(E0, E, s). Closed-form
        # derivative avoids a second (n_showers, n_nodes, n_E) tracklen call:
        #   -dT/dE = s * T * (1/(E0 + E) + 2e-4 / (1 + 1e-4*s*E))
        # Built in place to avoid spinning up transient (n, k, n_E) arrays for
        # each sub-expression (this block is the run() working-set peak).
        E0_b = E0[:, :, None].astype(dtype)
        s_b = s[:, :, None].astype(dtype)
        T = tracklen(E0_b, E_grid, s_b)  # (n_showers, n_nodes, n_E)
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

    return model


def e0(shape, s):
    """Age-dependent scale energy E0(s) (MeV) of the track-length spectrum.

    The ``E0`` appearing in :func:`tracklen` (Hillas 1982, eqn (8), p. 1466)::

        E0(s) = 44 - 17*(s - 1.46)**2   if s >= 0.4
              = 26                        if s <  0.4
    """
    E0 = np.full(shape, 26.0, dtype=np.float64)
    E0[s >= 0.4] = 44.0 - 17.0 * (s[(s >= 0.4)] - 1.46) ** 2
    return E0


def cherenkov_threshold_angle(AirN):
    """Calc Cherenkov Threshold energy and Cherenkov angle."""
    eCthres = 0.511 / np.sqrt(1.0 - np.reciprocal(np.power(AirN, 2)))
    thetaC = np.arccos(np.reciprocal(AirN))
    return eCthres, thetaC


def d_to_det(ThetView, ThetPrpA, zs, RadE):
    """Distance to detector."""
    AngE = np.pi / 2 - ThetView - ThetPrpA
    DistStep = np.sin(AngE) / np.sin(ThetView) * (RadE + zs)
    return DistStep


def cherenkov_area(AveCangI, dist_to_max):
    """Cherenkov area (km²) from the mean angle and the distance to the
    shower maximum. ``dist_to_max`` is the detector distance evaluated at the
    shower-maximum point (one value per shower)."""
    CherArea = np.tan(AveCangI) * 1e3 * dist_to_max
    return np.pi * CherArea**2


# ---------------------------------------------------------------------------
# Pure functions: atmospheric columns, field evaluation, transmission
# ---------------------------------------------------------------------------


def ozone_column(L_nodes, L_max, betaE, oz_zbnd, oz_seg_rate, R):
    """Analytic ozone column from each node to L_max (exact layer-walk).

    The ozone rate ``-dTotZon/dz`` is piecewise-constant in altitude, so the
    slant ozone column ``integral rate(z(L)) dL`` from a node to the visible top
    is just the sum, over the ozone altitude segments, of each segment's constant
    rate times the path length the ray spends in it. That path length is
    ``L(z_top_seg) - L(z_bot_seg)`` (via :func:`lexpr`) clamped to the node->top
    interval. Exact (no quadrature) and free of the per-node ``searchsorted`` and
    the ``(n, k, n_slant_sub)`` GL temporaries the former sub-quadrature needed --
    and more accurate, since Gauss-Legendre converged poorly across the rate's
    discontinuities (~5% median, ~30% max error at 8 nodes).

    Parameters
    ----------
    L_nodes : ndarray, shape (n_showers, n_nodes)
    L_max : ndarray, shape (n_showers,)
    betaE : ndarray, shape (n_showers,)
    oz_zbnd : ndarray, shape (n_seg + 1,)
        Ascending ozone altitude boundaries (km), ``[0, *OzZeta]``.
    oz_seg_rate : ndarray, shape (n_seg,)
        Constant ozone rate within each ``[oz_zbnd[j], oz_zbnd[j+1]]`` segment.
    R : float
        Earth radius (km).

    Returns
    -------
    ZonZ : ndarray, shape (n_showers, n_nodes)
    """
    # Propagation length of each ozone boundary, clamped into each ray's
    # [L_node, L_max] window: a segment outside the window contributes zero.
    Lb = lexpr(oz_zbnd[:, None], betaE[None, :], R=R)  # (n_seg+1, n_showers)
    Lc = np.clip(Lb[:, :, None], L_nodes[None], L_max[None, :, None])
    ZonZ = np.einsum("s,snk->nk", oz_seg_rate, Lc[1:] - Lc[:-1])
    # An ozone column is non-negative; a degenerate (decay-above-atmosphere)
    # shower has L_node >= L_max so every segment clamps to zero width -> 0.
    return np.maximum(ZonZ, 0.0)


def aerosol_column(L_nodes, L_max, betaE, aero_zbnd, aero_ext, R):
    """Slant aerosol optical depth from each node to the top of the aerosol layer.

    Identical in form to :func:`ozone_column`: the aerosol extinction
    (``dfaOD55``, the per-km vertical optical depth) is piecewise-constant in
    altitude, so the slant column ``integral ext(z(L)) dL`` is the sum over the
    1 km aerosol bands of each band's constant extinction times the path length
    the ray spends in it, ``L(z_top) - L(z_bot)`` (via :func:`lexpr`) clamped to
    the node->top window.

    This is the **curved-atmosphere** slant column. It replaces the former
    plane-parallel ``vertical_OD / cos(theta)``, which (like ``main``) diverges
    as ``cos(theta) -> 0`` at grazing incidence and so over-attenuates -- often
    zeroing -- the most grazing showers. For a vertical shower the walk reduces
    exactly to the vertical column ``aOD55(z_node)``.

    Parameters
    ----------
    L_nodes : ndarray, shape (n_showers, n_nodes)
    L_max : ndarray, shape (n_showers,)
    betaE : ndarray, shape (n_showers,)
    aero_zbnd : ndarray, shape (n_band + 1,)
        Ascending aerosol altitude boundaries (km), ``[0, 1, ..., 30]``.
    aero_ext : ndarray, shape (n_band,)
        Constant aerosol extinction (optical depth per km) within each band.
    R : float
        Earth radius (km).

    Returns
    -------
    AODepth : ndarray, shape (n_showers, n_nodes)
        Slant aerosol optical depth (non-negative).
    """
    Lb = lexpr(aero_zbnd[:, None], betaE[None, :], R=R)  # (n_band+1, n_showers)
    Lc = np.clip(Lb[:, :, None], L_nodes[None], L_max[None, :, None])
    AODepth = np.einsum("s,snk->nk", aero_ext, Lc[1:] - Lc[:-1])
    return np.maximum(AODepth, 0.0)


def refractive_index(z):
    """Refractive index of air from altitude (z) by vertical column_depth (g/cm²).
    numerical parameterization for 1976 STD atmosphere.
    """

    # formerly grammage
    Xv = np.empty_like(z)
    mask1 = z < 11
    mask2 = (z >= 11) & (z < 25)
    mask3 = z >= 25
    Xv[mask1] = ((z[mask1] - 44.34) / -11.861) ** (1 / 0.19)
    Xv[mask2] = np.exp((z[mask2] - 45.5) / -6.34)
    Xv[mask3] = np.exp(13.841 - np.sqrt(28.920 + 3.344 * z[mask3]))

    return 1.0 + 0.000296 * (Xv / 1032.9414) * (273.2 / (204.0 + 0.091 * Xv))


def shower_age(X_to_node, gb):
    """Atmospheric depth parameter t and shower age s.

    Parameters
    ----------
    X_to_node : ndarray, shape (..., n_nodes)
        Cumulative slant depth to each node (g/cm²).
    gb : ndarray, shape (n_showers,)
        Greisen beta = log(Eshow / ecrit).

    Returns
    -------
    t : ndarray, shape (..., n_nodes)
    s : ndarray, shape (..., n_nodes)
    """
    t = X_to_node / 36.66
    s = 3.0 * t / (t + 2.0 * np.atleast_1d(gb)[..., None])
    return t, s


def greisen_particles(t, s, gb):
    """Greisen particle count at each node.

    Parameters
    ----------
    t, s : ndarray, shape (..., n_nodes)
    gb : ndarray, shape (n_showers,)

    Returns
    -------
    RN : ndarray, shape (..., n_nodes)
    """
    alpha = 0.31 / np.sqrt(np.atleast_1d(gb))
    safe_s = np.where(s > 0, s, 1.0)
    RN = alpha[..., None] * np.exp(t * (1.0 - 1.5 * np.log(safe_s)))
    return np.maximum(RN, 0.0)


def hillas_e2(s):
    """Hillas E-squared parameter e2hill from shower age."""
    safe_s = np.where(s > 0, s, 1.0)
    return np.where(s > 0, 1150.0 + 454.0 * np.log(safe_s), 0.0)


def node_fields(z_nodes, X_to_node, gb):
    """Per-node shower fields from the propagation grid.

    Returns
    -------
    AirN : refractive index of air at each node
    s : shower age
    RN : Greisen particle count
    e2hill : Hillas E² parameter
    """
    AirN = refractive_index(z_nodes)
    t, s = shower_age(X_to_node, gb)
    RN = greisen_particles(t, s, gb)
    e2hill = hillas_e2(s)
    return AirN, s, RN, e2hill


# Dask block size (events/block) for __call__. Throughput is flat over a wide
# plateau of ~16k-50k events/block and falls off sharply BELOW it: now that
# run() costs ~0.012 ms/shower, the per-block fixed cost (dask task scheduling,
# Atmosphere() setup, and the GIL-held Python orchestration inside run()) is the
# binding constraint, so too-small blocks dominate (8k is ~24% slower than 16k,
# 1k is ~4x slower). The plateau ceiling is memory bandwidth: the per-block
# (n, n_nodes, n_wl) working set thrashes for very large blocks. Both bounds are
# in EVENT COUNT and roughly machine-independent (overhead- and bandwidth-set),
# unlike the old "blocks per core" assumption -- core count enters only as the
# block-count target so all cores stay busy at large N. Measured on real
# (grazing) showers, N=1e5 and 1e6, 10 cores; BLAS thread pinning is a non-factor
# (the transmission gemm is too small to spawn threads).
_CHUNK_MIN = 16_000
_CHUNK_MAX = 24_000
_CHUNK_BLOCKS_PER_CORE = 1


def _auto_chunk_size(n_events):
    """Parallelism-aware dask block size for :meth:`CphotAng.__call__`.

    Targets ~``_CHUNK_BLOCKS_PER_CORE`` block(s) per usable core, clamped to the
    empirically-flat optimal plateau ``[_CHUNK_MIN, _CHUNK_MAX]``. With one block
    per core the clamp does the real work: small/medium N lands on ``_CHUNK_MIN``
    (overhead-amortized, even if that leaves some cores idle -- the job is tiny
    then), and large N lands on ``_CHUNK_MAX`` (many blocks, bandwidth-capped).
    Uses the affinity-aware core count where available so it respects
    cgroup/taskset limits.
    """
    try:
        ncores = len(os.sched_getaffinity(0))  # honors affinity / cgroup pinning
    except AttributeError:
        ncores = os.cpu_count() or 4
    chunk = -(-int(n_events) // (_CHUNK_BLOCKS_PER_CORE * max(ncores, 1)))  # ceil div
    return int(min(max(chunk, _CHUNK_MIN), _CHUNK_MAX))


class BackgroundCluster:
    """A process-based dask ``LocalCluster`` spun up in a background thread.

    A simulation runs exactly one :meth:`CphotAng.__call__`, so the cluster's
    whole life fits inside that single call -- but its ~2s process spawn would
    otherwise be paid serially right when the EAS optical stage starts. Creating
    it here, at the top of the pipeline, lets the worker spawn overlap the
    geometry/spectra/tau/decay stages (which release the GIL in their numpy/
    C work, so the spawning thread makes real progress). By the time
    :meth:`client` is called the workers are warm.

    Own the lifecycle from the caller: construct early, pass :meth:`client` into
    the EAS optical call, then :meth:`close`. Construction failures are deferred
    and re-raised from :meth:`client` so the caller's stack frame sees them.
    """

    def __init__(self):
        self._holder = {}
        self._error = None
        self._thread = threading.Thread(target=self._spawn, daemon=True)
        self._thread.start()

    def _spawn(self):
        try:
            cluster = LocalCluster(processes=True)
            self._holder["cluster"] = cluster
            self._holder["client"] = Client(cluster)
        except BaseException as exc:
            self._error = exc

    def client(self):
        """Block until the cluster is up and return its :class:`Client`."""
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._holder["client"]

    def close(self):
        """Tear the cluster down (idempotent; safe even if spawn failed)."""
        self._thread.join()
        client = self._holder.get("client")
        cluster = self._holder.get("cluster")
        if client is not None:
            client.close(timeout=2)
        if cluster is not None:
            cluster.close(timeout=2)
        self._holder.clear()


class CphotAng:
    r"""Cherenkov Photon Angle"""

    def __init__(
        self,
        detector_altitude,
        longitudinal_profile_func="Greisen",
    ):
        r"""CphotAng: Cherenkov photon density and angle determination class.

        Iterative summation of cherenkov radiation reimplemented in numpy and
        C++.
        """
        self.detector_altitude = detector_altitude
        self.dtype = np.float32
        """numerical data type"""

        # fmt: off
        self.wave1 = np.array(
            [
                200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525,
                550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875,
                900,
            ],
            dtype=self.dtype,
        )
        # fmt: on
        """internal wavelength step array"""

        self.OzZeta = np.array(
            [5.35, 10.2, 14.75, 19.15, 23.55, 28.1, 32.8, 37.7, 42.85, 48.25, 100.0],
            dtype=self.dtype,
        )
        """internal Ozone zeta array"""

        self.OzDepth = np.array(
            [15.0, 9.0, 10.0, 31.0, 71.0, 87.2, 57.0, 29.4, 10.9, 3.2, 1.3],
            dtype=self.dtype,
        )
        """internal Ozone depth array"""

        self.OzDsum = np.array(
            [310.0, 301.0, 291.0, 260.0, 189.0, 101.8, 44.8, 15.4, 4.5, 1.3, 0.1],
            dtype=self.dtype,
        )
        """internal Ozone Dsum array"""

        # The ozone column is the path integral of the *piecewise-constant* ozone
        # rate (-dTotZon/dz), so it has an exact analytic layer-walk over the
        # ozone altitude breakpoints -- no GL sub-quadrature, no per-node
        # searchsorted (and exact where 8-point GL was ~5% off at the rate
        # discontinuities). Precompute the 12 altitude boundaries ``[0, *OzZeta]``
        # and the 11 per-segment constant rates (``ozone_rate`` at each segment
        # midpoint; the rate is constant across a segment, so the midpoint is
        # exact). See :func:`ozone_column`.
        self.oz_zbnd = np.concatenate([[0.0], np.asarray(self.OzZeta, np.float64)])
        self.oz_seg_rate = self.ozone_rate(0.5 * (self.oz_zbnd[:-1] + self.oz_zbnd[1:]))

        # fmt: off
        self.aOD55 = np.array(
            [
                0.250, 0.136, 0.086, 0.065, 0.055, 0.049, 0.045, 0.042, 0.038, 0.035,
                0.032, 0.029, 0.026, 0.023, 0.020, 0.017, 0.015, 0.012, 0.010, 0.007,
                0.006, 0.004, 0.003, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001,
            ],
            dtype=self.dtype,
        )
        """internal aOD55 array"""
        # fmt: on

        self.orbit_height = self.dtype(525.0)  # parin(2) orbit height km
        """Detector orbital altitude in Km"""

        # parin(5) record time dispersion at
        self.time_disp_rec_point = self.dtype(0.5)
        """record time dispersion point (km)"""
        # this radial point (km)

        # c  parameters for 1/Beta fit vs wavelength
        # c     5th order polynomial
        # fmt: off
        aP = Polynomial(
            np.array(
                [
                    -1.2971, 0.22046e-01, -0.19505e-04,
                    0.94394e-08, -0.21938e-11, 0.19390e-15,
                ],
                dtype=self.dtype,
            )
        )
        """1/beta fit vs wavelength polynomial"""
        # fmt: on

        self.wmean = self.wave1[:-1] + self.dtype(12.5)

        # Per-wavelength Rayleigh exponent coefficient: the fused transmission
        # exponent is X_to_detector (x) -(400/wmean)^4 / 2974. Precompute the
        # negative per-wavelength factor so the hot path is one outer-product
        # multiply (no per-call pow), matching the ozone/aerosol structure.
        self.RaylCoeff = (
            -((self.dtype(400.0) / self.wmean) ** 4) / self.dtype(2974.0)
        ).astype(self.dtype)

        tBetinv = aP(self.wmean)
        self.aBetaF = np.reciprocal(tBetinv, dtype=self.dtype)
        self.aBetaF /= self.dtype(0.158)
        # self.aBeta55 = self.dtype(0.158)

        # c            Ozone Trans = exp(-kappa dx)
        # c               where dx=ozone slant depth in atm-cm
        # c               and kappa = 110.5 x wave^(-44.21) in atm-cm^-1
        self.Okappa = np.log10(self.wmean, dtype=self.dtype)
        self.Okappa *= self.dtype(44.21)
        self.Okappa = self.dtype(110.5) - self.Okappa
        self.Okappa = np.power(10.0, self.Okappa, dtype=self.dtype)
        self.Okappa *= self.dtype(-1e-3)

        # c
        # c calc OD/km difference
        # c
        # self.dfaOD55 = np.diff(self.aOD55[::-1], append=0)
        tmp = [self.dtype(self.aOD55[i] - self.aOD55[i + 1]) for i in range(29)]
        tmp.append(self.dtype(0))
        self.dfaOD55 = np.array(tmp, dtype=self.dtype)
        # np.append(self.dfaOD55, 0)

        # Aerosol layer-walk table. ``dfaOD55[i]`` is the (vertical) aerosol
        # optical depth across the 1 km band ``[i, i+1]``, i.e. a piecewise-
        # constant extinction per km -- the same structure as the ozone rate. So
        # the *slant* aerosol column has an exact analytic curved-atmosphere walk
        # (see :func:`aerosol_column`), replacing the former plane-parallel
        # ``vertical_OD / cos(theta)`` (which diverges at grazing incidence).
        # 31 altitude boundaries [0, 1, ..., 30] km, 30 per-km extinctions.
        self.aero_zbnd = np.arange(31.0)
        self.aero_ext = np.asarray(self.dfaOD55, dtype=np.float64)

        self.alpha = np.reciprocal(self.dtype(137.04))
        self.pi = self.dtype(3.1415926)

        self.PYieldCoeff = (
            self.dtype(2e12)
            * self.pi
            * self.alpha
            * (np.reciprocal(self.wave1)[:-1] - np.reciprocal(self.wave1)[1:])
        )

        self.zmax = self.orbit_height
        self.zMaxZ = self.dtype(65.0)
        # Highest altitude whose Cherenkov light still reaches the detector: the
        # lesser of the detector altitude (light generated above it is beamed
        # behind the detector and lost) and zMaxZ (above which the air is too
        # rarified to radiate). For an orbital detector the radiation cap binds
        # (z_shower_top == zMaxZ); for a low detector (balloon/mountaintop) the
        # detector altitude binds and truncates the shower's visible tail.
        self.z_shower_top = self.dtype(min(float(detector_altitude), 65.0))
        self.RadE = self.dtype(6371.0)

        # Longitudinal Profile Funciton selection
        if longitudinal_profile_func == "Greisen":
            self.particle_count = greisen_particle_count
        elif longitudinal_profile_func == "Gaisser-Hillas Parameterized":
            self.particle_count = lambda *args, **kwargs: (
                particle_count_parameterized_gaisser_hillas(*args, **kwargs)
            )
        elif longitudinal_profile_func == "Gaisser-Hillas Fluctuated":
            with as_file(
                files("nuspacesim.data.CONEX_table")
                / "dumpGH_conex_pi_E17_95deg_0km_eposlhc_1394249052_211.dat"
            ) as file:
                CONEX_table = np.loadtxt(file, usecols=(4, 5, 6, 7, 8, 9))
                self.particle_count = lambda *args, **kwargs: (
                    particle_count_fluctuated_gaisser_hillas(
                        CONEX_table, *args, **kwargs
                    )
                )

    def ozone_losses(self, z):
        """Calculate ozone losses from altitudes (z in km)."""
        msk1 = z < 5.35
        TotZon = np.empty_like(z)
        TotZon[msk1] = 310 + ((5.35 - z[msk1]) / 5.35) * 15
        msk2 = z >= 100
        TotZon[msk2] = 0.1

        msk3 = ~msk1 & ~msk2
        idxs = np.searchsorted(self.OzZeta, z[msk3])
        TotZon[msk3] = (
            self.OzDsum[idxs]
            + (
                (self.OzZeta[idxs] - z[msk3])
                / (self.OzZeta[idxs] - self.OzZeta[idxs - 1])
            )
            * self.OzDepth[idxs]
        )
        return TotZon

    def ozone_rate(self, z):
        """Analytic ozone-column slope -dTotZon/dz (atm-cm per km), z in km.

        ``ozone_losses`` (TotZon) is piecewise-linear in z, so its derivative
        is piecewise-constant -- the exact within-segment slope using the same
        searchsorted('left') segment convention. Replaces the old central
        finite difference (two ozone_losses evaluations + 2*searchsorted),
        which only smeared the breakpoints.
        """
        # Dense form: one searchsorted over all z + np.where selection, instead
        # of boolean-mask scatter (which gathered/wrote the large (n,k,n_slant_sub)
        # array element-wise). idx is clipped to [1, len-1] so the segment-slope
        # gather is always in-bounds; the values it produces outside [5.35, 100)
        # are discarded by the np.where masks below.
        idx = np.clip(np.searchsorted(self.OzZeta, z), 1, len(self.OzZeta) - 1)
        seg = self.OzDepth[idx] / (self.OzZeta[idx] - self.OzZeta[idx - 1])
        rate = np.where(z < 5.35, 15.0 / 5.35, seg)
        rate[z >= 100.0] = 0.0
        return rate

    def theta_prop(self, z, sinThetView):
        """Theta propagation angle."""
        tp = (self.RadE + self.zmax) / (self.RadE + z)
        return np.arccos(sinThetView * tp)

    def shower_propagation_length(self, decay_altitude, beta, Xtarg, **kwargs):
        """
        Compute shower propagation length.

        Parameters
        ----------
        decay_altitude : float or ndarray
            Initial altitude where shower starts (km)
        beta : float or ndarray
            Propagation angle (radians)
        Xtarg : float
            Target slant depth (g/cm²). Must be scalar (same for all showers).
        **kwargs : dict
            Additional parameters passed to underlying implementation

        Returns
        -------
        L : float or ndarray
            Propagation length (km) where shower reaches Xtarg

        Notes
        -----
        Uses Halley's method + Gauss-Legendre quadrature.
        """
        # Convert to arrays for validation
        decay_altitude_arr = np.atleast_1d(decay_altitude)
        beta_arr = np.atleast_1d(beta)

        if not np.all(np.isfinite(decay_altitude_arr)):
            raise ValueError("decay_altitude must be finite")
        if not np.all(np.isfinite(beta_arr)):
            raise ValueError("beta must be finite")
        Xtarg_arr = np.atleast_1d(Xtarg)
        if not np.all(np.isfinite(Xtarg_arr) & (Xtarg_arr > 0)):
            raise ValueError("Xtarg must be finite and positive")

        return shower_propagation_length(
            decay_altitude, beta, Xtarg, R=float(self.RadE), **kwargs
        )

    def run(
        self,
        betaE,
        alt,
        Eshow100PeV,
        lat,
        long,
        cloudf=None,
        n_nodes=12,
        n_slant_sub=8,
        per_wavelength=True,
        n_energy_low=3,
        n_energy_high=8,
        photon_model=None,
    ):
        """Main simulation: compute photon density and Cherenkov angle.

        Fully vectorized over showers. Accepts scalar or array inputs.

        Array-shape vocabulary::

            N  = n_showers    batch size (the only ragged input dimension)
            k  = n_nodes      GL nodes along each shower axis     (default 16)
            ns = n_slant_sub  GL sub-quad nodes for slant-depth/ozone (default 8)
            W  = n_wl         wavelength bins = len(self.wmean)    (= 28)
            E  = n_E          energy-quadrature nodes in kernel (= nL+nH = 3+8 = 11)
            P  = N * k        flattened (shower, node) rows fed to the kernel

        The two per-shower outputs are (N,); per-node working arrays are
        (N, k); the transient wavelength/energy tensors are (N, k, W) and
        (N, k, E) -- the working-set peak. Everything is float64 until the
        photon-yield/kernel region, which runs float32 (see :meth:_photon_sum).

        Parameters
        ----------
        betaE : float or ndarray
            Earth emergence angle(s) (radians).
        alt : float or ndarray
            Decay altitude(s) (km).
        Eshow100PeV : float or ndarray
            Shower energy(ies) in 100 PeV.
        lat, long : float or ndarray
            Latitude and longitude for cloud model.
        cloudf : callable, optional
            Cloud top height function.
        n_nodes : int, optional
            Number of GL nodes for the longitudinal (slant-depth) shower grid,
            i.e. the node count fed to the kernel per shower (default 16).
        n_slant_sub : int, optional
            GL sub-quadrature nodes for the *slant-depth and ozone column*
            integrals (default 8). Independent of the energy quadrature below.
        per_wavelength : bool, optional
            If True, return photon density per wavelength bin (n_showers, n_wl)
            instead of collapsed (n_showers,). Default True.
        n_energy_low : int, optional
            GL nodes on the low-energy panel ``[eCthres, Eswitch=1 GeV]``
            (default 3). The low panel converges at ~4 nodes; accuracy is
            insensitive to n_energy_low >= 3. Configures the *default* photon
            model only; ignored if ``photon_model`` is supplied.
        n_energy_high : int, optional
            GL nodes on the high-energy panel ``[Eswitch, Eshow]`` (default 8).
            The high panel is the sole accuracy limiter of the energy integral
            (spans ~16 log-decades up to the primary energy). Configures the
            *default* photon model only; ignored if ``photon_model`` is supplied.
        photon_model : callable, optional
            Per-node photon-yield model: a callable mapping a
            :class:`PhotonYieldInputs` bundle to a ``(n_showers, n_nodes)`` array
            of per-node photon contributions. Default (None) uses the
            high-performance Gaisser/Hillas single-integral CPU kernel
            (:func:`hillas_single_integral_model`, configured by
            ``n_energy_low``/``n_energy_high``). Supply a custom callable to swap
            in a different parameterization, function approximation, or physical
            model without changing the surrounding pipeline.

        Returns
        -------
        photonDen : ndarray
            Photon density. Shape (n_showers,) when per_wavelength=False,
            or (n_showers, n_wl) when per_wavelength=True, where
            n_wl = len(self.wmean) = 28.
        Cang : ndarray, shape (n_showers,)
            Cherenkov angle + sigma (degrees).
        """
        # The pipeline is linearized so each value is born just before its
        # consumer and dies just after: detector-geometry pieces are computed at
        # their point of use (viewing angle in the atmosphere phase; distance to
        # max and altitude scaling at the end) rather than bundled up front, and
        # the independent Cherenkov-angle path runs as a short prelude whose only
        # surviving output is the cone area that bridges into photon density.

        # Phase 0: coerce inputs. The per-node photon-yield model defaults to the
        # built-in Gaisser/Hillas single-integral kernel, configured with the
        # energy-panel node counts; a user-supplied model is used as-is.
        if photon_model is None:
            photon_model = hillas_single_integral_model(
                n_energy_low=n_energy_low, n_energy_high=n_energy_high
            )
        betaE, alt, Eshow, lat, long_, gb = self._coerce_inputs(
            betaE, alt, Eshow100PeV, lat, long
        )
        R = float(self.RadE)
        atm = Atmosphere()

        # Phase 1: slant-depth grid. Nodes span exactly the valid window in X, so
        # they are all physically valid (no downstream masking). z_peak is the
        # shower-maximum altitude, the only node the detector geometry needs. The
        # per-event cloud top (if any) enters here as the window's lower bound:
        # light from below the clouds is obscured, so no node is placed there.
        cloud_top = (
            np.array([cloudf(la, lo) for la, lo in zip(lat, long_)])
            if cloudf is not None
            else None
        )
        L_max, X_top, X_nodes, L_nodes, L_weights, z_nodes, z_peak = (
            self._propagation_grid(
                alt, betaE, Eshow, gb, n_nodes, n_slant_sub, atm, R, cloud_top
            )
        )

        # Phase 2: atmosphere. The slant-depth, ozone and aerosol columns are
        # turned into transmission and immediately discarded. (The aerosol column
        # is now a curved-atmosphere layer-walk computed from the path geometry
        # in _atmospheric_columns, so the former per-node viewing angle is gone.)
        X_to_node, X_to_detector, ZonZ, AODepth = self._atmospheric_columns(
            X_nodes, X_top, L_nodes, L_max, betaE, R
        )
        transmission = self._atmospheric_transmission(
            X_to_detector, ZonZ, AODepth, per_wavelength
        )

        # Phase 3: shower-physics fields and the derived Cherenkov quantities.
        AirN, s, RN, e2hill = node_fields(z_nodes, X_to_node, gb)
        eCthres, thetaC = cherenkov_threshold_angle(AirN)
        E0 = e0(z_nodes.shape, s)
        Tfrac = tracklen(E0, eCthres, s)

        # Phase 4: photon yield = local Cherenkov scaling x transmission. Clouds
        # need no masking here: the cloud top entered Phase 1 as the window's
        # lower bound, so no node sits below the clouds to begin with.
        sig_per_node, wl_state = self._photon_yield(
            thetaC, L_weights, RN, transmission, per_wavelength
        )

        # Phase 5: Cherenkov-angle prelude. Independent of the kernel; run first
        # so its temporaries (Tfrac, z_peak, DistStep_max) die before the heavy
        # photon sum. Its only output onto the trunk is CherArea (the cone area
        # bridging the angle into the photon density).
        AveCangI, CangsigI, shower_valid = self._cherenkov_angle_stats(
            sig_per_node, Tfrac, thetaC
        )
        Cang = np.degrees(AveCangI + CangsigI)
        DistStep_max = self._distance_to_max(betaE, z_peak)
        CherArea = cherenkov_area(AveCangI, DistStep_max)

        # Phase 6: photon-density trunk -- the single-integral kernel, then final
        # assembly. altitude_scaling = f(betaE, alt) is the literal last multiply.
        node_contribs, photsum = self._photon_sum(
            eCthres,
            Eshow,
            thetaC,
            sig_per_node,
            e2hill,
            E0,
            s,
            n_nodes,
            photon_model,
        )
        altitude_scaling = self._altitude_scaling(betaE, alt)
        photonDen = self._photon_density(
            photsum,
            node_contribs,
            wl_state,
            CherArea,
            shower_valid,
            altitude_scaling,
            per_wavelength,
        )
        return photonDen.astype(self.dtype), Cang.astype(self.dtype)

    # ------------------------------------------------------------------
    # run() stages (each operates on (n_showers,) / (n_showers, n_nodes))
    # ------------------------------------------------------------------

    def _coerce_inputs(self, betaE, alt, Eshow100PeV, lat, long):
        """Coerce raw inputs to (n_showers,) float64 arrays + derived fields.

        Returns ``betaE`` (clamped to >= 1 deg), ``alt``, ``Eshow`` (GeV),
        ``lat``, ``long_``, and ``gb`` (Greisen beta = ln(Eshow / E_crit)).

        Shapes: scalars or arrays in; all six outputs are ``(N,)`` float64
        (scalars are promoted via ``np.atleast_1d``).
        """
        betaE = np.maximum(
            np.atleast_1d(np.asarray(betaE, dtype=np.float64)), np.radians(1.0)
        )
        alt = np.atleast_1d(np.asarray(alt, dtype=np.float64))
        Eshow = np.atleast_1d(np.asarray(Eshow100PeV, dtype=np.float64)) * 1e8  # GeV
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        long_ = np.atleast_1d(np.asarray(long, dtype=np.float64))
        ecrit = 0.710 / (7.4 + 0.96)
        gb = np.log(Eshow / ecrit)
        return betaE, alt, Eshow, lat, long_, gb

    def _propagation_grid(
        self, alt, betaE, Eshow, gb, n_nodes, n_slant_sub, atm, R, cloud_top=None
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
        L_max = lexpr(float(self.z_shower_top), betaE, R=R)
        X_lo, X_hi, X_top = self._valid_window(
            L_start, L_max, betaE, gb, Eshow, n_slant_sub, atm, R, cloud_top
        )
        X_nodes, wX, X_peak = self._node_grid(X_lo, X_hi, gb, n_nodes)
        L_nodes, L_weights, z_nodes, z_peak = self._node_geometry(
            L_start, X_nodes, X_peak, wX, betaE, atm, R
        )
        return L_max, X_top, X_nodes, L_nodes, L_weights, z_nodes, z_peak

    def _valid_window(
        self, L_start, L_max, betaE, gb, Eshow, n_slant_sub, atm, R, cloud_top=None
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
        Xtarg = slant_depth_of_greisen_particle_count(1.0, Eshow)
        X_lo = _X_LO_COEFF * np.atleast_1d(gb)
        if cloud_top is not None:
            # Depth from decay to where the shower rises above the cloud top.
            # Clamp the cloud length to >= L_start so a decay already above the
            # clouds contributes no extra lower bound (X_cloud = 0).
            L_cloud = np.maximum(lexpr(cloud_top, betaE, R=R), L_start)
            X_cloud = slant_depth(L_start, L_cloud, betaE, R=R, a=atm, n=n_slant_sub)
            X_lo = np.maximum(X_lo, X_cloud)
        X_hi = np.maximum(np.minimum(Xtarg, X_top), X_lo)
        return X_lo, X_hi, X_top

    def _node_grid(self, X_lo, X_hi, gb, n_nodes):
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
        X_peak = np.clip(36.66 * np.atleast_1d(gb), X_lo, X_hi)
        n1 = n_nodes // 2
        ref_x1, ref_w1 = _cached_leggauss(n1)
        ref_x2, ref_w2 = _cached_leggauss(n_nodes - n1)
        h1 = 0.5 * (X_peak - X_lo)
        h2 = 0.5 * (X_hi - X_peak)
        Xn1 = (0.5 * (X_peak + X_lo))[:, None] + h1[:, None] * ref_x1
        Xn2 = (0.5 * (X_hi + X_peak))[:, None] + h2[:, None] * ref_x2
        X_nodes = np.concatenate([Xn1, Xn2], axis=1)  # (n_showers, n_nodes)
        wX = np.concatenate([h1[:, None] * ref_w1, h2[:, None] * ref_w2], axis=1)
        return X_nodes, wX, X_peak

    def _node_geometry(self, L_start, X_nodes, X_peak, wX, betaE, atm, R):
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
            L_start, targets, betaE, float(self.z_shower_top), R=R, a=atm
        )
        L_nodes = L_all[:, :-1]
        z_nodes = zexpr(L_nodes, betaE[:, None], R=R)
        z_peak = zexpr(L_all[:, -1], betaE, R=R)
        L_weights = wX / dXl(L_nodes, betaE[:, None], a=atm, R=R)
        return L_nodes, L_weights, z_nodes, z_peak

    def _atmospheric_columns(self, X_nodes, X_top, L_nodes, L_max, betaE, R):
        """Slant depth (to node, to detector), ozone and aerosol columns per node.

        ``X_nodes`` is already the exact slant depth from the decay point to each
        node (the GL-in-X grid placed it there), so the to-node column needs no
        re-integration; the to-detector column is the complementary depth. The
        ozone and aerosol columns are both exact analytic curved-atmosphere
        layer-walks (see :func:`ozone_column`, :func:`aerosol_column`) -- the
        aerosol in particular needs the path geometry here, which is why it has
        moved out of :meth:`_atmospheric_transmission`.

        Shapes: ``X_nodes, L_nodes`` are ``(N, k)``; ``X_top, L_max, betaE`` are
        ``(N,)``. Returns ``X_to_node (N, k)``, ``X_to_detector (N, k)``,
        ``ZonZ (N, k)`` (ozone column), ``AODepth (N, k)`` (aerosol column).
        """
        X_to_node = X_nodes
        X_to_detector = np.maximum(X_top[:, None] - X_nodes, 0.0)
        ZonZ = ozone_column(L_nodes, L_max, betaE, self.oz_zbnd, self.oz_seg_rate, R)
        AODepth = aerosol_column(
            L_nodes, L_max, betaE, self.aero_zbnd, self.aero_ext, R
        )
        return X_to_node, X_to_detector, ZonZ, AODepth

    def _atmospheric_transmission(self, X_to_detector, ZonZ, AODepth, per_wavelength):
        """Atmospheric transmission of Cherenkov light from each node to detector.

        Shapes: the three column inputs (``X_to_detector, ZonZ, AODepth``) are
        ``(N, k)``. The fused exponent tensor is built once at ``(N, k, W)``.

        Consumes the slant-depth, ozone and aerosol columns here (the atmospheric
        phase) so they do not outlive it. Always returns the wavelength-collapsed
        per-node transmission ``T_sum`` of shape ``(n_showers, n_nodes)`` -- which
        is all the kernel and the Cherenkov-angle path need, since
        ``per_wavelength`` only re-decomposes the FINAL photon density (see
        :meth:`run`). When ``per_wavelength`` it additionally returns the raw
        per-wavelength transmission tensor ``exp(...)`` of shape
        ``(n_showers, n_nodes, n_wl)`` (without ``PYieldCoeff``), carried to
        :meth:`_photon_density` where the wavelength-resolved combination is
        deferred to a single contraction.

        The exponent is the sum of three outer products (Rayleigh, ozone,
        aerosol), each a per-node ``(n, k)`` coefficient times a per-wavelength
        vector. They are summed as one ``(n*k, 3) @ (3, n_wl)`` BLAS sgemm (~3x
        faster than three broadcast multiply-adds), in float32 (transmission
        budget ~0.3% >> f32 eps) for ~2x exp throughput / half bandwidth.
        """
        X_to_detector = X_to_detector.astype(self.dtype, copy=False)
        ZonZ = ZonZ.astype(self.dtype, copy=False)
        # Aerosol exponent coefficient = -(slant optical depth). The curved-
        # atmosphere column was already integrated in _atmospheric_columns.
        aero_nk = (-AODepth).astype(self.dtype)
        coeff_nk = np.stack([X_to_detector, ZonZ, aero_nk], axis=-1)  # (n, k, 3)
        wl_matrix = np.stack([self.RaylCoeff, self.Okappa, self.aBetaF])  # (3, n_wl)
        wl_trans = coeff_nk @ wl_matrix  # (n, k, n_wl)
        np.exp(wl_trans, out=wl_trans)  # raw per-wavelength transmission exp(...)
        # Collapse over wavelength via gemv: exp(...) @ PYieldCoeff folds the
        # per-wavelength yield scaling and the wavelength sum into one reduction.
        T_sum = wl_trans @ self.PYieldCoeff  # (n, k)
        if per_wavelength:
            return T_sum, wl_trans
        return T_sum

    def _photon_yield(self, thetaC, L_weights, RN, transmission, per_wavelength):
        """Cherenkov photon yield per node = local Cherenkov scaling x transmission.

        ``transmission`` is the collapsed ``T_sum`` from _atmospheric_transmission
        (or ``(T_sum, wl_trans)`` when ``per_wavelength``). The yield ``sig_per_node``
        = ``sin^2(thetaC) * L_weights * RN * T_sum`` is the SAME in both modes (the
        kernel and angle path only ever see this collapsed yield). Returns
        ``(sig_per_node, wl_state)``; ``wl_state`` is ``None`` (collapsed) or the
        ``(wl_trans, T_sum)`` pair the per-wavelength decomposition needs.

        Shapes: ``thetaC, L_weights, RN`` are ``(N, k)``; ``transmission`` is
        ``T_sum (N, k)`` (collapsed) or ``(T_sum (N, k), wl_trans (N, k, W))``.
        Returns ``sig_per_node (N, k)`` and ``wl_state`` = ``None`` or
        ``(wl_trans (N, k, W), T_sum (N, k))``.
        """
        # Build the local Cherenkov scaling in place: sin^2(thetaC) * L_weights * RN.
        scaling = np.sin(thetaC)  # (n_showers, n_nodes)
        scaling *= scaling
        scaling *= L_weights
        scaling *= RN
        if per_wavelength:
            T_sum, wl_trans = transmission
            # Cast to T_sum's dtype so sig_per_node is bit-identical to the
            # collapsed path's in-place ``T_sum *= scaling`` -> the kernel and
            # Cherenkov angle are then identical in both modes.
            sig_per_node = (scaling * T_sum).astype(T_sum.dtype, copy=False)
            return sig_per_node, (wl_trans, T_sum)
        # transmission is dead after this; reuse its buffer for sig_per_node.
        transmission *= scaling
        return transmission, None

    def _photon_sum(
        self,
        eCthres,
        Eshow,
        thetaC,
        sig_per_node,
        e2hill,
        E0,
        s,
        n_nodes,
        photon_model,
    ):
        """Apply the photon-yield model per node, then reduce over nodes.

        ``photon_model`` is any callable mapping a :class:`PhotonYieldInputs`
        bundle to ``node_contribs`` of shape ``(n_showers, n_nodes)`` -- by
        default the Gaisser/Hillas single-integral kernel
        (:func:`hillas_single_integral_model`), but a user may supply an
        alternative parameterization, function approximation, or physical model.
        This stage owns only the input marshalling and the node-axis reduction;
        all model-specific work (energy quadrature, precision, etc.) lives in the
        model itself.

        Shapes: ``eCthres, thetaC, sig_per_node, e2hill, E0, s`` are ``(N, k)``;
        ``Eshow`` is ``(N,)``. Returns ``node_contribs (N, k)`` and
        ``photsum (N,)`` (= sum over nodes).
        """
        inputs = PhotonYieldInputs(
            eCthres=eCthres,
            thetaC=thetaC,
            sig_per_node=sig_per_node,
            e2hill=e2hill,
            E0=E0,
            s=s,
            Eshow=Eshow,
            n_nodes=n_nodes,
        )
        node_contribs = photon_model(inputs)
        return node_contribs, np.sum(node_contribs, axis=1)

    def _cherenkov_angle_stats(self, sig_per_node, Tfrac, thetaC):
        """Photon-weighted mean Cherenkov angle and its spread (mean + 1 sigma).

        ``Cang`` is the photon-weighted mean angle plus one standard deviation,
        an effective cone half-angle. It is built here from three raw moments of
        the per-node angle, weighted by the photon yield
        ``w = sig_per_node * Tfrac``::

            M0 = sum_k w_k ,  M1 = sum_k w_k theta_k ,  M2 = sum_k w_k theta_k^2
            mean = M1 / M0
            var  = max(M2 / M0 - mean^2, 0)        # population variance
            AveCangI = mean ,  CangsigI = sqrt(var)

        This is a single fused pass over the node axis (no second pass about the
        mean) and uses the *population* variance -- there is deliberately no
        ``n/(n-1)`` Bessel correction: the longitudinal nodes are deterministic
        Gauss-Legendre abscissae of a continuous integral, not i.i.d. samples, so
        a sample-variance correction would make ``Cang`` depend explicitly on the
        node count. Expressing the spread as three *linear* functionals of the
        same smooth integrand family (w, w*theta, w*theta^2) keeps ``Cang`` well
        conditioned as the node count is reduced (it tracks the photon-yield
        quadrature instead of diverging from it). The ``max(.,0)`` guards the
        float32 rounding floor of the one-pass ``M2/M0 - mean^2`` form (verified
        to agree with the two-pass form to <1.2e-4 deg).

        Returns ``(AveCangI, CangsigI, shower_valid)``. The cone area is a
        separate concern (it bridges into photon density) -- see
        :func:`cherenkov_area`.

        Shapes: ``sig_per_node, Tfrac, thetaC`` are ``(N, k)`` (reduced over the
        k nodes); returns ``AveCangI (N,)``, ``CangsigI (N,)``, and
        ``shower_valid (N,)`` bool.
        """
        w = sig_per_node * Tfrac  # (n_showers, n_nodes) photon-yield weight
        M0 = np.sum(w, axis=-1)  # (n_showers,)
        shower_valid = M0 > 0
        safe_M0 = np.where(shower_valid, M0, 1.0)
        mean = np.sum(w * thetaC, axis=-1) / safe_M0
        var = np.sum(w * thetaC * thetaC, axis=-1) / safe_M0 - mean * mean
        np.maximum(var, 0.0, out=var)
        AveCangI = np.where(shower_valid, mean, 0.0)
        CangsigI = np.where(shower_valid, np.sqrt(var), 0.0)
        return AveCangI, CangsigI, shower_valid

    def _distance_to_max(self, betaE, z_peak):
        """Detector distance to the shower maximum (the Greisen peak altitude).

        One value per shower: the cone area only needs the brightest point, not
        all nodes. Computed at its point of use (the ground-area bridge) so it
        does not outlive the kernel.

        Shapes: ``betaE, z_peak`` are ``(N,)``; returns ``(N,)``.
        """
        sin_view = (self.RadE / (self.RadE + self.zmax)) * np.cos(betaE)
        ThetView = np.arcsin(sin_view)
        ThetPrpA_peak = self.theta_prop(z_peak, sin_view)
        return d_to_det(ThetView, ThetPrpA_peak, z_peak, self.RadE)

    def _altitude_scaling(self, betaE, alt):
        """(distance to orbit / distance to detector)^2 brightness scaling.

        Shapes: ``betaE, alt`` are ``(N,)``; returns ``(N,)``.
        """
        theta_prop_alt = propagation_angle(betaE, alt, Re=self.RadE)

        theta_view_orbit = viewing_angle(betaE, self.orbit_height, self.RadE)
        ang_e_orbit = np.pi / 2 - theta_view_orbit - theta_prop_alt
        dist_orbit = np.sin(ang_e_orbit) / np.sin(theta_view_orbit) * (alt + self.RadE)

        theta_view_det = viewing_angle(betaE, self.detector_altitude, self.RadE)
        ang_e_det = np.pi / 2 - theta_view_det - theta_prop_alt
        dist_det = np.sin(ang_e_det) / np.sin(theta_view_det) * (alt + self.RadE)

        return (dist_orbit / dist_det) ** 2

    def _photon_density(
        self,
        photsum,
        node_contribs,
        wl_state,
        CherArea,
        shower_valid,
        altitude_scaling,
        per_wavelength,
    ):
        """Assemble photon density (collapsed or per-wavelength), detector-scaled.

        Shapes: ``photsum, CherArea, shower_valid, altitude_scaling`` are ``(N,)``;
        ``node_contribs`` is ``(N, k)``; ``wl_state`` (per-wavelength only) is
        ``(wl_trans (N, k, W), T_sum (N, k))``. Returns ``photonDen`` of shape
        ``(N,)`` (collapsed) or ``(N, W)`` (per-wavelength), contracting away k.
        """
        ok = shower_valid & (CherArea > 0)
        if per_wavelength:
            # Deferred wavelength decomposition (the only place the (n,k,n_wl)
            # tensor is consumed). The per-node Cherenkov scaling cancels in the
            # wavelength fraction (SPYield/sig = wl_trans / T_sum), so
            #   photsum_wl[n,w] = PYieldCoeff[w] * sum_k (node_contribs/T_sum) wl_trans
            # is one contraction over the K nodes -> (n_showers, n_wl). PYieldCoeff
            # is applied to the (n, w) result, never to the rank-3 tensor.
            wl_trans, T_sum = wl_state
            g = node_contribs / np.where(T_sum > 0, T_sum, 1.0)  # (n, k)
            photsum_wl = np.einsum("nk,nkw->nw", g, wl_trans)  # (n, n_wl)
            photsum_wl *= self.PYieldCoeff
            photonDen = np.where(ok[:, None], 0.5 * photsum_wl / CherArea[:, None], 0.0)
            return photonDen * altitude_scaling[:, None]

        photonDen = np.where(ok, 0.5 * photsum / CherArea, 0.0)
        return photonDen * altitude_scaling

    def __call__(
        self,
        betaE,
        alt,
        Eshow100PeV,
        init_lat,
        init_long,
        cloudf=None,
        chunks=None,
        photon_model=None,
        per_wavelength=False,
        client=None,
    ):
        """
        Iterate over the list of events and return the result as pair of
        numpy arrays.

        ``chunks`` is the dask block size along the event axis. ``None``
        (default) auto-sizes it via :func:`_auto_chunk_size` from the usable
        core count, targeting good load balance within the empirically-flat
        optimal band (~8k-16k events/block). A blocked (non-``"auto"``) size
        is essential: the inputs are small, so dask ``"auto"`` makes a single
        block, which both serializes the work and forces huge
        ``(n, n_nodes, n_E)`` temporaries that thrash memory bandwidth.
        Blocking lets the distributed scheduler fan out across worker
        processes. Pass an int to override. Note: run() derives its
        energy-quadrature bounds per block, so per-event photon densities
        depend slightly (~1e-2, within batch tolerance) on the block size.

        ``photon_model`` is forwarded to :meth:`run` (None uses the default
        Gaisser/Hillas single-integral kernel); see :class:`PhotonYieldInputs`.

        ``per_wavelength`` is forwarded to :meth:`run`. When ``False`` (default)
        the returned photon density is a per-shower scalar ``(N,)`` -- the
        contract ``eas.py`` consumes. When ``True`` the density keeps its
        wavelength axis, ``(N, n_wl)`` with ``n_wl = len(self.wmean) = 28``; the
        Cherenkov angle ``Cang`` stays ``(N,)`` in both cases.

        ``client`` is an optional pre-built distributed :class:`Client`. The
        simulation runs exactly one ``__call__``, so the orchestrator can spin a
        :class:`BackgroundCluster` up early (overlapping the ~2s process spawn
        with the earlier pipeline stages) and pass its client in here; the
        caller then owns teardown. When ``None`` (e.g. direct/standalone use)
        this method spins up and tears down its own ``LocalCluster``.
        """

        if (
            len(betaE) < 1
            or len(alt) < 1
            or len(Eshow100PeV) < 1
            or len(init_lat) < 1
            or len(init_long) < 1
        ):
            return np.empty([]), np.empty([])

        if chunks is None:
            chunks = _auto_chunk_size(len(betaE))

        #######################
        args = [betaE, alt, Eshow100PeV, init_lat, init_long]
        arx = [np.asarray(x) for x in args]
        owns_cluster = client is None
        cluster = None
        if owns_cluster:
            cluster = LocalCluster(processes=True)
            client = Client(cluster)
        d_args = [da.from_array(a, chunks=chunks) for a in arx]

        # Per block, run() yields density (N,) [collapsed] or (N, n_wl)
        # [per-wavelength] plus Cang (N,). We pack both into a single
        # (rows, N) block so one map_blocks carries them out: the first
        # ``n_den`` rows are the density (n_den = 1 collapsed, n_wl per-
        # wavelength), the last row is Cang. The collapsed path stays
        # (2, N) -- bit-identical to before; per-wavelength avoids the
        # (N, n_nodes, n_wl) tensor only when the caller actually wants it.
        n_wl = len(self.wmean)
        n_den = n_wl if per_wavelength else 1
        n_rows = n_den + 1

        def chunk_worker(b, a, e, lat, lon):
            d_batch, c_batch = self.run(
                b,
                a,
                e,
                lat,
                lon,
                cloudf=cloudf,
                per_wavelength=per_wavelength,
                photon_model=photon_model,
            )
            # d_batch is (N,) collapsed or (N, n_wl); make it (n_den, N).
            d_rows = d_batch.T if per_wavelength else d_batch[None, :]
            return np.concatenate([d_rows, c_batch[None, :]], axis=0)

        # Apply vectorization via map_blocks. output will be 2D (n_rows, N)
        result_grid = da.map_blocks(
            chunk_worker,
            *d_args,
            dtype=float,
            chunks=(n_rows, d_args[0].chunks[0]),
            new_axis=0,
        )

        n_chunks = result_grid.npartitions
        from rich.progress import BarColumn, ProgressColumn, TextColumn
        from rich.text import Text

        class _ElapsedSecondsColumn(ProgressColumn):
            def render(self, task):
                elapsed = (
                    task.finished_time
                    if task.finished_time is not None
                    else task.elapsed
                )
                return Text(f"{elapsed:.2f}s")

        # dask.diagnostics.Callback only works with synchronous schedulers.
        # With the distributed scheduler, persist the dask graph so each array
        # chunk has a Future, then drive the Rich progress bar incrementally via
        # as_completed() as worker processes finish their chunks.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Sending large graph")
            persisted = client.persist(result_grid)

        futures = client.futures_of(persisted)

        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            _ElapsedSecondsColumn(),
        ) as progress:
            progress_task = progress.add_task(
                "Processing EAS photons...", total=n_chunks
            )
            for _ in as_completed(futures):
                progress.advance(progress_task, 1)

        # Gather result from workers (all futures already done by this point).
        results = persisted.compute()

        # Shut down workers quickly: close synchronously to avoid orphaned worker heartbeats.
        if owns_cluster:
            client.close(timeout=2)
            cluster.close(timeout=2)

        # Unpack (n_rows, N): density rows then the Cang row. Collapsed ->
        # (N,); per-wavelength -> (N, n_wl) (transpose back the n_den rows).
        dphots = results[0] if n_den == 1 else results[:n_den].T
        return dphots, results[n_den]
