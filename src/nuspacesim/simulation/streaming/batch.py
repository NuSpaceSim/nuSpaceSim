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

"""One streaming batch: throw -> physics -> per-event contribution -> sketch.

:func:`run_batch` is the unit of work the driver ships to a dask worker. It runs
the same stage sequence as :func:`nuspacesim.compute.compute` on a small batch
of thrown trajectories and returns only the two small, mergeable summaries --
a :class:`~.sketch.MomentSketch` (a few scalars) and a
:class:`~.reservoir.WeightedReservoir` (<= k rows). The full per-event arrays
never leave the worker.

Heavy engine objects (geometry, spectra, taus, the optical kernel) are built
once per worker process and cached; a streaming run uses a single config, so the
cache holds one entry for the worker's lifetime.
"""

from __future__ import annotations

import numpy as np

from .reservoir import WeightedReservoir
from .sketch import MomentSketch

__all__ = ["run_batch"]

# Per-process engine cache. A worker handles many batches of one config; build
# the engines (which load atmosphere/kernel tables) once and reuse them.
_ENGINE_CACHE: dict = {}


def _get_engines(config):
    """Build (and cache) the per-worker simulation engines for ``config``."""
    key = config.model_dump_json()
    engines = _ENGINE_CACHE.get(key)
    if engines is None:
        from ..atmosphere.clouds import CloudTopHeight
        from ..eas_optical.eas import EAS
        from ..geometry.region_geometry import RegionGeom
        from ..spectra.spectra import Spectra
        from ..taus.taus import Taus

        engines = {
            "geom": RegionGeom(config),
            "cloud": CloudTopHeight(config),
            "spec": Spectra(config),
            "tau": Taus(config),
            "eas": EAS(config),
        }
        _ENGINE_CACHE.clear()  # only ever one config per run; bound the cache
        _ENGINE_CACHE[key] = engines
    return engines


def _seed_batch_rng(root_entropy: int, batch_idx: int) -> np.random.Generator:
    """Seed this batch's RNG state deterministically from ``(root, batch_idx)``.

    Two independent child streams are derived: one seeds the legacy global
    ``np.random`` state used by the physics stages (geometry throw, power
    spectrum, tau sampling, decay altitude), the other is returned as an
    explicit :class:`~numpy.random.Generator` for the reservoir sampling keys.
    Because the seed depends only on ``batch_idx`` (not on worker or schedule),
    batch *i* always produces the same events and keys -- the materialized
    reservoir is reproducible even though the float-sum sketch may reorder.
    """
    seq = np.random.SeedSequence(entropy=root_entropy, spawn_key=(batch_idx,))
    phys_seq, key_seq = seq.spawn(2)
    np.random.seed(int(phys_seq.generate_state(1)[0]))
    return np.random.default_rng(key_seq)


# Per-event fields materialized into the reservoir (optical diagnostics +
# provenance). Mirrors the columns the one-shot path stores, minus radio-only.
_RESERVOIR_FIELDS = (
    "beta_rad",
    "theta",
    "path_len",
    "altDec",
    "lenDec",
    "showerEnergy",
    "numPEs",
    "costhetaChEff",
    "mcintfactor",
    "init_lat",
    "init_lon",
)


def run_batch(
    config,
    batch_idx: int,
    batch_size: int,
    reservoir_size: int,
    weighting: str,
    root_entropy: int,
):
    """Throw and process one batch; return ``(MomentSketch, WeightedReservoir)``.

    Parameters
    ----------
    config : NssConfig
        Full simulation configuration (shipped from the driver).
    batch_idx : int
        Index of this batch; seeds the deterministic per-batch RNG.
    batch_size : int
        Number of trajectories to throw (the integral denominator contribution).
    reservoir_size : int
        ``k`` for the weighted reservoir.
    weighting : {"contribution", "uniform"}
        Reservoir sampling weight: ``|mcintfactor|`` (importance) or uniform.
    root_entropy : int
        Run-level RNG entropy; combined with ``batch_idx`` for reproducibility.
    """
    from ..geometry.region_geometry import mc_contribution

    key_rng = _seed_batch_rng(root_entropy, batch_idx)
    eng = _get_engines(config)
    geom, cloud, spec, tau, eas = (
        eng["geom"],
        eng["cloud"],
        eng["spec"],
        eng["tau"],
        eng["eas"],
    )

    geom.throw(batch_size)
    n_thrown = len(geom.betaTrSubN)
    beta_tr = geom.beta_rad()
    n_valid = beta_tr.size

    mcnorm = geom.mcnorm

    # No valid geometry this batch: still counts toward the denominator.
    if n_valid == 0:
        return (
            MomentSketch(mcnorm=mcnorm, n_thrown=n_thrown),
            WeightedReservoir.empty(reservoir_size),
        )

    init_lat, init_long = geom.find_lat_long_along_traj(np.zeros_like(beta_tr))
    theta = geom.thetas()
    path_len = geom.pathLens()

    log_e_nu, spec_norm, spec_weights_sum = spec(n_valid)
    tauBeta, tauLorentz, _tauEnergy, showerEnergy, tauExitProb = tau(beta_tr, log_e_nu)
    altDec, lenDec = eas.altDec(beta_tr, tauBeta, tauLorentz)

    numPEs, costhetaChEff = eas(
        beta_tr,
        altDec,
        showerEnergy,
        init_lat,
        init_long,
        cloudf=cloud,
        serial=True,
    )

    mcintfactor, geo = mc_contribution(
        geom.valid_costhetaTrSubN(),
        geom.valid_costhetaNSubV(),
        geom.valid_costhetaTrSubV(),
        numPEs,
        costhetaChEff,
        tauExitProb,
        config.detector.optical.photo_electron_threshold,
        spec_norm,
        spec_weights_sum,
    )

    sketch = MomentSketch.from_batch(mcintfactor, geo, n_thrown, mcnorm)

    weights = (
        np.ones_like(mcintfactor) if weighting == "uniform" else np.abs(mcintfactor)
    )
    records = dict(
        zip(
            _RESERVOIR_FIELDS,
            (
                beta_tr,
                theta,
                path_len,
                altDec,
                lenDec,
                showerEnergy,
                numPEs,
                costhetaChEff,
                mcintfactor,
                init_lat,
                init_long,
            ),
        )
    )
    reservoir = WeightedReservoir.from_batch(reservoir_size, records, weights, key_rng)
    return sketch, reservoir
