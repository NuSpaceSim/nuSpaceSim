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

"""Distributed shower dispatch: chunk a batch kernel across a dask cluster.

Pure execution machinery -- chunk sizing, cluster lifecycle, ``map_blocks``
fan-out, and the live progress bar. It knows nothing about Cherenkov physics:
the physics enters only through the ``chunk_fn`` callable handed to
:func:`map_showers_distributed` (see ``CphotAng.__call__``).
"""

import os
import warnings

import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster, as_completed
from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn
from rich.text import Text

__all__ = ["map_showers_distributed", "auto_chunk_size"]


# Dask block size (events/block). Throughput is flat over a wide plateau of
# ~16k-50k events/block and falls off sharply BELOW it: now that run() costs
# ~0.012 ms/shower, the per-block fixed cost (dask task scheduling,
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


def auto_chunk_size(n_events):
    """Parallelism-aware dask block size for :func:`map_showers_distributed`.

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


class _ElapsedSecondsColumn(ProgressColumn):
    def render(self, task):
        elapsed = task.finished_time if task.finished_time is not None else task.elapsed
        return Text(f"{elapsed:.2f}s")


def map_showers_distributed(
    chunk_fn,
    arrays,
    n_rows,
    chunks=None,
    client=None,
    description="Processing EAS photons...",
):
    """Fan a packed batch kernel out over a process-based dask cluster.

    ``chunk_fn(*blocks) -> (n_rows, N) ndarray`` is applied per chunk via
    ``da.map_blocks`` over ``arrays`` (equal-length 1-D inputs, chunked along
    the event axis). Results are gathered as one ``(n_rows, N_total)`` array.

    ``chunks`` is the block size along the event axis; ``None`` auto-sizes via
    :func:`auto_chunk_size`. A blocked (non-``"auto"``) size is essential: the
    inputs are small, so dask ``"auto"`` makes a single block, which both
    serializes the work and forces huge per-block temporaries.

    ``client`` is an optional pre-built distributed :class:`Client` (see
    ``utils.distributed.BackgroundCluster``); the caller then owns teardown.
    When ``None`` this function spins up and tears down its own
    ``LocalCluster``.

    A Rich progress bar advances as worker processes complete chunks
    (``dask.diagnostics.Callback`` only works with synchronous schedulers, so
    the graph is persisted and its futures drive the bar via
    ``as_completed``).
    """
    arx = [np.asarray(x) for x in arrays]
    if chunks is None:
        chunks = auto_chunk_size(len(arx[0]))

    owns_cluster = client is None
    cluster = None
    if owns_cluster:
        cluster = LocalCluster(processes=True)
        client = Client(cluster)
    d_args = [da.from_array(a, chunks=chunks) for a in arx]

    result_grid = da.map_blocks(
        chunk_fn,
        *d_args,
        dtype=float,
        chunks=(n_rows, d_args[0].chunks[0]),
        new_axis=0,
    )

    n_chunks = result_grid.npartitions

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Sending large graph")
        persisted = client.persist(result_grid)

    futures = client.futures_of(persisted)

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        _ElapsedSecondsColumn(),
    ) as progress:
        progress_task = progress.add_task(description, total=n_chunks)
        for _ in as_completed(futures):
            progress.advance(progress_task, 1)

    # Gather result from workers (all futures already done by this point).
    results = persisted.compute()

    # Shut down workers quickly: close synchronously to avoid orphaned worker
    # heartbeats.
    if owns_cluster:
        client.close(timeout=2)
        cluster.close(timeout=2)

    return results
