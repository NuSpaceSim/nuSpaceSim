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

"""Streaming Monte-Carlo driver: batch -> merge -> live stats -> stop.

:func:`compute_streaming` is the streaming counterpart to
:func:`nuspacesim.compute.compute` for the optical channel. It throws events in
batches across a process-based dask cluster, merges each batch's
:class:`~.sketch.MomentSketch` and :class:`~.reservoir.WeightedReservoir` into
running totals, shows an always-updating ``rich`` panel, and stops when the MC
integral's relative uncertainty reaches the configured target (with thrown-count
and wall-clock backstops). Only the weighted reservoir of representative events
is materialized into the returned table.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
import warnings

import numpy as np
from dask.distributed import as_completed
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ... import results_table
from ...utils.distributed import BackgroundCluster
from ..geometry.region_geometry import RegionGeom
from .batch import run_batch
from .reservoir import WeightedReservoir
from .sketch import MomentSketch

__all__ = ["compute_streaming"]

_META_COMMENTS = {
    "OMCINT": "Optical MonteCarlo Integral",
    "OMCINTGO": "Optical MonteCarlo Integral, GEO Only",
    "ONEVPASS": "Optical Number of Passing Events",
    "OMCINTUN": "Stat unc of MonteCarlo Integral",
}


def _silence_worker_logs():
    """Raise dask worker-process log levels to CRITICAL (runs on each worker).

    Shipped via ``client.run`` just before teardown to quiet the benign
    heartbeat / comm-closed ERRORs a worker emits while the cluster shuts down.
    Called only after the computation is complete -- real task failures have
    already surfaced in the driver via ``future.result()``.
    """
    import logging as _logging

    for _name in (
        "distributed.worker",
        "distributed.nanny",
        "distributed.core",
        "distributed.comm",
    ):
        _logging.getLogger(_name).setLevel(_logging.CRITICAL)


class _SpacebarListener:
    """Watch stdin for a spacebar press in a background thread (TTY only).

    Lets the user end a streaming run early *gracefully* -- the partial result
    is finalized normally (unlike Ctrl-C, which aborts). A daemon thread puts
    the terminal in cbreak mode and polls stdin; pressing space sets
    :attr:`pressed`, which the driver loop checks each batch. Automatically
    disabled when stdin is not an interactive TTY (pipes, tests, CI), so
    non-interactive runs are unaffected.
    """

    def __init__(self):
        self.pressed = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        try:
            self.enabled = sys.stdin is not None and sys.stdin.isatty()
        except (ValueError, OSError):
            self.enabled = False

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def _watch(self):
        try:
            import select
            import termios
            import tty
        except ImportError:  # non-POSIX terminal: silently disable
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready and sys.stdin.read(1) == " ":
                    self.pressed.set()
                    return
        except Exception:
            return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def _fmt_max(max_thrown):
    """Format the max-thrown backstop, rendering an infinite cap as ``∞``."""
    return "∞" if not np.isfinite(max_thrown) else f"{int(max_thrown):,}"


class _AdaptiveBatch:
    """Tune the streaming batch size toward the throughput "knee".

    When ``--batch-size`` is not given, the driver lets this controller choose
    each batch's size. From the batch size ``B`` and the inter-completion gap
    ``dt`` (wall seconds between consecutive batch completions), ``B / dt``
    estimates the aggregate event throughput -- in steady state with ``W`` busy
    workers, completions arrive roughly every ``service_time / W`` seconds, so
    ``B / dt ~ B * W / service_time``. The controller grows ``B`` while a smoothed
    throughput keeps improving and *settles at the knee* once gains fall below
    ``min_gain`` -- i.e. the smallest batch that still saturates throughput. That
    keeps the run efficient (overhead amortized) while staying responsive
    (fine-grained stopping / live updates / quick reaction to the precision
    target). ``B`` is clamped to ``[lo, hi]``.

    The integral is exact for *any* ``B`` (the sketch is additive), so adaptation
    never affects correctness, only speed. It does make the *contents* of batch
    ``i`` timing-dependent, so an adaptive run is not bit-reproducible -- pass
    ``--batch-size`` for a fixed, reproducible schedule.
    """

    def __init__(
        self,
        start,
        lo=1000,
        hi=50000,
        step=1.4,
        min_gain=0.03,
        alpha=0.4,
        eval_every=3,
    ):
        self.lo = lo
        self.hi = hi
        self.step = step
        self.min_gain = min_gain
        self.alpha = alpha
        self.eval_every = eval_every
        self.size = self._clamp(int(start))
        self.mode = "grow"  # grow -> hold once throughput gains diminish
        self._ewma = None
        self._ref = None  # smoothed throughput at the last decision point
        self._since = 0

    def _clamp(self, b):
        return int(min(self.hi, max(self.lo, b)))

    def current(self):
        return self.size

    def update(self, batch_size, dt):
        """Fold one completion ``(size, gap)`` and return the next batch size."""
        if dt <= 0.0:
            return self.size
        rate = batch_size / dt
        self._ewma = (
            rate
            if self._ewma is None
            else self.alpha * rate + (1.0 - self.alpha) * self._ewma
        )
        self._since += 1
        if self._since < self.eval_every:  # average out parallel-overlap noise
            return self.size
        self._since = 0

        if self.mode == "grow":
            if self._ref is not None:
                gain = (self._ewma - self._ref) / self._ref
                if gain < self.min_gain:
                    # Diminishing returns: step back to the knee and hold.
                    self.size = self._clamp(int(self.size / self.step))
                    self.mode = "hold"
                    self._ref = self._ewma
                    return self.size
            self._ref = self._ewma
            grown = self._clamp(int(self.size * self.step))
            if grown == self.size:  # hit the cap; nothing more to gain
                self.mode = "hold"
            self.size = grown
        return self.size


def _stop_reason(sketch, started, st, max_thrown):
    """Return the name of a tripped stop condition, or ``None`` to continue."""
    if sketch.n_thrown >= max_thrown:
        return "max_thrown"
    if st.max_walltime_s > 0 and (time.monotonic() - started) >= st.max_walltime_s:
        return "walltime"
    if (
        sketch.n_thrown >= st.min_thrown
        and sketch.mcint > 0.0
        and sketch.rel_unc <= st.rel_unc_target
    ):
        return "rel_unc_target"
    return None


def _render(sketch, n_res, k, elapsed, st, max_thrown, interruptible=False, batch=None):
    """A rich panel of the current running statistics."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="blue")
    grid.add_column(style="bold")

    rate = sketch.n_thrown / elapsed if elapsed > 0 else 0.0
    rel = sketch.rel_unc
    rel_str = f"{rel * 100:.3f}%" if np.isfinite(rel) else "—"

    grid.add_row("Thrown", f"{sketch.n_thrown:,} / {_fmt_max(max_thrown)}")
    grid.add_row("Valid", f"{sketch.n_valid:,}")
    grid.add_row("Passing", f"{sketch.n_pass:,}")
    grid.add_row("MC integral", f"{sketch.mcint:.6g}  ±  {sketch.mcunc:.3g}")
    grid.add_row("Rel. uncert", f"{rel_str}  (target {st.rel_unc_target * 100:.3f}%)")
    grid.add_row("GEO-only", f"{sketch.mcint_geo:.6g}")
    grid.add_row("Reservoir", f"{n_res:,} / {k:,}")
    if batch is not None:
        grid.add_row("Batch", batch)
    grid.add_row("Elapsed", f"{elapsed:.1f}s   ({rate:,.0f} ev/s)")

    subtitle = "[dim]Press [Space] to end[/]" if interruptible else None
    return Panel(
        grid,
        title="[cyan]NuSpaceSim streaming (optical)[/]",
        subtitle=subtitle,
        width=64,
    )


def _stream_loop(
    client,
    config,
    st,
    sketch,
    reservoir,
    *,
    adaptive,
    sizer,
    fixed_size,
    max_thrown,
    started,
    root_entropy,
    verbose,
    console,
):
    """Submit/merge batches until a stop condition trips.

    Drives a sliding window of dask futures: each completion merges into the
    running sketch + reservoir, feeds the adaptive sizer, refreshes the live
    panel, and checks the stop criteria. Returns ``(sketch, reservoir,
    stop_reason)``. Outstanding futures are cancelled on stop.
    """
    cfg = client.scatter(config, broadcast=True)
    n_workers = max(len(client.scheduler_info().get("workers", {})), 1)
    window = max(2 * n_workers, 4)

    def cur_size():
        return sizer.current() if adaptive else fixed_size

    ac = as_completed()
    pending = set()
    fut_size = {}
    next_idx = 0

    def submit_one(idx):
        size = cur_size()
        fut = client.submit(
            run_batch,
            cfg,
            idx,
            size,
            st.reservoir_size,
            st.reservoir_weighting,
            root_entropy,
            pure=False,
        )
        ac.add(fut)
        pending.add(fut)
        fut_size[fut] = size

    for _ in range(window):
        submit_one(next_idx)
        next_idx += 1

    listener = _SpacebarListener()
    live_cm = (
        Live(console=console, refresh_per_second=8, transient=False)
        if verbose
        else None
    )
    stop_reason = "max_thrown"
    if live_cm is not None:
        live_cm.__enter__()
    listener.__enter__()
    last_t = time.monotonic()
    try:
        for fut in ac:
            part_sketch, part_res = fut.result()
            pending.discard(fut)

            # Feed the adaptive controller the completed batch's size and the
            # gap since the previous completion (the throughput signal).
            bs = fut_size.pop(fut, None)
            now = time.monotonic()
            if adaptive and bs is not None:
                sizer.update(bs, now - last_t)
            last_t = now

            sketch = sketch.merge(part_sketch)
            reservoir = reservoir.merge(part_res)

            if live_cm is not None:
                batch_now = (
                    f"{cur_size():,} (adaptive)" if adaptive else f"{fixed_size:,}"
                )
                live_cm.update(
                    _render(
                        sketch,
                        len(reservoir),
                        st.reservoir_size,
                        now - started,
                        st,
                        max_thrown,
                        interruptible=listener.enabled,
                        batch=batch_now,
                    )
                )

            if listener.pressed.is_set():
                stop_reason = "user_interrupt"
                break

            reason = _stop_reason(sketch, started, st, max_thrown)
            if reason is not None:
                stop_reason = reason
                break

            submit_one(next_idx)
            next_idx += 1
    finally:
        listener.__exit__()
        if live_cm is not None:
            live_cm.__exit__(None, None, None)

    # Computation done: quiet the workers before cancelling/closing so their
    # shutdown heartbeat/comm errors don't spill onto the console. Best-effort:
    # a worker that already died mid-teardown is exactly the noise case, so a
    # failure here must never mask the finished result.
    with contextlib.suppress(Exception):
        client.run(_silence_worker_logs)
    if pending:
        client.cancel(list(pending))
    return sketch, reservoir, stop_reason


def compute_streaming(
    config,
    verbose: bool = False,
    output_file: str | None = None,
    to_plot: list | None = None,
    write_stages: bool = False,
    seed: int | None = None,
    max_thrown: float | None = None,
    batch_size: int | str | None = None,
):
    """Run the streaming optical Monte-Carlo and return a results table.

    Parameters
    ----------
    config : NssConfig
        Simulation configuration. ``config.simulation.streaming`` supplies the
        batch size, reservoir size, stop target, and weighting.
    verbose : bool
        Render the live statistics panel.
    output_file, to_plot, write_stages
        Accepted for signature parity with :func:`compute`. The streaming path
        materializes only the reservoir; per-batch staged writes and reservoir
        plots are not produced in this first cut. The caller writes the returned
        table.
    seed : int, optional
        Root RNG entropy. ``None`` draws fresh entropy (logged for reproducibility).
    max_thrown : float, optional
        Upper bound on thrown trajectories (the backstop). ``None`` falls back to
        ``config.simulation.thrown_events``; pass ``float("inf")`` to run until
        the precision target or a user interrupt. The ``stream`` CLI command
        defaults this to infinity.
    batch_size : int or "adaptive", optional
        Trajectories per batch. ``None`` uses ``config.simulation.streaming``'s
        fixed value; an int overrides it (fixed, reproducible schedule);
        ``"adaptive"`` lets :class:`_AdaptiveBatch` tune the size for throughput
        over the run. The ``stream`` CLI command passes ``"adaptive"`` when
        ``--batch-size`` is not given.

    Returns
    -------
    astropy.table.Table
        Reservoir rows as columns; sketch results and streaming provenance in
        ``.meta`` (legacy ``OMCINT``/``OMCINTGO``/``ONEVPASS``/``OMCINTUN`` keys).

    Notes
    -----
    On an interactive terminal, pressing the **spacebar** ends the run early but
    gracefully -- the partial result is finalized normally (``STRMSTOP`` =
    ``user_interrupt``).
    """
    console = Console(width=80, log_path=False)

    def logv(*args):
        if verbose:
            console.log(*args)

    if not config.detector.optical.enable:
        raise ValueError("Streaming compute is optical-only; enable detector.optical.")
    if config.simulation.mode == "Target":
        raise NotImplementedError(
            "Streaming compute does not support Target-of-Opportunity mode yet."
        )

    st = config.simulation.streaming
    if max_thrown is None:
        max_thrown = float(config.simulation.thrown_events)
    max_thrown = float(max_thrown)
    root_entropy = int(seed) if seed is not None else np.random.SeedSequence().entropy

    # Batch sizing: "adaptive" -> throughput controller; else a fixed size.
    if batch_size is None:
        batch_size = st.batch_size
    adaptive = isinstance(batch_size, str) and batch_size == "adaptive"
    sizer = _AdaptiveBatch(start=st.batch_size) if adaptive else None
    fixed_size = None if adaptive else int(batch_size)
    batch_label = "adaptive" if adaptive else f"{fixed_size:,}"

    # mcnorm is config-only; build one RegionGeom to seed the empty sketch.
    mcnorm = RegionGeom(config).mcnorm
    sketch = MomentSketch.empty(mcnorm)
    reservoir = WeightedReservoir.empty(st.reservoir_size)

    logv(f"Streaming optical MC: seed={root_entropy}")
    logv(
        f"\tbatch={batch_label}  reservoir={st.reservoir_size:,}  "
        f"target={st.rel_unc_target:.3g}  max_thrown={_fmt_max(max_thrown)}"
    )

    # Stopping mid-stream cancels the in-flight sliding-window futures, which
    # makes each *worker* process log a benign CancelledError at WARNING level.
    # ``silence_logs`` raises the worker/scheduler/nanny log threshold to ERROR
    # so that teardown noise (and dask's INFO chatter) stays out of the console;
    # genuine worker errors still surface via ``future.result()``.
    cluster = BackgroundCluster(threads_per_worker=1, silence_logs=logging.ERROR)
    started = time.monotonic()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Sending large graph")
            client = cluster.client()
            sketch, reservoir, stop_reason = _stream_loop(
                client,
                config,
                st,
                sketch,
                reservoir,
                adaptive=adaptive,
                sizer=sizer,
                fixed_size=fixed_size,
                max_thrown=max_thrown,
                started=started,
                root_entropy=root_entropy,
                verbose=verbose,
                console=console,
            )
    finally:
        cluster.close()

    elapsed = time.monotonic() - started
    logv(
        f"\t[green]Stopped[/] ({stop_reason}): "
        f"thrown={sketch.n_thrown:,} valid={sketch.n_valid:,} "
        f"passing={sketch.n_pass:,} in {elapsed:.1f}s"
    )
    logv(f"\t[blue]Optical MC Integral[/]: {sketch.mcint:.6g} ± {sketch.mcunc:.3g}")
    logv(f"\t[blue]GEO-only[/]: {sketch.mcint_geo:.6g}")

    # ``sizer is not None`` iff adaptive; test the object so mypy sees it too.
    batch_meta = f"adaptive({sizer.current()})" if sizer is not None else batch_label
    return _finalize(
        config, sketch, reservoir, root_entropy, stop_reason, elapsed, batch_meta
    )


def _finalize(
    config, sketch, reservoir, root_entropy, stop_reason, elapsed, batch_meta
):
    """Assemble the results table: reservoir rows + sketch/provenance metadata."""
    sim = results_table.init(config)

    if len(reservoir) > 0:
        names = list(reservoir.records.keys())
        sim.add_columns([reservoir.records[n] for n in names], names=names)

    for key, value in sketch.as_meta().items():
        sim.meta[key] = (value, _META_COMMENTS[key])

    sim.meta["STRMSEED"] = (str(root_entropy), "Streaming RNG root entropy")
    sim.meta["STRMSTOP"] = (stop_reason, "Streaming stop reason")
    sim.meta["STRMNTHR"] = (sketch.n_thrown, "Streaming total thrown trajectories")
    sim.meta["STRMNVAL"] = (sketch.n_valid, "Streaming total valid events")
    sim.meta["STRMNRES"] = (len(reservoir), "Streaming reservoir rows materialized")
    sim.meta["STRMWALL"] = (float(elapsed), "Streaming wall-clock seconds")
    sim.meta["STRMBATC"] = (batch_meta, "Streaming batch size (or adaptive(final))")
    return sim
