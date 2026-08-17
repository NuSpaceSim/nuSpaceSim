"""End-to-end smoke test for the streaming optical Monte-Carlo driver.

Spins up a real (small) dask cluster and runs ``compute_streaming`` to the
max-thrown backstop, checking the returned table's structure, metadata, and
that the integral is physically sane. The scientific streaming-vs-one-shot
equivalence (within MC noise) is verified separately; here we keep it fast.
"""

import numpy as np
import pytest
from astropy.table import Table as AstropyTable

from nuspacesim.config import NssConfig
from nuspacesim.simulation.streaming import compute_streaming


@pytest.fixture
def small_streaming_config():
    config = NssConfig()
    config.detector.optical.enable = True
    config.detector.radio.enable = False
    config.simulation.mode = "Diffuse"
    config.simulation.thrown_events = 40_000  # max-thrown backstop
    s = config.simulation.streaming
    s.batch_size = 10_000
    s.reservoir_size = 200
    s.rel_unc_target = 0.0  # never met -> deterministic max_thrown stop
    s.min_thrown = 0
    return config


def test_compute_streaming_end_to_end(small_streaming_config):
    sim = compute_streaming(small_streaming_config, verbose=False, seed=20260625)

    assert isinstance(sim, AstropyTable)

    # Sketch metadata, legacy keys.
    assert sim.meta["OMCINT"][0] > 0.0
    assert sim.meta["OMCINTGO"][0] > 0.0
    assert sim.meta["ONEVPASS"][0] > 0
    assert sim.meta["OMCINTUN"][0] > 0.0
    # GEO-only integral exceeds the trigger-cut full integral.
    assert sim.meta["OMCINTGO"][0] > sim.meta["OMCINT"][0]

    # Streaming provenance.
    assert sim.meta["STRMSTOP"][0] == "max_thrown"
    assert sim.meta["STRMNTHR"][0] >= 40_000
    assert sim.meta["STRMNVAL"][0] > 0
    assert sim.meta["STRMSEED"][0] == "20260625"

    # Reservoir materialized, bounded by k.
    n_res = sim.meta["STRMNRES"][0]
    assert 0 < n_res <= 200
    assert len(sim) == n_res
    for col in ("beta_rad", "numPEs", "costhetaChEff", "mcintfactor", "init_lat"):
        assert col in sim.colnames
    assert np.all(np.isfinite(sim["numPEs"]))

    # Contribution-weighted reservoir: retained events should be passing
    # (non-zero contribution) while passing events outnumber k.
    assert sim.meta["ONEVPASS"][0] > 200
    assert np.all(sim["mcintfactor"] > 0.0)


def test_compute_streaming_infinite_cap_stops_on_precision(small_streaming_config):
    # max_thrown=inf must NOT loop forever: the precision target drives the stop.
    cfg = small_streaming_config
    cfg.simulation.streaming.rel_unc_target = 0.05
    cfg.simulation.streaming.min_thrown = 0
    sim = compute_streaming(cfg, verbose=False, seed=1, max_thrown=float("inf"))
    assert sim.meta["STRMSTOP"][0] == "rel_unc_target"
    assert 0 < sim.meta["STRMNTHR"][0] < 10_000_000
    assert sim.meta["OMCINTUN"][0] / sim.meta["OMCINT"][0] <= 0.05 + 1e-9


def test_compute_streaming_adaptive_batch(small_streaming_config):
    cfg = small_streaming_config
    cfg.simulation.streaming.rel_unc_target = 0.05
    cfg.simulation.streaming.min_thrown = 0
    sim = compute_streaming(
        cfg, verbose=False, seed=2, max_thrown=float("inf"), batch_size="adaptive"
    )
    assert sim.meta["STRMBATC"][0].startswith("adaptive(")
    assert sim.meta["OMCINT"][0] > 0.0
    assert sim.meta["STRMSTOP"][0] == "rel_unc_target"


def test_adaptive_batch_climbs_and_settles():
    from nuspacesim.simulation.streaming.driver import _AdaptiveBatch

    asymptote = 1_000_000.0
    half = 8000.0  # throughput half-saturation batch size

    def rate(B):  # increasing, saturating: B/dt == rate(B) below drives the loop
        return asymptote * B / (half + B)

    ctrl = _AdaptiveBatch(start=1000)
    B = ctrl.current()
    for _ in range(80):
        dt = B / rate(B)
        B = ctrl.update(B, dt)

    assert ctrl.mode == "hold"  # settled
    assert ctrl.current() > 1000  # grew from the small start
    assert ctrl.lo <= ctrl.current() <= ctrl.hi
    assert rate(ctrl.current()) > 0.8 * asymptote  # near saturation


def test_adaptive_batch_holds_when_starting_past_knee():
    from nuspacesim.simulation.streaming.driver import _AdaptiveBatch

    def rate(B):
        return 1_000_000.0 * B / (2000.0 + B)  # knee at a few thousand

    ctrl = _AdaptiveBatch(start=40000)  # already saturated
    B = ctrl.current()
    for _ in range(20):
        dt = B / rate(B)
        B = ctrl.update(B, dt)

    assert ctrl.mode == "hold"
    assert ctrl.current() <= 40000  # did not keep growing


def test_adaptive_batch_ignores_nonpositive_dt():
    from nuspacesim.simulation.streaming.driver import _AdaptiveBatch

    ctrl = _AdaptiveBatch(start=5000)
    assert ctrl.update(5000, 0.0) == 5000
    assert ctrl.update(5000, -1.0) == 5000


def test_fmt_max_renders_infinity():
    from nuspacesim.simulation.streaming.driver import _fmt_max

    assert _fmt_max(float("inf")) == "∞"
    assert _fmt_max(1_000_000) == "1,000,000"


def test_spacebar_listener_safe_construction():
    # Construction and enter/exit must never raise or pre-trip, regardless of
    # whether stdin is a TTY in this environment.
    from nuspacesim.simulation.streaming.driver import _SpacebarListener

    lis = _SpacebarListener()
    assert lis.pressed.is_set() is False
    with lis:
        pass
    assert lis.pressed.is_set() is False


def test_compute_streaming_rejects_target_mode(small_streaming_config):
    small_streaming_config.simulation.mode = "Target"
    with pytest.raises(NotImplementedError):
        compute_streaming(small_streaming_config, verbose=False)


def test_compute_streaming_requires_optical(small_streaming_config):
    small_streaming_config.detector.optical.enable = False
    with pytest.raises(ValueError, match="optical"):
        compute_streaming(small_streaming_config, verbose=False)
