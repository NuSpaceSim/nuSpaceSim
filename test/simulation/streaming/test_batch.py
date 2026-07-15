"""In-process tests for the streaming batch worker.

``run_batch`` normally executes inside dask worker processes, where the parent
process's coverage and assertions can't see it. These tests drive it directly
in-process to lock its contracts: determinism from ``(root_entropy,
batch_idx)``, batch independence, and the shape of the returned summaries.
"""

import numpy as np

from nuspacesim.config import NssConfig
from nuspacesim.simulation.streaming.batch import run_batch


def _config():
    config = NssConfig()
    config.detector.optical.enable = True
    config.detector.radio.enable = False
    config.simulation.mode = "Diffuse"
    return config


def test_run_batch_returns_consistent_summaries():
    sk, res = run_batch(
        _config(),
        batch_idx=0,
        batch_size=5000,
        reservoir_size=100,
        weighting="contribution",
        root_entropy=20260625,
    )
    assert sk.n_thrown == 5000
    assert 0 < sk.n_valid <= 5000
    assert sk.n_pass >= 0
    assert sk.mcint >= 0.0 and np.isfinite(sk.mcint)
    assert sk.mcint_geo > 0.0
    assert len(res) <= 100
    # contribution weighting: retained rows all passed the trigger
    if len(res):
        assert np.all(res.records["mcintfactor"] > 0.0)
        assert set(res.records) >= {"beta_rad", "numPEs", "costhetaChEff"}


def test_run_batch_deterministic_in_batch_idx_and_entropy():
    kw = {
        "batch_size": 5000,
        "reservoir_size": 100,
        "weighting": "contribution",
        "root_entropy": 12345,
    }
    sk1, res1 = run_batch(_config(), batch_idx=3, **kw)
    sk2, res2 = run_batch(_config(), batch_idx=3, **kw)
    assert sk1 == sk2  # frozen dataclass: bit-identical accumulators
    np.testing.assert_array_equal(np.sort(res1.keys), np.sort(res2.keys))

    # a different batch_idx must give a different event stream
    sk3, _ = run_batch(_config(), batch_idx=4, **kw)
    assert sk3.s1 != sk1.s1
