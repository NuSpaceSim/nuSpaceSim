"""WeightedReservoir: mergeability, bounded size, and sampling correctness."""

import numpy as np
import pytest
from scipy import stats

from nuspacesim.simulation.streaming.reservoir import WeightedReservoir


def _batch_records(start, n):
    ids = np.arange(start, start + n)
    return {"id": ids, "value": ids.astype(np.float64)}


def test_size_never_exceeds_k():
    rng = np.random.default_rng(0)
    k = 100
    res = WeightedReservoir.empty(k)
    for b in range(20):
        recs = _batch_records(b * 500, 500)
        w = rng.uniform(0.1, 5.0, 500)
        res = res.merge(WeightedReservoir.from_batch(k, recs, w, rng))
        assert len(res) <= k
    assert len(res) == k  # well over k seen total


def test_merge_is_order_independent():
    # Fixed per-batch reservoirs (keys frozen), then merge in different orders.
    parts = []
    for b in range(12):
        rng = np.random.default_rng(1000 + b)  # deterministic keys per batch
        recs = _batch_records(b * 300, 300)
        w = rng.uniform(0.1, 9.0, 300)
        parts.append(WeightedReservoir.from_batch(64, recs, w, rng))

    def fold(seq):
        r = WeightedReservoir.empty(64)
        for p in seq:
            r = r.merge(p)
        return set(r.records["id"].tolist())

    forward = fold(parts)
    backward = fold(list(reversed(parts)))
    shuffled = fold(
        [parts[i] for i in np.random.default_rng(7).permutation(len(parts))]
    )

    assert forward == backward == shuffled
    assert len(forward) == 64


def test_uniform_weights_give_uniform_sample():
    # Equal weights -> keys are i.i.d. uniform -> retained values are a uniform
    # random subset, independent of the value field.
    n, k = 40_000, 4_000
    rng = np.random.default_rng(20260625)
    recs = {"value": np.arange(n, dtype=np.float64)}
    w = np.ones(n)
    res = WeightedReservoir.from_batch(k, recs, w, rng)
    assert len(res) == k
    retained = res.records["value"] / n  # -> [0, 1)
    ks = stats.kstest(retained, "uniform")
    assert ks.pvalue > 0.01, f"retained sample not uniform (p={ks.pvalue})"


def test_contribution_weighting_biases_toward_large_values():
    # weight == value: large-value items should be over-represented vs uniform.
    n, k = 40_000, 2_000
    rng = np.random.default_rng(42)
    values = np.arange(1, n + 1, dtype=np.float64)
    recs = {"value": values}
    weighted = WeightedReservoir.from_batch(k, recs, values, rng)
    uniform = WeightedReservoir.from_batch(k, recs, np.ones(n), rng)

    pop_mean = values.mean()
    # Weighted reservoir's retained mean should sit well above the population
    # mean; the uniform one should hug it.
    assert weighted.records["value"].mean() > 1.2 * pop_mean
    assert uniform.records["value"].mean() == pytest.approx(pop_mean, rel=0.1)


def test_determinism_same_seed_same_keys():
    recs = _batch_records(0, 1000)
    w = np.linspace(0.1, 3.0, 1000)
    a = WeightedReservoir.from_batch(50, recs, w, np.random.default_rng(5))
    b = WeightedReservoir.from_batch(50, recs, w, np.random.default_rng(5))
    np.testing.assert_array_equal(np.sort(a.keys), np.sort(b.keys))
    assert set(a.records["id"].tolist()) == set(b.records["id"].tolist())


def test_empty_merge_identity_and_underfull():
    rng = np.random.default_rng(3)
    recs = _batch_records(0, 30)
    r = WeightedReservoir.from_batch(100, recs, np.ones(30), rng)
    assert len(r) == 30  # under-full: keep everything
    e = WeightedReservoir.empty(100)
    assert set(e.merge(r).records["id"].tolist()) == set(r.records["id"].tolist())
    assert set(r.merge(e).records["id"].tolist()) == set(r.records["id"].tolist())


def test_invalid_k_raises():
    with pytest.raises(ValueError, match="k must be >= 1"):
        WeightedReservoir.empty(0)


def test_mismatched_column_length_raises():
    with pytest.raises(ValueError, match="length"):
        WeightedReservoir(k=10, keys=np.zeros(3), records={"a": np.zeros(2)})


def test_merge_mismatched_k_raises():
    a = WeightedReservoir.empty(10)
    b = WeightedReservoir.empty(20)
    with pytest.raises(ValueError, match="size mismatch"):
        a.merge(b)


def test_merge_mismatched_columns_raises():
    rng = np.random.default_rng(0)
    a = WeightedReservoir.from_batch(10, {"x": np.arange(5)}, np.ones(5), rng)
    b = WeightedReservoir.from_batch(10, {"y": np.arange(5)}, np.ones(5), rng)
    with pytest.raises(ValueError, match="different columns"):
        a.merge(b)


def test_zero_weight_items_only_backfill():
    # Half the items have zero weight; with k smaller than the positive-weight
    # count, no zero-weight item should be retained.
    rng = np.random.default_rng(11)
    n = 1000
    w = np.ones(n)
    w[: n // 2] = 0.0  # first half zero-weight
    recs = {"idx": np.arange(n)}
    res = WeightedReservoir.from_batch(200, recs, w, rng)
    assert np.all(res.records["idx"] >= n // 2)
