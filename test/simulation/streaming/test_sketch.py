"""MomentSketch: the streaming integral must equal the one-shot integral.

Two proofs:

1. **Authoritative single-batch** — a real :class:`RegionGeom` run: a sketch
   built from the same per-event arrays reproduces ``RegionGeom.mcintegral``
   exactly for the linear terms and to floating tolerance for the variance.
2. **Merge exactness** — splitting one sample into many random batches and
   merging the partial sketches reproduces the single-batch sketch (additivity
   is order-independent).
"""

import numpy as np
import pytest

from nuspacesim.config import NssConfig
from nuspacesim.simulation.geometry.region_geometry import RegionGeom, mc_contribution
from nuspacesim.simulation.streaming.sketch import MomentSketch


def _real_region_geom(n_thrown=200_000, seed=20260625):
    np.random.seed(seed)
    geom = RegionGeom(NssConfig())
    geom.throw(n_thrown)
    n_valid = int(np.count_nonzero(geom.event_mask))
    rng = np.random.default_rng(seed + 1)
    # Fabricate detector response on the valid set.
    triggers = rng.lognormal(mean=2.5, sigma=1.5, size=n_valid)  # numPEs-like
    tauexitprob = rng.uniform(0.0, 1.0, n_valid)
    costheta = rng.uniform(0.99, 1.0, n_valid)  # costhetaChEff-like
    return geom, triggers, costheta, tauexitprob


def test_sketch_matches_real_region_geom_single_batch():
    geom, triggers, costheta, tauexitprob = _real_region_geom()
    threshold, spec_norm, spec_weights_sum = 10.0, 1.0, 1.0

    want = geom.mcintegral(
        triggers, costheta, tauexitprob, threshold, spec_norm, spec_weights_sum
    )
    mcint_ref, geo_ref, npass_ref, unc_ref = want

    mcintfactor, geo = mc_contribution(
        geom.valid_costhetaTrSubN(),
        geom.valid_costhetaNSubV(),
        geom.valid_costhetaTrSubV(),
        triggers,
        costheta,
        tauexitprob,
        threshold,
        spec_norm,
        spec_weights_sum,
    )
    sketch = MomentSketch.from_batch(
        mcintfactor, geo, n_thrown=len(geom.betaTrSubN), mcnorm=geom.mcnorm
    )

    # Linear terms: identical float operations -> bit-exact.
    assert sketch.mcint == mcint_ref
    assert sketch.mcint_geo == geo_ref
    assert sketch.n_pass == npass_ref
    # Variance via moments vs np.var(ddof=1): equal to floating tolerance.
    assert sketch.mcunc == pytest.approx(unc_ref, rel=1e-9)
    assert sketch.rel_unc == pytest.approx(unc_ref / mcint_ref, rel=1e-9)


def test_sketch_merge_reproduces_single_batch():
    geom, triggers, costheta, tauexitprob = _real_region_geom()
    threshold = 10.0
    mcintfactor, geo = mc_contribution(
        geom.valid_costhetaTrSubN(),
        geom.valid_costhetaNSubV(),
        geom.valid_costhetaTrSubV(),
        triggers,
        costheta,
        tauexitprob,
        threshold,
        1.0,
        1.0,
    )
    n_thrown_total = len(geom.betaTrSubN)
    n_valid = mcintfactor.size

    single = MomentSketch.from_batch(mcintfactor, geo, n_thrown_total, geom.mcnorm)

    # Partition valid events AND the thrown denominator across random batches.
    rng = np.random.default_rng(99)
    order = rng.permutation(n_valid)
    n_batches = 17
    valid_chunks = np.array_split(order, n_batches)
    # distribute the geometry-invalid throws (n_thrown_total - n_valid)
    invalid_total = n_thrown_total - n_valid
    inv_per = np.full(n_batches, invalid_total // n_batches, dtype=int)
    inv_per[: invalid_total % n_batches] += 1

    merged = MomentSketch.empty(geom.mcnorm)
    for chunk, inv in zip(valid_chunks, inv_per):
        part = MomentSketch.from_batch(
            mcintfactor[chunk],
            geo[chunk],
            n_thrown=len(chunk) + int(inv),
            mcnorm=geom.mcnorm,
        )
        merged = merged.merge(part)

    assert merged.n_thrown == single.n_thrown
    assert merged.n_valid == single.n_valid
    assert merged.n_pass == single.n_pass
    assert merged.mcint == pytest.approx(single.mcint, rel=1e-12)
    assert merged.mcint_geo == pytest.approx(single.mcint_geo, rel=1e-12)
    assert merged.mcunc == pytest.approx(single.mcunc, rel=1e-10)


def test_sketch_merge_is_order_independent():
    rng = np.random.default_rng(3)
    mcnorm = 12345.0
    parts = []
    for _ in range(8):
        n = int(rng.integers(10, 500))
        mc = rng.lognormal(size=n)
        mc[rng.uniform(size=n) < 0.4] = 0.0  # some non-passing
        geo = rng.lognormal(size=n)
        parts.append(MomentSketch.from_batch(mc, geo, n + 5, mcnorm))

    a = MomentSketch.empty(mcnorm)
    for p in parts:
        a = a.merge(p)
    b = MomentSketch.empty(mcnorm)
    for p in reversed(parts):
        b = b.merge(p)

    assert a.n_thrown == b.n_thrown and a.n_pass == b.n_pass
    assert a.mcint == pytest.approx(b.mcint, rel=1e-12)
    assert a.mcunc == pytest.approx(b.mcunc, rel=1e-10)


def test_empty_sketch_is_merge_identity():
    rng = np.random.default_rng(1)
    mc = rng.lognormal(size=100)
    geo = rng.lognormal(size=100)
    s = MomentSketch.from_batch(mc, geo, 150, 99.0)
    e = MomentSketch.empty(99.0)
    assert e.merge(s).mcint == s.mcint
    assert s.merge(e).mcint == s.mcint
    assert e.merge(s).mcunc == s.mcunc


def test_merge_mismatched_mcnorm_raises():
    a = MomentSketch.empty(100.0)
    b = MomentSketch.empty(200.0)
    with pytest.raises(ValueError, match="different mcnorm"):
        a.merge(b)


def test_no_signal_gives_zero_integral_and_inf_rel_unc():
    s = MomentSketch.empty(1000.0)
    assert s.mcint == 0.0
    assert s.mcint_geo == 0.0
    assert s.rel_unc == np.inf
    # valid events but all below threshold -> zero contribution, finite sketch
    zeros = np.zeros(50)
    geo = np.ones(50)
    s2 = MomentSketch.from_batch(zeros, geo, 80, 1000.0)
    assert s2.mcint == 0.0
    assert s2.mcint_geo > 0.0
    assert s2.rel_unc == np.inf
