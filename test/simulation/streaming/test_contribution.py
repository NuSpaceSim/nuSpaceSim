"""Regression lock: ``mc_contribution`` must reproduce the original inlined
``RegionGeom.mcintegral`` per-event math byte-for-byte.

The oracle below is a verbatim copy of the pre-refactor ``mcintegral`` body
(operating on plain per-event arrays). If the extracted pure function ever
drifts from it, these bit-exact assertions fail.
"""

import numpy as np

from nuspacesim.simulation.geometry.region_geometry import mc_contribution


def _original_mcintegral(
    cos_tr_sub_n,
    cos_n_sub_v,
    cos_tr_sub_v,
    triggers,
    costheta,
    tauexitprob,
    threshold,
    spec_norm,
    spec_weights_sum,
    mcnorm,
    numTrajs,
):
    """Verbatim pre-refactor math (the regression oracle)."""
    cossepangle = cos_tr_sub_v

    mcintfactor = cos_tr_sub_n / cos_n_sub_v / cos_tr_sub_v

    mcintfactor[cossepangle < costheta] = 0
    mcintegralgeoonly = np.sum(mcintfactor) * mcnorm / numTrajs

    Bshr = 0.826

    mcintfactor *= Bshr * tauexitprob
    mcintfactor /= spec_norm
    mcintfactor /= spec_weights_sum

    mcintfactor[triggers < threshold] = 0
    mcintegral = np.sum(mcintfactor) * mcnorm / numTrajs
    mcintegraluncert = np.sqrt(np.var(mcintfactor, ddof=1) / numTrajs) * mcnorm

    numEvPass = np.count_nonzero(mcintfactor)

    return mcintegral, mcintegralgeoonly, numEvPass, mcintegraluncert


def _sample(rng, n, per_event_costheta=True):
    cos_tr_sub_n = rng.uniform(0.2, 1.0, n)
    cos_n_sub_v = rng.uniform(0.3, 1.0, n)
    cos_tr_sub_v = rng.uniform(0.985, 1.0, n)
    # numPEs-like: many below threshold, some far above
    triggers = rng.lognormal(mean=2.0, sigma=2.0, size=n)
    tauexitprob = rng.uniform(0.0, 1.0, n)
    if per_event_costheta:
        # costhetaChEff near 1; some events fail the separation cut
        costheta = rng.uniform(0.99, 1.0, n)
    else:
        costheta = 0.9986  # scalar (radio-like)
    return cos_tr_sub_n, cos_n_sub_v, cos_tr_sub_v, triggers, tauexitprob, costheta


def test_mc_contribution_matches_original_reduction_bitexact():
    rng = np.random.default_rng(20260625)
    for trial in range(25):
        n = int(rng.integers(50, 5000))
        cos_tr_sub_n, cos_n_sub_v, cos_tr_sub_v, triggers, tauexitprob, costheta = (
            _sample(rng, n, per_event_costheta=bool(trial % 2))
        )
        threshold = 10.0
        spec_norm = rng.uniform(0.5, 2.0)
        spec_weights_sum = rng.uniform(0.5, 2.0)
        mcnorm = rng.uniform(1e3, 1e6)
        numTrajs = n + int(rng.integers(0, n))  # invalid events inflate the denom

        want = _original_mcintegral(
            cos_tr_sub_n.copy(),
            cos_n_sub_v.copy(),
            cos_tr_sub_v.copy(),
            triggers.copy(),
            np.array(costheta).copy(),
            tauexitprob.copy(),
            threshold,
            spec_norm,
            spec_weights_sum,
            mcnorm,
            numTrajs,
        )

        mcintfactor, geo = mc_contribution(
            cos_tr_sub_n.copy(),
            cos_n_sub_v.copy(),
            cos_tr_sub_v.copy(),
            triggers.copy(),
            costheta,
            tauexitprob.copy(),
            threshold,
            spec_norm,
            spec_weights_sum,
        )
        got = (
            np.sum(mcintfactor) * mcnorm / numTrajs,
            np.sum(geo) * mcnorm / numTrajs,
            np.count_nonzero(mcintfactor),
            np.sqrt(np.var(mcintfactor, ddof=1) / numTrajs) * mcnorm,
        )

        # Bit-exact: same float operations in the same order.
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
        assert got[2] == want[2]
        np.testing.assert_array_equal(got[3], want[3])


def test_mc_contribution_does_not_mutate_geo_into_full():
    """``geo`` (GEO-only sum) must stay the separation-cut base, not the full
    contribution."""
    rng = np.random.default_rng(7)
    n = 1000
    cos_tr_sub_n, cos_n_sub_v, cos_tr_sub_v, triggers, tauexitprob, costheta = _sample(
        rng, n
    )
    mcintfactor, geo = mc_contribution(
        cos_tr_sub_n,
        cos_n_sub_v,
        cos_tr_sub_v,
        triggers,
        costheta,
        tauexitprob,
        10.0,
        1.0,
        1.0,
    )
    # geo is the pre-weight base: where it survived the sep cut it must equal
    # cos_tr_sub_n / cos_n_sub_v / cos_tr_sub_v exactly.
    survived = cos_tr_sub_v >= costheta
    base = cos_tr_sub_n / cos_n_sub_v / cos_tr_sub_v
    np.testing.assert_array_equal(geo[survived], base[survived])
    # full contribution is <= geo*const, and zero below threshold
    assert np.all(mcintfactor[triggers < 10.0] == 0.0)
