"""Select equally weighted exiting taus from unchanged geometry proposals."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ExitSample:
    groundecef: np.ndarray
    vecef: np.ndarray
    beta_tr: np.ndarray
    azimuth: np.ndarray
    nexitprobcandidates: int
    naccepted: int
    pmax: float
    beta_bins: np.ndarray
    before_counts: np.ndarray
    weighted_counts: np.ndarray
    cut_counts: dict[str, int]


def sample_exit_geometries(
    tau,
    gen_points,
    n,
    radius,
    log_e_nu,
    maxangle,
    *,
    batch_size=None,
    rng=None,
    log=None,
):
    """Return n geometries selected with probability Pexit(beta, E) / pmax(E).

    Request every valid geometry in each generated batch. nexitprobcandidates counts
    exit trials in candidate order, through the nth accepted event, excluding
    internal geometry rejections and the unused tail of the final batch.
    naccepted is exactly n; nexitprobcandidates - naccepted counts rejected exit trials.
    Histograms likewise include only the consumed candidates.
    cut_counts also traces the consumed prefix back to original sphere pairs.
    norig accumulates all raw proposals from every completed batch, including
    batches with no accepted exits, plus the consumed prefix of the final batch.
    Geometry rejections count; the unused final-batch tail does not.
    """
    if n <= 0:
        raise ValueError("The requested number of exiting events must be positive.")
    if batch_size is None:
        batch_size = min(n, 100_000)
    if batch_size <= 0:
        raise ValueError("Geometry batch size must be positive.")
    rng = np.random if rng is None else rng
    pmax = tau.tau_exit_prob_max(log_e_nu)
    beta_bins = np.linspace(0, maxangle, 101)
    before_counts = np.zeros(100, dtype=np.int64)
    weighted_counts = np.zeros(100, dtype=float)
    chunks = [[], [], [], []]
    nexitprobcandidates = naccepted = retained = 0
    cut_counts = dict(norig=0, n_involume=0, n_cutEarth=0, n_beta=0, n_exitprob=0)

    while retained < n:
        geometry, ancestry = gen_points(
            batch_size, radius, maxang=maxangle, return_all=True,
            return_cutflow=True,
        )
        betas = geometry[2]
        if betas.size == 0:
            raise RuntimeError("Geometry generation returned no valid candidates.")
        probabilities = tau.tau_exit_prob(
            betas, np.full(betas.shape, log_e_nu, dtype=float)
        )
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or np.any(probabilities > pmax * (1 + 1e-12))
        ):
            raise ValueError("Exit probabilities are invalid or exceed pmax.")
        keep = rng.random(betas.size) < np.minimum(probabilities / pmax, 1)
        selected = np.flatnonzero(keep)
        remaining = n - retained
        # Consume the candidate stream in order. After the last required success,
        # the remaining generated geometries are unused, not exit trials.
        if selected.size >= remaining:
            selected = selected[:remaining]
            tested = int(selected[-1]) + 1
        else:
            tested = betas.size
        # In complete batches, geometry failures after the last beta survivor
        # also count. In the final batch stop at the original index of the last
        # required success, excluding every later raw candidate at every stage.
        raw_stop = (
            int(ancestry["beta_indices"][tested - 1]) + 1
            if retained + selected.size == n else int(ancestry["n_generated"])
        )
        cut_counts["norig"] += raw_stop
        cut_counts["n_involume"] += int(np.searchsorted(
            ancestry["volume_indices"], raw_stop
        ))
        cut_counts["n_cutEarth"] += int(np.searchsorted(
            ancestry["earth_indices"], raw_stop
        ))
        cut_counts["n_beta"] += tested
        cut_counts["n_exitprob"] += int(selected.size)
        nexitprobcandidates += tested
        naccepted += int(selected.size)
        before_counts += np.histogram(betas[:tested], bins=beta_bins)[0]
        weighted_counts += np.histogram(
            betas[:tested], bins=beta_bins, weights=probabilities[:tested]
        )[0]

        if selected.size:
            for parts, values in zip(chunks, geometry):
                parts.append(values[selected])
            retained += selected.size
        if log is not None:
            log(
                f"Exit selection: {retained}/{n} retained, {nexitprobcandidates} candidates "
                f"({nexitprobcandidates - naccepted} rejected)"
            )

    return ExitSample(
        *(np.concatenate(parts) for parts in chunks),
        nexitprobcandidates=nexitprobcandidates,
        naccepted=naccepted,
        pmax=pmax,
        beta_bins=beta_bins,
        before_counts=before_counts,
        weighted_counts=weighted_counts,
        cut_counts=cut_counts,
    )
