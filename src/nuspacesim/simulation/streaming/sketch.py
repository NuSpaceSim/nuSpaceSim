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

"""Exact additive sketch for the streaming Monte-Carlo integral.

The MC integral, GEO-only integral, statistical uncertainty, and passing-event
count reported by :meth:`RegionGeom.mcintegral` are all linear functionals of a
per-event contribution scaled by config-only constants (``mcnorm``,
``spec_norm``, ``sum_spec_weights``). They therefore reduce to a handful of
**additive moment accumulators** that can be folded batch-by-batch and merged
across dask workers with *no loss of precision* relative to a single one-shot
run.

This module owns only the reduction. The per-event integrand is produced by
:func:`nuspacesim.simulation.geometry.region_geometry.mc_contribution`, shared
with the one-shot path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["MomentSketch"]


@dataclass(frozen=True)
class MomentSketch:
    """Additive, mergeable accumulator for the optical MC integral.

    All fields are plain scalars so the sketch pickles to a few dozen bytes and
    ships cheaply back from a dask worker. Instances are immutable; folding a
    batch (:meth:`from_batch`) or combining partials (:meth:`merge`) returns a
    new ``MomentSketch``.

    Fields
    ------
    mcnorm
        Config-derived geometry normalization (``RegionGeom.mcnorm``); identical
        across all batches of a run.
    n_thrown
        Total trajectories thrown, **including** geometry-invalid ones. This is
        the integral's denominator (the legacy ``numTrajs``).
    n_valid
        Trajectories surviving the geometry mask (length of the contribution
        arrays). The variance is taken over this set, matching the legacy
        ``np.var(mcintfactor, ddof=1)``.
    s1, s2
        Sum and sum-of-squares of the full per-event contribution
        ``mcintfactor`` over valid events.
    s1_geo
        Sum of the geometry-only factor (post separation-angle cut) over valid
        events.
    n_pass
        Count of valid events with non-zero contribution (passing all cuts).
    """

    mcnorm: float
    n_thrown: int = 0
    n_valid: int = 0
    s1: float = 0.0
    s2: float = 0.0
    s1_geo: float = 0.0
    n_pass: int = 0

    @classmethod
    def empty(cls, mcnorm: float) -> MomentSketch:
        """A zero sketch; the identity element for :meth:`merge`."""
        return cls(mcnorm=float(mcnorm))

    @classmethod
    def from_batch(
        cls,
        mcintfactor: np.ndarray,
        geo: np.ndarray,
        n_thrown: int,
        mcnorm: float,
    ) -> MomentSketch:
        """Fold one batch's per-event arrays into a fresh sketch.

        ``mcintfactor`` and ``geo`` are the per-valid-event outputs of
        :func:`mc_contribution`. ``n_thrown`` is the number of trajectories
        thrown to produce them (valid + geometry-invalid).
        """
        mc = np.asarray(mcintfactor, dtype=np.float64)
        return cls(
            mcnorm=float(mcnorm),
            n_thrown=int(n_thrown),
            n_valid=int(mc.size),
            s1=float(np.sum(mc)),
            s2=float(np.sum(mc * mc)),
            s1_geo=float(np.sum(geo)),
            n_pass=int(np.count_nonzero(mc)),
        )

    def merge(self, other: MomentSketch) -> MomentSketch:
        """Combine two partial sketches (associative & commutative)."""
        if not np.isclose(self.mcnorm, other.mcnorm):
            raise ValueError(
                f"Cannot merge sketches with different mcnorm: "
                f"{self.mcnorm} vs {other.mcnorm}"
            )
        return replace(
            self,
            n_thrown=self.n_thrown + other.n_thrown,
            n_valid=self.n_valid + other.n_valid,
            s1=self.s1 + other.s1,
            s2=self.s2 + other.s2,
            s1_geo=self.s1_geo + other.s1_geo,
            n_pass=self.n_pass + other.n_pass,
        )

    # -- Derived quantities (the reported integral) ------------------------

    @property
    def mcint(self) -> float:
        """Optical Monte-Carlo integral."""
        if self.n_thrown == 0:
            return 0.0
        return self.s1 * self.mcnorm / self.n_thrown

    @property
    def mcint_geo(self) -> float:
        """GEO-only Monte-Carlo integral."""
        if self.n_thrown == 0:
            return 0.0
        return self.s1_geo * self.mcnorm / self.n_thrown

    @property
    def variance(self) -> float:
        """Sample variance of ``mcintfactor`` over valid events (ddof=1).

        Computed from moments: ``(s2 - s1**2 / n_valid) / (n_valid - 1)``,
        algebraically equal to ``np.var(..., ddof=1)`` (differs only by
        floating-point rounding). ``nan`` for fewer than two valid events,
        matching numpy.
        """
        if self.n_valid < 2:
            return np.nan
        v = (self.s2 - self.s1 * self.s1 / self.n_valid) / (self.n_valid - 1)
        # Guard tiny negative values from catastrophic cancellation.
        return max(v, 0.0)

    @property
    def mcunc(self) -> float:
        """Statistical uncertainty of :attr:`mcint`."""
        if self.n_thrown == 0 or self.n_valid < 2:
            return 0.0 if self.n_thrown == 0 else np.nan
        return float(np.sqrt(self.variance / self.n_thrown) * self.mcnorm)

    @property
    def rel_unc(self) -> float:
        """Relative uncertainty ``mcunc / mcint`` (``inf`` when no signal)."""
        mc = self.mcint
        if mc <= 0.0 or not np.isfinite(self.mcunc):
            return np.inf
        return self.mcunc / mc

    def as_meta(self) -> dict:
        """Sketch results keyed by the legacy FITS header names."""
        return {
            "OMCINT": self.mcint,
            "OMCINTGO": self.mcint_geo,
            "ONEVPASS": self.n_pass,
            "OMCINTUN": self.mcunc,
        }
