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

"""Mergeable weighted reservoir sample (Efraimidis-Spirakis bottom-k).

Each item ``i`` drawn from the stream is assigned a random key

    key_i = u_i ** (1 / w_i),   u_i ~ Uniform(0, 1),

and the reservoir keeps the ``k`` items with the **largest** keys. This is the
A-Res / A-ExpJ scheme of Efraimidis & Spirakis (2006). We store and compare the
keys in **log space** (``log(u_i) / w_i``): the ordering is identical, but the
log form is immune to underflow. Optical contributions are tiny in absolute
value (~1e-5), so ``1/w`` is ~1e5 and the direct ``u ** (1/w)`` rounds almost
every key to ``0.0`` -- making real events indistinguishable from zero-weight
ones. The log key also makes the sample invariant to a global rescaling of the
weights (it only shifts every log key by a constant factor, preserving order).

Two properties make it the right fit for a distributed, batched pipeline:

* **Associative & commutative** — merging two reservoirs is just "keep the top-k
  keys of the union", so partial reservoirs from independent dask batches
  tree-reduce in any order to the same result.
* **Weight-tunable** — with equal weights it reduces to a uniform sample (the
  distribution of Vitter's Algorithm L); with ``w_i = |contribution_i|`` it
  biases the retained sample toward the events that dominate the MC integral.

Larger weight ⇒ ``1/w`` smaller ⇒ key closer to its maximum ⇒ more likely kept.
Zero-weight items get key ``-inf`` and are retained only to backfill an
under-full reservoir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["WeightedReservoir"]


def _topk(keys: np.ndarray, records: dict[str, np.ndarray], k: int):
    """Return the ``k`` rows with the largest keys (no order guarantee)."""
    n = keys.size
    if n <= k:
        return keys, records
    # Indices of the k largest keys; argpartition is O(n).
    idx = np.argpartition(keys, n - k)[n - k :]
    return keys[idx], {name: col[idx] for name, col in records.items()}


@dataclass
class WeightedReservoir:
    """A bounded weighted-without-replacement sample of stream records.

    Holds at most ``k`` rows as parallel arrays: ``keys`` (the E-S sampling
    keys) and ``records`` (a dict of column name -> aligned array). Pickles
    cheaply (``k`` rows), so a worker returns its batch reservoir to the driver
    for merging.
    """

    k: int
    keys: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    records: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.k < 1:
            raise ValueError(f"reservoir size k must be >= 1, got {self.k}")
        for name, col in self.records.items():
            if len(col) != self.keys.size:
                raise ValueError(
                    f"record column {name!r} length {len(col)} != "
                    f"keys length {self.keys.size}"
                )

    @classmethod
    def empty(cls, k: int) -> WeightedReservoir:
        return cls(k=k)

    @classmethod
    def from_batch(
        cls,
        k: int,
        records: dict[str, np.ndarray],
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> WeightedReservoir:
        """Build a reservoir from one batch of records and their weights.

        ``records`` maps column name -> array (all the same length ``n``);
        ``weights`` is the per-row sampling weight (``n``-vector). ``rng`` is a
        per-batch :class:`numpy.random.Generator` so keys are reproducible and
        independent of how batches are scheduled.
        """
        w = np.asarray(weights, dtype=np.float64)
        n = w.size
        if n == 0:
            return cls.empty(k)
        u = rng.random(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Log-domain E-S key: order by log(u)/w == order by u**(1/w), but
            # without underflow for tiny weights. Zero/negative weight -> -inf,
            # so such rows are retained only to backfill an under-full reservoir.
            keys = np.where(w > 0.0, np.log(u) / w, -np.inf)
        records = {name: np.asarray(col) for name, col in records.items()}
        keys, records = _topk(keys, records, k)
        return cls(k=k, keys=keys, records=records)

    def merge(self, other: WeightedReservoir) -> WeightedReservoir:
        """Combine two reservoirs: keep the top-k keys of the union."""
        if self.k != other.k:
            raise ValueError(f"reservoir size mismatch: {self.k} vs {other.k}")
        if self.keys.size == 0:
            return other
        if other.keys.size == 0:
            return self
        if self.records.keys() != other.records.keys():
            raise ValueError(
                "cannot merge reservoirs with different columns: "
                f"{sorted(self.records)} vs {sorted(other.records)}"
            )
        keys = np.concatenate([self.keys, other.keys])
        records = {
            name: np.concatenate([self.records[name], other.records[name]])
            for name in self.records
        }
        keys, records = _topk(keys, records, self.k)
        return WeightedReservoir(k=self.k, keys=keys, records=records)

    def __len__(self) -> int:
        return self.keys.size
