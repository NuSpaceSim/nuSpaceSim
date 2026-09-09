#!/usr/bin/env python
"""Single-threaded line-profiling harness for CphotAng.run().

Usage:
    pip install line_profiler
    python profile_run.py [N]          # N = batch size, default 200000

Calls run() directly (NOT __call__), so there is no dask threading. Thread
counts are pinned to 1 before numpy imports, so the profile is genuinely
single-threaded (the kernel's pow/exp are already single-threaded elementwise
ops; this just guarantees no BLAS fan-out muddies the line timings).
"""
import os
import sys

import numpy as np

from nuspacesim.simulation.eas_optical import cphotang as C
from nuspacesim.simulation.eas_optical import hillas_batch_kernel as K
from nuspacesim.simulation.eas_optical import hillas_kernel as HK
from nuspacesim.simulation.eas_optical.cphotang import CphotAng

# Pin to one thread BEFORE numpy is imported.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

# Large single batch. run() is fully vectorized over showers (no Python loop),
# so one call processes the whole batch. Memory note: peak working set is a few
# (N, 16, 16) float32 arrays (~0.2 GB each at N=2e5) -- lower N if constrained.
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

rng = np.random.default_rng(20240620)
betaE = np.radians(rng.uniform(0.5, 42.0, N))  # Earth-emergence angle (rad)
alt = rng.uniform(0.0, 18.0, N)  # decay altitude (km)
Eshow = rng.uniform(0.2, 8.0, N)  # shower energy (100 PeV units)
lat = np.zeros(N)  # cloud lat/long; cloudf=None below
long_ = np.zeros(N)

cpa = CphotAng(detector_altitude=525.0)

# Warm up (lru_cache'd leggauss, import-time costs) so they don't skew line times.
cpa.run(betaE[:1000], alt[:1000], Eshow[:1000], lat[:1000], long_[:1000])

cpa.run(betaE, alt, Eshow, lat, long_)
