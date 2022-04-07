"""
Intended to be ran as a script
"""
import matplotlib.pyplot as plt
import numpy as np

# relative imports are not for scripts, absolute imports here
from nuspacesim.simulation.eas_composite.composite_eas import CompositeShowers
from nuspacesim.simulation.eas_composite.mc_mean_shwr_sampler import MCVariedMean
from nuspacesim.simulation.eas_composite.plt_routines import (
    mean_rms_plt,
    decay_channel_mult_plt,
    recursive_plt,
)

make_composites = CompositeShowers(
    alt=0, shower_end=8e3, grammage=1, tau_table_start=2000
)

comp_showers, comp_depths = make_composites(filter_errors=False)


full, trimmed, shallow = make_composites.shower_end_cuts(
    composite_showers=comp_showers,
    composite_depths=comp_depths,
    separate_showers=True,
)

full_y, full_x = full
trimmed_y, trimmed_x = trimmed
shallow_y, shallow_x = shallow

#%%
plt.figure(figsize=(8, 6), dpi=200)
recursive_plt(full_x, full_y)


full_sampler = MCVariedMean(
    full_y,
    full_x,
    n_throws=400,
    hist_bins=30,
    sample_grammage=1,
)
full_mc_rms, full_sample_grm, full_sample_shwr = full_sampler.sampling_nmax_once()
# rms_err_upper = full_sample_shwr + full_mc_rms * full_sample_shwr
# rms_err_lower = full_sample_shwr - full_mc_rms * full_sample_shwr
# abs_error = rms_err_upper - full_sample_shwr
plt.plot(full_sample_grm, full_sample_shwr * full_mc_rms, "--k")
plt.yscale("log")
#%%
plt.figure(figsize=(8, 6), dpi=200)
recursive_plt(trimmed_x, trimmed_y)
trimmed_sampler = MCVariedMean(
    trimmed_y,
    trimmed_x,
    n_throws=400,
    hist_bins=30,
    sample_grammage=1,
)
(
    trimmed_mc_rms,
    trimmed_sample_grm,
    trimmed_sample_shwr,
) = trimmed_sampler.sampling_nmax_once()
(
    trimmed_mc_rms,
    trimmed_sample_grm,
    trimmed_sample_shwr,
) = trimmed_sampler.sampling_nmax_once()
plt.plot(trimmed_sample_grm, trimmed_sample_shwr * trimmed_mc_rms, "--k")
plt.yscale("log")
