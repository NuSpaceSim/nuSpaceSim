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


make_composites_00km = CompositeShowers(
    alt=0, shower_end=8e3, grammage=10, tau_table_start=3000
)

comp_showers_00km, comp_depths_00km = make_composites_00km(filter_errors=False)

trimmed_showers_00km, _ = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km,
    composite_depths=comp_depths_00km,
    separate_showers=False,
)

full, trimmed, shallow = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km,
    composite_depths=comp_depths_00km,
    separate_showers=True,
)
full_y, full_x = full
trimmed_y, trimmed_x = trimmed
shallow_y, shallow_x = shallow

#%%
for i in range(50):
    sampler = MCVariedMean(
        trimmed_y,
        trimmed_x,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr = sampler.sampling_nmax_once()

    plt.figure(figsize=(8, 6), dpi=200)
    recursive_plt(trimmed_x, trimmed_y)
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="k",
        linestyle=":",
        zorder=20,
        linewidth=5,
    )
    plt.ylim(1e0, 1e8)

    plt.title("Sampling Nmax Once For Rebounded and Cut Showers")
    plt.xlabel("slant depth g cm$^{-2}$")
    plt.ylabel("$N$")
    plt.yscale("log")
    plt.savefig("./once_{}.png".format(str(i).zfill(2)))
