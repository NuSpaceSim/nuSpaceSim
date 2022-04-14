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
    alt=0, shower_end=8e3, grammage=1, tau_table_start=3000
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
#%% drawing lines
def separate_showers(shwr_dpths, shwr_n, sep_dpth, sep_n):

    dpth_idx = int(np.argwhere(shwr_dpths[0, :] == sep_dpth))
    shwr_content_at_depth = shwr_n[:, dpth_idx]

    above_n_mask = shwr_content_at_depth > sep_n
    below_n_mask = shwr_content_at_depth <= sep_n

    above_depths = shwr_dpths[above_n_mask]
    below_depths = shwr_dpths[below_n_mask]
    above_showers = shwr_n[above_n_mask]
    below_showers = shwr_n[below_n_mask]

    return below_depths, below_showers, above_depths, above_showers


line_at = 1e4
separation_depth = 8e3

below_x, below_y, above_x, above_y = separate_showers(
    comp_depths_00km, trimmed_showers_00km, separation_depth, line_at
)

plt.figure(figsize=(8, 6), dpi=200)
recursive_plt(below_x, below_y, color="tab:red", lbl="Below {:.1e} $N$".format(line_at))
recursive_plt(
    above_x, above_y, color="tab:blue", lbl="Above {:.1e} $N$".format(line_at)
)
plt.axhline(line_at, color="k", linestyle=":", zorder=20, linewidth=2)
plt.axvline(separation_depth, color="k", linestyle=":", zorder=20, linewidth=2)

plt.ylim(bottom=1e0)
plt.xlabel("slant depth (g cm$^{-2}$)")
plt.ylabel("$N$")
plt.yscale("log")
plt.tick_params(axis="both", which="both", direction="in", top="on", right="on")
plt.grid(True, which="both", linestyle="--")
plt.legend()
#%%

decay_channel_mult_plt(above_x, above_y)
decay_channel_mult_plt(below_x, below_y)
#%%
plt.figure(figsize=(8, 6), dpi=200)
recursive_plt(trimmed_x, trimmed_y, lbl="Rebounding to 1% Nmax", color="tab:red")
recursive_plt(full_x, full_y, lbl="Non Rebounding", color="tab:blue")

plt.axvline(
    736.0,
    label="Mean Xmax For All Showers",
    color="k",
    linestyle=":",
    zorder=20,
    linewidth=5,
)
plt.axvline(728.0, color="tab:red", linestyle=":", zorder=20, linewidth=5)
plt.axvline(740.0, color="tab:blue", linestyle=":", zorder=20, linewidth=5)


plt.ylim(bottom=1e0)


plt.xlabel("slant depth (g cm$^{-2}$)")
plt.ylabel("$N$")
plt.yscale("log")
plt.tick_params(axis="both", which="both", direction="in", top="on", right="on")
plt.grid(True, which="both", linestyle="--")
plt.legend()
#%%
plt.figure(figsize=(8, 6), dpi=200)
recursive_plt(
    trimmed_x, trimmed_y, lbl="Rebounding to 1% Nmax", color="tab:red", zorder=10
)
recursive_plt(full_x, full_y, lbl="Non Rebounding", color="tab:blue")


for i in range(20):
    sampler = MCVariedMean(
        trimmed_showers_00km,
        comp_depths_00km,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, all_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="k",
        linestyle=":",
        zorder=20,
        linewidth=5,
        # label="All Shower Sample Mean",
    )

    sampler = MCVariedMean(
        trimmed_y,
        trimmed_x,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, reb_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="orange",
        linestyle=":",
        zorder=20,
        linewidth=5,
        # label="Rebounding Sample Mean",
    )

    sampler = MCVariedMean(
        full_y,
        full_x,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, noreb_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="violet",
        linestyle=":",
        zorder=20,
        linewidth=5,
        # label="Non Rebounding Sample Mean",
    )

plt.ylim(bottom=1e0)

plt.title("Sampling RMS Distribution At Mean XMax")
plt.xlabel("slant depth (g cm$^{-2}$)")
plt.ylabel("$N$")
plt.yscale("log")
plt.tick_params(axis="both", which="both", direction="in", top="on", right="on")
plt.grid(True, which="both", linestyle="--")
plt.legend()
# plt.savefig("./once_{}.png".format(str(i).zfill(2)))
#%% Histogram for xmax rms

plt.figure(figsize=(8, 6), dpi=200)

bins = 20

strict_bins = np.linspace(0, 2.6, 21, endpoint=True)

plt.hist(
    all_rms / np.mean(all_rms),
    bins=strict_bins,
    histtype="step",
    hatch="|",
    linewidth=4,
    alpha=1,
    label=r"All Showers",
    color="k",
)

plt.hist(
    reb_rms / np.mean(reb_rms),
    bins=strict_bins,
    histtype="step",
    hatch="\\",
    linewidth=4,
    alpha=1,
    label=r"Rebounding Showers",
    color="tab:red",
)

plt.hist(
    noreb_rms / np.mean(noreb_rms),
    bins=strict_bins,
    histtype="step",
    hatch="//",
    linewidth=4,
    alpha=1,
    label=r"Non Rebounding Showers",
    color="tab:blue",
)

plt.ylabel("# of Showers / 0.13 ", fontsize=16)
plt.xlabel(r"$ N_{at Xmax} / \overline{N_{at Xmax} } $", fontsize=16)
plt.legend()
