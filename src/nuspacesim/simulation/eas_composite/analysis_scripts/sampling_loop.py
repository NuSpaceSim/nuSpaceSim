import matplotlib.pyplot as plt
import numpy as np

# relative imports are not for scripts, absolute imports here
from nuspacesim.simulation.eas_composite.composite_eas import CompositeShowers
from nuspacesim.simulation.eas_composite.mc_mean_shwr_sampler import MCVariedMean
from nuspacesim.simulation.eas_composite.plt_routines import (
    mean_rms_plt,
    decay_channel_mult_plt,
    recursive_plt,
    get_decay_channel,
)

#%%

# TODO: put all these in utils
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


def decay_channel_filter(
    shwr_dpths, shwr_n, decay_channel, nth_digit=None, digit_flag=None, discarded=None
):
    r"""Filter out specific decay channels or decay channel type"""
    if nth_digit is not None and digit_flag is not None:

        n_mask = shwr_dpths[:, 1][nth_digit - 1] == digit_flag

        out_shwr_dpths = shwr_dpths[n_mask]
        out_shwr_n = shwr_n[n_mask]

        out_not_shwr_dpths = shwr_dpths[~n_mask]
        out_not_shwr_n = shwr_n[~n_mask]
    else:
        decay_mask = shwr_dpths[:, 1] == decay_channel

        out_shwr_dpths = shwr_dpths[decay_mask]
        out_shwr_n = shwr_n[decay_mask]

        out_not_shwr_dpths = shwr_dpths[~decay_mask]
        out_not_shwr_n = shwr_n[~decay_mask]

    if discarded is not None:
        return out_shwr_dpths, out_shwr_n, out_not_shwr_dpths, out_not_shwr_n
    else:
        return out_shwr_dpths, out_shwr_n


make_composites_00km = CompositeShowers(
    alt=0, shower_end=10e3, grammage=1, tau_table_start=3000
)

comp_showers_00km, comp_depths_00km = make_composites_00km(filter_errors=False)

trimmed_showers_00km, _ = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km,
    composite_depths=comp_depths_00km,
    shwr_threshold=0.01,
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
#%% filtering out specific decay channels
# other plots for progress report

e_dpths, e_n, not_e_dpths, not_e_n = decay_channel_filter(
    comp_depths_00km, trimmed_showers_00km, 300001, discarded=True
)

x_300111, y_300111 = decay_channel_filter(
    comp_depths_00km,
    trimmed_showers_00km,
    300111,
)

x_200011, y_200011 = decay_channel_filter(
    comp_depths_00km,
    trimmed_showers_00km,
    200011,
)


#%% drawing lines
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
    trimmed_x,
    trimmed_y,
    lbl="Rebounding to 1% Nmax",
    color="tab:red",
)
recursive_plt(full_x, full_y, lbl="Non Rebounding", color="tab:blue")


for i in range(10):
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
        linewidth=2,
        # label="All Shower Sample Mean",
    )
    # =============================================================================

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
        linewidth=2,
        # label="Rebounding Sample Mean",
    )
    # =============================================================================
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
        linewidth=2,
        # label="Non Rebounding Sample Mean",
    )
    # =============================================================================
    sampler = MCVariedMean(
        e_n,
        e_dpths,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, elec_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="tab:green",
        linestyle=":",
        zorder=20,
        linewidth=2,
        # label="Non Rebounding Sample Mean",
    )

    # =============================================================================
    sampler = MCVariedMean(
        e_n,
        e_dpths,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, elec_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="tab:green",
        linestyle=":",
        zorder=20,
        linewidth=2,
        # label="Non Rebounding Sample Mean",
    )

    sampler = MCVariedMean(
        x_300111,
        y_300111,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, pi0_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="tab:purple",
        linestyle=":",
        zorder=20,
        linewidth=2,
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
    linewidth=4,
    alpha=1,
    label=r"Rebounding Showers",
    color="tab:red",
)

plt.hist(
    noreb_rms / np.mean(noreb_rms),
    bins=strict_bins,
    histtype="step",
    linewidth=4,
    alpha=1,
    label=r"Non Rebounding Showers",
    color="tab:blue",
)


plt.ylabel("# of Showers / 0.13 ", fontsize=16)
plt.xlabel(r"$ N_{at Xmax} / \overline{N_{at Xmax} } $", fontsize=16)
plt.legend()

#%%
plt.figure(figsize=(8, 6), dpi=200)
recursive_plt(e_dpths, e_n, lbl=get_decay_channel(300001), color="tab:red")
recursive_plt(x_300111, y_300111, lbl=get_decay_channel(300111), color="tab:purple")
recursive_plt(x_200011, y_200011, lbl=get_decay_channel(200011), color="tab:green")

for i in range(20):

    sampler = MCVariedMean(
        e_n,
        e_dpths,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, elec_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="tab:red",
        linestyle=":",
        zorder=20,
        linewidth=5,
        # label="Non Rebounding Sample Mean",
    )

    sampler = MCVariedMean(
        y_300111,
        x_300111,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, pi0_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="tab:purple",
        linestyle=":",
        zorder=20,
        linewidth=5,
        # label="Non Rebounding Sample Mean",
    )

    sampler = MCVariedMean(
        y_200011,
        x_200011,
        n_throws=400,
        hist_bins=30,
        sample_grammage=20,
    )
    mc_rms, sample_grm, sample_shwr, pi_rms = sampler.sampling_nmax_once(
        return_rms_dist=True
    )
    plt.plot(
        sample_grm,
        sample_shwr * mc_rms,
        color="tab:green",
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

plt.figure(figsize=(8, 6), dpi=200)
plt.hist(
    elec_rms / np.mean(elec_rms),
    bins=strict_bins,
    histtype="step",
    linewidth=4,
    alpha=1,
    label=get_decay_channel(300001),
    color="tab:red",
)


plt.hist(
    pi0_rms / np.mean(pi0_rms),
    bins=strict_bins,
    histtype="step",
    linewidth=4,
    alpha=1,
    label=get_decay_channel(300111),
    color="tab:purple",
)


plt.hist(
    pi_rms / np.mean(pi_rms),
    bins=strict_bins,
    histtype="step",
    linewidth=4,
    alpha=1,
    label=get_decay_channel(200011),
    color="tab:green",
)

plt.ylabel("# of Showers / 0.13 ", fontsize=16)
plt.xlabel(r"$ N_{at Xmax} / \overline{N_{at Xmax} } $", fontsize=16)
plt.legend()
