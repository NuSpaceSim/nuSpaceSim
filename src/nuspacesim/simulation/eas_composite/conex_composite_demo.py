import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex

from nuspacesim.simulation.eas_composite.plt_routines import decay_channel_mult_plt
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py
from comp_eas_utils import numpy_argmax_reduceat, get_decay_channel
from matplotlib.lines import Line2D
from nuspacesim.simulation.eas_composite.comp_eas_utils import (
    decay_channel_filter,
    slant_depth_to_alt,
)
from nuspacesim.simulation.eas_composite.comp_eas_conex import ConexCompositeShowers
from matplotlib.lines import Line2D


from scipy.signal import argrelextrema

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)


try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


def mean_shower(showers_n):
    average = np.nanmean(showers_n, axis=0)
    # test = average_composites  - comp_showers
    # take the square root of the mean of the difference between the average
    # and each particle content of each shower for one bin, squared
    rms_error = np.sqrt(np.mean((average - showers_n) ** 2, axis=0))

    # rms = np.sqrt(np.nanmean((showers_n) ** 2, axis=0))
    # std = np.nanstd(showers_n, axis=0)
    # err_in_mean = np.nanstd(showers_n, axis=0) / np.sqrt(
    #     np.sum(~np.isnan(showers_n), 0)
    # )
    rms_low = average - rms_error
    rms_high = average + rms_error
    return average, rms_error


tup_folder = "/home/fabg/g_drive/Research/NASA/Work/conex2r7_50-runs/"
# tup_folder = "C:/Users/144/Desktop/g_drive/Research/NASA/Work/conex2r7_50-runs"
# we can read in the showers with different primaries
elec_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_2033993834_11.root",
    )
)
pion_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_730702871_211.root",
    )
)
gamma_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_1722203790_22.root",
    )
)
# we can get the charged component for each type of particle initiated
elec_charged = elec_init.get_charged()
gamma_charged = gamma_init.get_charged()
pion_charged = pion_init.get_charged()
depths = elec_init.get_depths()

pids = [11, 22, 211]
init = [elec_charged, gamma_charged, pion_charged]
gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids, tau_table_start=5000)
comp_charged = gen_comp(return_table=False)

#%% segregate by decay channel and see most common one

decay_channels, shwrs_perchannel = np.unique(
    comp_charged[:, 1].astype("int"), return_counts=True
)
most_common_sort = np.flip(shwrs_perchannel.argsort())
decay_channels = decay_channels[most_common_sort]

shwrs_perchannel = shwrs_perchannel[most_common_sort]
branch_percent = shwrs_perchannel / np.sum(shwrs_perchannel)

# sum channels that contributed less than 3 percent to the decay
other_mask = branch_percent < 0.02
other_category = np.sum(shwrs_perchannel[other_mask])
decay_labels = [get_decay_channel(x) for x in decay_channels[~other_mask]]
decay_labels.append("other")


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300, figsize=(5, 4))
ax.pie(
    np.append(shwrs_perchannel[~other_mask], other_category),
    labels=decay_labels,
    autopct="%1.0f%%",
    # rotatelabels=True,
    startangle=75,
    pctdistance=0.8,
    radius=1.0,
    # labeldistance=None,
    textprops={"fontsize": 10},
)


ax.text(
    -0.2,
    0.88,
    r"${{\rm{:.0f}\:Composites}}$"
    "\n"
    r"${{\rm 100\:PeV}}$"
    "\n"
    r"${{\rm \beta\:=\:5\:\degree}}$".format(comp_charged.shape[0]),
    transform=ax.transAxes,
)

# plt.savefig(
#     "../../../../../g_drive/Research/NASA/composite_branching_ratio.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )

#!!! discovered a bug, having them 1000 each for a shower means the selection is
# skewed so that 1000 is used up for electron regardless
#%%

# get gh fits
elec_gh = elec_init.gh_fits()
gamma_gh = gamma_init.gh_fits()
pion_gh = pion_init.gh_fits()


gh_init = [elec_gh, gamma_gh, pion_gh]
gen_comp_gh = ConexCompositeShowers(
    shower_comps=gh_init, init_pid=pids, tau_table_start=5000
)
comp_gh = gen_comp_gh()

mean, rms_err = mean_shower(comp_charged[:, 2:])

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    dpi=300,
    figsize=(4, 3),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)
plt.subplots_adjust(hspace=0.05)

ax[0].plot(
    depths[0, :],
    np.log10(comp_charged[:, 2:].T),
    lw=1,
    color="tab:red",
    alpha=0.2,
    zorder=0,
)
ax[0].plot(
    depths[0, :],
    np.log10(comp_gh[:, 2:].T),
    lw=1,
    color="grey",
    alpha=0.2,
    zorder=0,
)
ax[0].plot(depths[0, :], np.log10(mean), lw=1, color="black", ls="--", zorder=12)

ax[1].plot(
    depths[0, :],
    (comp_gh[:, 2:] / comp_charged[:, 2:]).T,
    lw=1,
    color="tab:blue",
    alpha=0.1,
    zorder=1,
)
ax[1].axhline(y=1, ls="--", lw=1, color="k")

ax[0].set(
    xlim=(1, 6e3),
    ylim=(1, 8),
    ylabel=r"$\log_{10} \: N(X)$",
    # yscale="log",
)
ax[1].set(
    ylim=(0, 2.5),
    xlabel=r"$X\:{\rm (g \: cm^{-2})}$",
    ylabel=r"${\rm GH\:Fit/Actual}$",
)
custom_lines = [
    Line2D([0], [0], color="tab:red", lw=1),
    Line2D([0], [0], color="grey", lw=1),
    Line2D([0], [0], color="black", lw=1, ls="--"),
]
ax[0].legend(
    custom_lines,
    [
        r"${\rm Composite,\:100\:PeV,\:\beta = 5\degree}$",
        r"${\rm GH\:Fit}$",
        r"${\rm Mean\:Composite}$",
    ],
    fontsize=6,
    loc="upper center",
)
ax_twin = ax[0].twiny()
ax_twin.plot(depths[0, :], np.log10(mean), alpha=0)
ax_twin.set(xlim=(1, 6e3), xlabel=r"${\rmaltitude\:(km)}$")
ax_twin.set_xticklabels(
    list(
        np.round(
            slant_depth_to_alt(
                earth_emergence_ang=5, slant_depths=ax[0].get_xticks(), alt_stop=200
            ),
            1,
        ).astype("str")
    )
)


# plt.savefig(
#     "/home/fabg/g_drive/Research/NASA/gh_vs_profiles.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )

# fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(4, 3))
# ax.scatter(depths[0, :], slant_depth_to_alt(5, depths[0, :]), s=1)
# ax.set(xscale="log", yscale="log")
#%% filter out composites with subshowers

#!!! how to add stochastic process
subshwr_idx = []  # index of composite showers with subshowers
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(4, 3))
for i, s in enumerate(comp_charged):
    num_of_extrema = len(argrelextrema(np.log10(s), np.greater)[0])
    if num_of_extrema <= 2:
        # sub_showers = False
        ax.plot(depths[0, :], np.log10(s[2:]), lw=1, color="tab:blue", alpha=0.2)

    else:
        # sub_showers = True
        ax.plot(
            depths[0, :], np.log10(s[2:]), lw=1, color="tab:red", alpha=0.25, zorder=12
        )
        subshwr_idx.append(i)

ax.set(ylim=(0, 8), xlabel=r"${\rm slant\:depth\:(g \: cm^{-2})}$", ylabel=r"${\rm N}$")
ax_twin = ax.twiny()
ax_twin.plot(depths[0, :], np.log10(s[2:]), alpha=0)

ax_twin.set_xticklabels(
    list(
        np.round(
            slant_depth_to_alt(
                earth_emergence_ang=5, slant_depths=ax.get_xticks(), alt_stop=200
            ),
            1,
        ).astype("str")
    )
)
ax_twin.set(xlabel=r"${\rmaltitude\:(km)}$")


#%%

fig, ax = plt.subplots(nrows=2, ncols=1, dpi=300, figsize=(10, 5))
ax[0].pie(
    np.append(shwrs_perchannel[~other_mask], other_category),
    labels=decay_labels,
    autopct="%1.0f%%",
    # rotatelabels=True,
    startangle=75,
    pctdistance=0.8,
    radius=1.2,
    # labeldistance=None,
)
# ax.legend(
#     title=r"100 PeV Composites, $\beta = 5 \degree$",
#     ncol=2,
#     bbox_to_anchor=(0.08, 1.1),
#     fontsize=5,
# )

ax[0].text(
    -0.3,
    0.98,
    r"${{\rm{:.0f}\:Composites}}$"
    "\n"
    r"${{\rm 100\:PeV}}$"
    "\n"
    r"${{\rm \beta\:=\:5\:\degree}}$".format(comp_charged.shape[0]),
    transform=ax[0].transAxes,
)

subshwr_idx = np.array(subshwr_idx)
comp_sub = comp_charged[subshwr_idx]

decay_channels, shwrs_perchannel = np.unique(
    comp_sub[:, 1].astype("int"), return_counts=True
)
most_common_sort = np.flip(shwrs_perchannel.argsort())
decay_channels = decay_channels[most_common_sort]

shwrs_perchannel = shwrs_perchannel[most_common_sort]
branch_percent = shwrs_perchannel / np.sum(shwrs_perchannel)

# sum channels that contributed less than 3 percent to the decay
other_mask = branch_percent < 0.01
other_category = np.sum(shwrs_perchannel[other_mask])
decay_labels = [get_decay_channel(x) for x in decay_channels[~other_mask]]
decay_labels.append("other")


ax[1].pie(
    np.append(shwrs_perchannel[~other_mask], other_category),
    labels=decay_labels,
    autopct="%1.0f%%",
    # rotatelabels=True,
    startangle=0,
    pctdistance=0.8,
    # explode=[0, 0, 0, 0, 0, 0, 0.1],
    radius=1.2,
    # labeldistance=None,
)
# ax.legend(
#     title=r"100 PeV Composites, $\beta = 5 \degree$",
#     ncol=2,
#     bbox_to_anchor=(0.08, 1.1),
#     fontsize=5,
# )

ax[1].text(
    -0.3,
    0.98,
    r"${{\rm{:.0f}\:Composites, with\:sub-showers}}$" "\n".format(comp_sub.shape[0]),
    transform=ax[1].transAxes,
)