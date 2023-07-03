import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.plt_routines import decay_channel_mult_plt
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py
from comp_eas_utils import numpy_argmax_reduceat, get_decay_channel

from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter
from nuspacesim.simulation.eas_composite.comp_eas_conex import ConexCompositeShowers
from nuspacesim.simulation.eas_composite.comp_eas_utils import slant_depth_to_alt
from matplotlib.lines import Line2D

from scipy.signal import argrelextrema
from scipy.stats import poisson
from scipy.stats import skewnorm
import scipy.special as sse
from scipy import stats
from scipy.stats import exponnorm
import matplotlib

cmap = matplotlib.cm.get_cmap("Spectral")

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


def gauss_exp(x, l, s, m):
    return (
        0.5
        * l
        * np.exp(0.5 * l * (2 * m + l * s * s - 2 * x))
        * sse.erfc((m + l * s * s - x) / (np.sqrt(2) * s))
    )  # exponential gaussian


def gauss(x, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def no_subshwrs(comp_charged):
    no_subshwr_idx = []  # index of composite showers with subshowers
    for i, s in enumerate(comp_charged):
        num_of_extrema = len(argrelextrema(np.log10(s), np.greater)[0])
        if num_of_extrema <= 2:
            # sub_showers = False
            # ax.plot(depths[0, :], s[2:], lw=1, color="tab:blue", alpha=0.2)
            no_subshwr_idx.append(i)
        else:
            # sub_showers = True
            # ax.plot(depths[0, :], s[2:], lw=1, alpha=0.25, zorder=12)
            pass
    no_subshwr_idx = np.array(no_subshwr_idx)
    return comp_charged[no_subshwr_idx]


beta = 5
log_e = 17
total_gen = 1000
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
gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids, tau_table_start=0)
comp_charged = gen_comp(n_comps=total_gen, no_subshwrs=True)
# filter out composites with subshowers
# comp_charged = no_subshwrs(comp_charged)

#!!! how to add stochastic process
# decay_labels
# fluctuate by decay channel

decay_channels, shwrs_perchannel = np.unique(
    comp_charged[:, 1].astype("int"), return_counts=True
)
most_common_sort = np.flip(shwrs_perchannel.argsort())
decay_channels = decay_channels[most_common_sort]
shwrs_perchannel = shwrs_perchannel[most_common_sort]
decay_labels = [get_decay_channel(x) for x in decay_channels]
branch_percent = shwrs_perchannel / np.sum(shwrs_perchannel)

# cmap = plt.cm.get_cmap("inferno")(np.linspace(0, 1, 25))

# fig, ax = plt.subplots(
#     nrows=4, ncols=5, dpi=300, figsize=(12, 10), sharex=True, sharey=True
# )
# fig.subplots_adjust(hspace=0.05, wspace=0.05)
# ax = ax.ravel()

# for i, decay in enumerate(decay_channels):
#     y = comp_charged[comp_charged[:, 1] == decay]
#     x = depths[0, :]
#     print(i)
#     if i < 19:
#         ax[i].plot(x, np.log10(y[:, 2:].T), color=cmap[i], lw=1, alpha=0.60)

#         t = decay_labels[i] + "\n" + r"${{\rm \: {} \: EAS}}$".format(y.shape[0])
#         ax[i].text(
#             0.95,
#             0.96,
#             t,
#             transform=ax[i].transAxes,
#             horizontalalignment="right",
#             verticalalignment="top",
#         )
#         ax[i].grid(ls=":")
#         if i <= 4:
#             ax_twin = ax[i].twiny()
#             ax_twin.plot(x, np.log10(y[0, 2:]), alpha=0)
#             ax_twin.set_xticklabels(
#                 list(
#                     np.round(
#                         slant_depth_to_alt(
#                             earth_emergence_ang=beta,
#                             slant_depths=ax[i].get_xticks(),
#                             alt_stop=200,
#                         ),
#                         1,
#                     ).astype("str")
#                 )
#             )

#     else:
#         ax[19].plot(
#             x,
#             np.log10(y[0, 2:].T),
#             color=cmap[i],
#             lw=1,
#             alpha=1,
#             label=decay_labels[i],
#         )


# ax[0].set(ylim=(1.5, 8))
# ax[19].legend(frameon=False)
# ax[19].grid(ls=":")
# fig.text(0.5, 0.09, r"${\rm slant\:depth\:(g \: cm^{-2})}$", ha="center")
# fig.text(0.5, 0.91, r"${\rmaltitude\:(km)}$", ha="center")
# fig.text(
#     0.10,
#     0.91,
#     r"${{\rm\:\beta ={}\degree,E_{{primary}}=\:10^{{{}}}\:eV,{}\:total\:Composite\:EAS}}$".format(
#         beta, log_e, total_gen
#     ),
#     ha="left",
# )
# fig.text(0.1, 0.5, r"$\log_{10}\: N$", va="center", rotation="vertical")

# plt.savefig(
#     "../../../../../g_drive/Research/NASA/eas_gallery.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )


# sum channels that contributed less than 3 percent to the decay
other_mask = branch_percent < 0.18
other_category = np.sum(shwrs_perchannel[other_mask])
decay_labels = [get_decay_channel(x) for x in decay_channels[~other_mask]]
decay_labels.append(r"${\rm other}$")
decay_codes = list(decay_channels[~other_mask])
decay_codes.append(decay_channels[other_mask])
shwrs_perchannel = np.append(shwrs_perchannel[~other_mask], other_category)

nmax_scale_perchan = []
xmax_scale_perchan = []

mean_perchan = []
rms_err_perchan = []
mean_xmax_perchan = []
cmap = plt.cm.get_cmap("inferno")(np.linspace(0, 1, 5))[1:]
fig, ax = plt.subplots(
    nrows=1,
    ncols=3,
    dpi=300,
    figsize=(10, 3),
    sharey=True,
)

for ci, chnl in enumerate(decay_codes):

    if ci == len(decay_codes) - 1:
        cc = comp_charged[np.isin(comp_charged[:, 1], chnl)]
    else:
        cc = comp_charged[comp_charged[:, 1] == chnl]

    mean, rms_err = mean_shower(cc[:, 2:])
    xmax_idx = np.argmax(mean)
    sample_idx = xmax_idx  # np.argmin(np.abs(depths[0, 2:] - 6000))

    sample_column = cc[:, sample_idx]

    mean_val = depths[0, 2:][sample_idx]
    mean_perchan.append(mean)
    rms_err_perchan.append(rms_err)
    mean_xmax_perchan.append(mean_val)

    # xmax_column = cc[:, xmax_idx]
    # xmaxs_idx = np.argmax(cc[:, 2:], axis=1)
    # xmaxs_column = np.take(comp_charged[:, 2:], xmaxs_idx)

    # let's get the grammages where each shower peaks
    # shower_xmaxs = np.take(depths[0, :], xmaxs_idx)
    # xmax_multipliers = shower_xmaxs / mean_xmax
    rms_dist = sample_column / mean[xmax_idx]

    bin_end = 3  # np.round(np.max(xmax_column / mean[xmax_idx]), 0)
    hist_bins = np.linspace(0, bin_end, 25)

    # histogram from x max
    # cts, bin_edges = np.histogram(xmax_column / mean[xmax_idx], bins=hist_bins)

    cts, bin_edges = np.histogram(rms_dist, bins=hist_bins, density=True)
    bin_ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2
    print(cts)
    # xmaxs_cts, xmaxs_edges = np.histogram(
    #     xmax_multipliers, bins=np.linspace(0.5, 1.5, 25), density=True
    # )
    # xmax_bin_ctrs = (xmaxs_edges[:-1] + xmaxs_edges[1:]) / 2
    # xmaxdist_params, pcov = curve_fit(gauss_exp, xmax_bin_ctrs, xmaxs_cts)
    # xmax_lamb = xmaxdist_params[0]
    # xmax_sig = xmaxdist_params[1]
    # xmax_mu = xmaxdist_params[2]

    # cts = dens_cts
    # ax[1].errorbar(
    #     bin_ctrs, cts, (dens_cts / cts)[0] * np.sqrt(cts), fmt=".", color="black"
    # )

    # fit the nmax distribution
    params, pcov = curve_fit(gauss_exp, bin_ctrs, cts)
    lamb = params[0]
    sig = params[1]
    mu = params[2]

    nonzero_mask = cts > 0
    chi2 = np.sum(
        (cts[nonzero_mask] - gauss_exp(bin_ctrs, *params)[nonzero_mask]) ** 2
        / gauss_exp(bin_ctrs, *params)[nonzero_mask]
    )
    p_value = stats.chi2.sf(chi2, len(cts[nonzero_mask]))
    reduced_ch2 = chi2 / len(cts)

    # plot the theoretical fit, but 1 + the end
    theory_x = np.linspace(0, bin_end + 0.5, 200)

    # let's loop so that we can control the actual nuber of samples, not just mask it away
    n_samples = cc[:, 2:].shape[0]

    rand_nmax_scale = []
    # while is not good, not sure how to approach other way
    while len(rand_nmax_scale) != n_samples:
        r = exponnorm.rvs(1 / (lamb * sig), loc=mu, scale=sig)
        if r > 0:
            rand_nmax_scale.append(r)

    nmax_scale_perchan.append(rand_nmax_scale)

    # rand_xmax_scale = []
    # while len(rand_xmax_scale) != n_samples:

    #     r = exponnorm.rvs(1 / (xmax_lamb * xmax_sig), loc=xmax_mu, scale=xmax_sig)
    #     if r > 0:
    #         rand_xmax_scale.append(r)

    # xmax_scale_perchan.append(rand_xmax_scale)

    ax[0].hist(
        rms_dist,
        bins=hist_bins,
        color=cmap[ci],
        histtype="step",
        lw=3,
        label=decay_labels[ci],
        alpha=0.90,
    )

    ax[1].hist(
        rms_dist,
        bins=hist_bins,
        color=cmap[ci],  # "tab:red",
        histtype="step",
        density=True,
        lw=2,
        # hatch="///",
        # fill=True,
        # label=decay_labels[ci],
        alpha=0.5,
    )
    ax[1].plot(
        theory_x,
        gauss_exp(theory_x, *params),
        ls="--",
        color=cmap[ci],  # "tab:red",
        lw=2,
        # label=r"Prob($\chi^2$, dof) = {:.2f}".format(p_value),
        # label=decay_labels[ci],
        alpha=0.9,
        label="$({:.2f}, {:.1f}, {:.1f}, {:.1f})$".format(chi2, lamb, sig, mu),
    )

    ax[2].hist(
        rand_nmax_scale,
        bins=hist_bins,
        color=cmap[ci],
        histtype="step",
        # hatch="\\\\",
        lw=3,
        density=True,
        alpha=0.9,
        label=r"${{\rm {} \:scaling\:values}}$".format(len(rand_nmax_scale)),
    )

    # =============================================================================
    #     ax[1].hist(
    #         xmax_multipliers,
    #         bins=np.linspace(0.5, 1.5, 25),
    #         color="tab:red",
    #         histtype="step",
    #         density=True,
    #         lw=1.5,
    #         hatch="///",
    #     )
    #     ax[1].plot(
    #         np.linspace(0.5, 1.5, 100),
    #         gauss_exp(np.linspace(0.5, 1.5, 100), *xmaxdist_params),
    #         ls="--",
    #         color="tab:red",
    #         lw=3,
    #         # label=r"Prob($\chi^2$, dof) = {:.2f}".format(p_value),
    #         # label=r"$(\mu, \sigma, a)$"
    #         # "\n"
    #         # r"$({:.2f}, {:.2f}, {:.2f})$".format(*xmaxdist_params),
    #     )
    #     ax[1].hist(
    #         rand_xmax_scale,
    #         bins=np.linspace(0.5, 1.5, 25),
    #         color="tab:blue",
    #         histtype="step",
    #         hatch="\\\\",
    #         lw=1.5,
    #         density=True,
    #         label=r"${\rm resampled}$",
    #     )
    # =============================================================================

ax[0].set(
    ylabel="Raw Counts",
    # xlabel=r"${\rm shower\:N_{max} \: / \: mean \: N_{max}}$",
    # ylim=(0, 245),
)
sample_label = r"N_{\rm max}"
ax[0].legend(title=r"${\rm Decay\:Channel}$", fontsize=8)
ax[1].set(
    xlabel=r"${{\rm shower\:{} \: / \: mean \: {} }}$".format(
        sample_label, sample_label
    ),
    ylabel="PDF",
    ylim=(0, 1.5),
)

ax[2].set(
    # xlim=(theory_x.min(), 3),
    # xlabel=r"${\rm shower\:X_{max} \: / \: mean \: X_{max}}$",
    # xlabel=r"${\rm shower\:N_{max} \: / \: mean \: N_{max}}$",
    ylabel="PDF",
    ylim=(0, 1.5),
)

ax[1].legend(
    loc="upper right", title=r"${\rm Fit}(\chi^2,\:\lambda,\:\sigma,\:\mu)$", fontsize=8
)

ax[2].legend(loc="upper right", title=r"${\rm Resampled}$", fontsize=8)

plt.savefig(
    "../../../../../g_drive/Research/NASA/rms_common.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
#%%
fig, ax = plt.subplots(
    nrows=3,
    ncols=1,
    dpi=300,
    figsize=(5, 8),
    sharey=True,
)
plt.subplots_adjust(hspace=0)

recomean_actualmean = []

for ci, chnl in enumerate(decay_codes):

    if ci == len(decay_codes) - 1:
        cc = comp_charged[np.isin(comp_charged[:, 1], chnl)]
    else:
        cc = comp_charged[comp_charged[:, 1] == chnl]

    ax[0].plot(
        depths[0, :],
        cc[:, 2:].T,
        lw=1,
        color=cmap[ci],
        alpha=0.1,
    )
    # just to include the label
    ax[0].plot(
        depths[0, :],
        cc[0, 2:],
        lw=1,
        color=cmap[ci],
        alpha=1,
        label=decay_labels[ci],
    )

    ax[1].plot(
        depths[0, :],
        mean_perchan[ci],
        alpha=1,
        color=cmap[ci],
        ls="--",
        label=r"${{\rm n_{{showers}} = {:.0f}\:}}$".format(cc.shape[0]),
    )
    ax[1].fill_between(
        depths[0, :],
        mean_perchan[ci] - rms_err_perchan[ci],
        mean_perchan[ci] + rms_err_perchan[ci],
        color=cmap[ci],
        alpha=0.5,
        # hatch="////",
        zorder=5,
        # label=r"${\rm RMS\:Error}$",
    )

    #!!! how to add variations in xmax without extending tails
    fluctuated_mean = mean_perchan[ci] * np.array(nmax_scale_perchan[ci])[:, np.newaxis]
    fluctuated_bins = depths[0, :]  # * np.array(rand_xmax_scale)[:, np.newaxis]
    # reco_ax = ax.inset_axes([0, -2.4, 1, 1])

    ax[2].plot(
        fluctuated_bins.T,
        fluctuated_mean.T,
        lw=1,
        color=cmap[ci],
        alpha=0.2,
    )

    reco_mean, _ = mean_shower(fluctuated_mean)

    recomean_actualmean.append(reco_mean / mean_perchan[ci])

    ax[2].plot(
        depths[0, :],
        fluctuated_mean[0, :],
        lw=1,
        color=cmap[ci],
        alpha=0.2,
        label="${:.2f}$".format(np.max(reco_mean / mean_perchan[ci])),
    )


ax[0].axvline(depths[0, 2:][sample_idx], ls="--", lw=2, color="grey")
ax[0].legend(
    loc="upper right",
    # bbox_to_anchor=(0.5, 1),
    # title=r"${\rm Groupings}$",
)
ax[1].legend(
    loc="upper right",
    # bbox_to_anchor=(0.5, 1),
    # title=r"${\rm Mean\:and\:RMS}$",
)
ax[2].legend(
    # title=r"${\rm Reconstructed\:Showers}$",
    loc="upper right",
    # bbox_to_anchor=(0.5, 1),
)

ax[1].set(
    yscale="log",
    ylabel="$N$",
    ylim=(15, 2e8),
)
ax[2].set(xlabel=r"${\rm\:slant\:depth\:(g \: cm^{-2})}$")

plt.savefig(
    "../../../../../g_drive/Research/NASA/reco_common.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
