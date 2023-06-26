import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.plt_routines import decay_channel_mult_plt
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py
from comp_eas_utils import numpy_argmax_reduceat, get_decay_channel
from nuspacesim.simulation.eas_composite.x_to_z_lookup import depth_to_alt_lookup_v2
from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter
from nuspacesim.simulation.eas_composite.comp_eas_conex import ConexCompositeShowers
from matplotlib.lines import Line2D
from nuspacesim.simulation.eas_composite.mc_mean_shwr import MCVariedMean
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
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "font.size": 7,
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


def decay_filter(codes, nth_digit, rel, digit_flag):
    grouping = []
    for c in codes:
        l = [int(x) for x in str(c)]
        idx = nth_digit - 1
        if (rel == "leq") and (l[idx] <= digit_flag):
            grouping.append(c)
        elif (rel == "eq") and (l[idx] == digit_flag):
            grouping.append(c)
        elif rel == "geq" and (l[idx] >= digit_flag):
            grouping.append(c)

    return grouping


# load showers

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
init = [elec_charged, gamma_charged, pion_charged]
pids = [11, 22, 211]
#%%


decay_codes = get_decay_channel(None, just_codes=True)
gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids, tau_table_start=0)
comp_elec = gen_comp(n_comps=1000, channel=300001, return_table=False, no_subshwrs=True)

# decay codes with pi0
_, pi0codes = get_decay_channel(None, group_decay=(4, 1))
_, pichargedcodes = get_decay_channel(None, group_decay=(5, 1))


no_chargedpi = []
for c in pi0codes:
    l = [int(x) for x in str(c)]
    if l[4] == 0:
        no_chargedpi.append(c)
no_pi0 = []
for c in pichargedcodes:
    l = [int(x) for x in str(c)]
    if l[3] == 0:
        no_pi0.append(c)

singlekaoncodes = decay_filter(decay_codes, 2, "eq", 1)
singlepioncodes = decay_filter(decay_codes, 4, "eq", 1)

comp_no_chargedpi = gen_comp(
    n_comps=1000, channel=no_chargedpi, return_table=False, no_subshwrs=True
)
comp_no_pi0 = gen_comp(
    n_comps=1000, channel=no_pi0, return_table=False, no_subshwrs=True
)
comp_kaons = gen_comp(
    n_comps=1000, channel=singlekaoncodes, return_table=False, no_subshwrs=True
)
comp_single_pi = gen_comp(
    n_comps=1000, channel=singlepioncodes, return_table=False, no_subshwrs=True
)

shwr_groups = [
    comp_elec,
    comp_no_pi0,
    comp_no_chargedpi,
    comp_kaons,
    comp_single_pi,
]


decay_labels = [
    r"$e^{-}$",
    r"${\rm \pi^{+/-} \: no \:  \pi_0}$",
    r"${\rm \pi_0 \:  no  \: \pi^{+/-}}$",
    r"$1 K$",
    r"$1 \pi^{+/-}$",
]
#%%

cmap = plt.cm.get_cmap("inferno")(np.linspace(0, 1, 7))

nmax_scale_perchan = []
xmax_scale_perchan = []

mean_perchan = []
rms_err_perchan = []
mean_xmax_perchan = []

sample_label = "6000 g cm^{-2}"
# sample_label = "Nmax"
fig, ax = plt.subplots(nrows=1, ncols=3, dpi=400, figsize=(9, 3), sharex=True)
for ci, chnl in enumerate(shwr_groups):

    mean, rms_err = mean_shower(chnl[:, 2:])
    xmax_idx = np.argmax(mean)
    sample_idx = np.argmin(np.abs(depths[0, 2:] - 6000))  # xmax_idx

    sample_column = chnl[:, sample_idx]

    mean_val = depths[0, 2:][sample_idx]
    mean_perchan.append(mean)
    rms_err_perchan.append(rms_err)
    mean_xmax_perchan.append(mean_val)

    # xmaxs_idx = np.argmax(chnl[:, 2:], axis=1)
    # xmaxs_column = np.take(comp_charged[:, 2:], xmaxs_idx)

    # let's get the grammages where each shower peaks
    # shower_xmaxs = np.take(depths[0, :], xmaxs_idx)
    # xmax_multipliers = shower_xmaxs / mean_xmax

    rms_dist = sample_column / mean[sample_idx]

    bin_end = 2.8  # np.round(np.max(xmax_column / mean[xmax_idx]), 0)
    hist_bins = np.linspace(0, bin_end, 30)

    # histogram from x max
    # cts, bin_edges = np.histogram(xmax_column / mean[xmax_idx], bins=hist_bins)

    cts, bin_edges = np.histogram(rms_dist, bins=hist_bins, density=True)
    bin_ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2

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
    n_samples = chnl[:, 2:].shape[0]

    rand_nmax_scale = []
    # while is not good, not sure how to approach other way
    while len(rand_nmax_scale) != n_samples:
        r = exponnorm.rvs(1 / (lamb * sig), loc=mu, scale=sig)
        if (r > 0) and (r <= bin_end):
            rand_nmax_scale.append(r)

    nmax_scale_perchan.append(rand_nmax_scale)

    # rand_xmax_scale = []
    # while len(rand_xmax_scale) != n_samples:

    #     r = exponnorm.rvs(1 / (xmax_lamb * xmax_sig), loc=xmax_mu, scale=xmax_sig)
    #     if (r > 0) and (r <= bin_end):
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
        label="$({:.2f}, {:.1f}, {:.1f}, {:.1f})$".format(chi2, *params),
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
        label="{} scaling values".format(len(rand_nmax_scale)),
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
ax[0].legend(title=r"${\rm Decay\:Channel}$")
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
    loc="upper right",
    title=r"${\rm Fit}(\chi^2,\:\lambda,\:\sigma,\:\mu)$",
)

ax[2].legend(loc="upper right", title=r"${\rm Resampled}$")
#%%
fig, ax = plt.subplots(
    nrows=1,
    ncols=3,
    dpi=300,
    figsize=(7.5, 2.5),
    sharey=True,
)
plt.subplots_adjust(wspace=0)

# inset = ax[0].inset_axes([0, -0.85, 3, 0.70], transform=ax[0].transAxes)


recomean_actualmean = []

for ci, chnl in enumerate(shwr_groups):

    ax[0].plot(
        depths[0, :],
        chnl[:, 2:].T,
        lw=1,
        color=cmap[ci],
        alpha=0.2,
    )
    # just to include the label
    ax[0].plot(
        depths[0, :],
        chnl[0, 2:],
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
        label=r"${{\rm n_{{showers}} = {:.0f}\:}}$".format(chnl.shape[0]),
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
    loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, title=r"${\rm Groupings}$"
)
ax[1].legend(
    loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, title=r"${\rm Mean\:and\:RMS}$"
)
ax[2].legend(
    title=r"${\rm Reconstructed\:Showers}$",
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
    ncol=2,
)

ax[0].set(
    yscale="log",
    ylabel="$N$",
    ylim=(1, 5e8),
)
ax[1].set(xlabel=r"${\rm\:slant\:depth\:(g \: cm^{-2})}$")
