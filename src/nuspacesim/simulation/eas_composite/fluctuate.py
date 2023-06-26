import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from comp_eas_utils import get_decay_channel
from nuspacesim.simulation.eas_composite.x_to_z_lookup import depth_to_alt_lookup_v2
from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter
from nuspacesim.simulation.eas_composite.comp_eas_conex import ConexCompositeShowers

from nuspacesim.simulation.eas_composite.mc_mean_shwr import MCVariedMean
from scipy.signal import argrelextrema
from scipy.stats import poisson
from scipy.stats import skewnorm
import scipy.special as sse
from scipy import stats
from scipy.stats import exponnorm

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


#%% load showers

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
gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids, tau_table_start=6000)
comp_charged = gen_comp()


#%% filter out composites with subshowers

#!!! how to add stochastic process

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
comp_charged = comp_charged[no_subshwr_idx]
# comp_sub = comp_charged[~subshwr_idx]
#%% sampling just the xmax, with no sub showers

mean, rms_err = mean_shower(comp_charged[:, 2:])
xmax_idx = np.argmax(mean)
mean_xmax = depths[0, 2:][xmax_idx]

xmax_column = comp_charged[:, xmax_idx]
xmaxs_idx = np.argmax(comp_charged[:, 2:], axis=1)
# xmaxs_column = np.take(comp_charged[:, 2:], xmaxs_idx)

# let's get the grammages where each shower peaks
shower_xmaxs = np.take(depths[0, :], xmaxs_idx)
xmax_multipliers = shower_xmaxs / mean_xmax
rms_dist = xmax_column / mean[xmax_idx]

bin_end = 3  # np.round(np.max(xmax_column / mean[xmax_idx]), 0)
hist_bins = np.linspace(0, bin_end, 25)

# histogram from x max
# cts, bin_edges = np.histogram(xmax_column / mean[xmax_idx], bins=hist_bins)

cts, bin_edges = np.histogram(rms_dist, bins=hist_bins, density=True)
bin_ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2

xmaxs_cts, xmaxs_edges = np.histogram(
    xmax_multipliers, bins=np.linspace(0.5, 1.5, 25), density=True
)
xmax_bin_ctrs = (xmaxs_edges[:-1] + xmaxs_edges[1:]) / 2
gauss_params, pcov = curve_fit(gauss, xmax_bin_ctrs, xmaxs_cts)

#!!! how to add variations in xmax without extending tails

# cts = dens_cts
# ax[1].errorbar(
#     bin_ctrs, cts, (dens_cts / cts)[0] * np.sqrt(cts), fmt=".", color="black"
# )

# fit it
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
n_samples = comp_charged[:, 2:].shape[0]
rand_nmax_scale = []
# while is not good, not sure how to approach other way
while len(rand_nmax_scale) != n_samples:
    r = exponnorm.rvs(1 / (lamb * sig), loc=mu, scale=sig)
    if r > 0:
        rand_nmax_scale.append(r)

rand_xmax_scale = []
while len(rand_xmax_scale) != n_samples:
    r = np.random.normal(loc=gauss_params[0], scale=gauss_params[1])
    if r > 0:
        rand_xmax_scale.append(r)


fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(6, 3))
# fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(4, 3))
ax[0].hist(
    rms_dist,
    bins=hist_bins,
    color="tab:red",
    histtype="step",
    density=True,
    lw=1.5,
    hatch="///",
)
ax[0].plot(
    theory_x,
    gauss_exp(theory_x, *params),
    ls="--",
    color="tab:red",
    lw=3,
    # label=r"Prob($\chi^2$, dof) = {:.2f}".format(p_value),
    label=r"$(\chi_\nu^2,\:\lambda,\:\sigma,\:\mu)$"
    "\n"
    r"$({:.2f}, {:.2f}, {:.2f}, {:.2f})$".format(reduced_ch2, *params),
)

ax[0].hist(
    rand_nmax_scale,
    bins=hist_bins,
    color="tab:blue",
    histtype="step",
    hatch="\\\\",
    lw=1.5,
    density=True,
    label=r"${\rm resampled}$",
)


ax[0].set(
    xlim=(theory_x.min(), 3),
    xlabel=r"${\rm shower\:N_{max} \: / \: mean \: N_{max}}$",
    ylabel="PDF",
)
ax[0].legend()

ax[1].hist(
    xmax_multipliers,
    bins=np.linspace(0.5, 1.5, 25),
    color="tab:red",
    histtype="step",
    density=True,
    lw=1.5,
    hatch="///",
)
ax[1].plot(
    np.linspace(0.5, 1.5, 100),
    gauss(np.linspace(0.5, 1.5, 100), *gauss_params),
    ls="--",
    color="tab:red",
    lw=3,
    # label=r"Prob($\chi^2$, dof) = {:.2f}".format(p_value),
    label=r"$(\mu, \sigma, a)$"
    "\n"
    r"$({:.2f}, {:.2f}, {:.2f})$".format(*gauss_params),
)
ax[1].hist(
    rand_xmax_scale,
    bins=np.linspace(0.5, 1.5, 25),
    color="tab:blue",
    histtype="step",
    hatch="\\\\",
    lw=1.5,
    density=True,
    label=r"${\rm resampled}$",
)
ax[1].set(
    xlim=(theory_x.min(), 3),
    xlabel=r"${\rm shower\:X_{max} \: / \: mean \: X_{max}}$",
    ylabel="PDF",
)
ax[1].legend()
#%%
fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(6, 3))
ax[0].plot(depths[0, :], comp_charged[:, 2:].T, lw=1, color="tab:red", alpha=0.2)
ax[0].set(
    yscale="log",
    ylim=(1, 1e8),
    ylabel="N",
    xlabel=r"${\rm\:slant\:depth\:(g/cm^2)}$",
    xlim=(0, 2000),
)
ax[0].plot(
    depths[0, :],
    mean,
    "k",
    alpha=1,
    label=r"${{\rm \:Mean\:of\:{:.0f}\:composites}}$".format(comp_charged.shape[0]),
)
ax[0].fill_between(
    depths[0, :],
    mean - rms_err,
    mean + rms_err,
    facecolor="black",
    alpha=0.5,
    # hatch="////",
    zorder=5,
    label=r"${\rm RMS\:Error}$",
)
ax[0].axvline(mean_xmax, ls="--", lw=2, color="black")
ax[0].legend(loc="lower center")


fluctuated_mean = mean * np.array(rand_nmax_scale)[:, np.newaxis]
fluctuated_bins = depths[0, :] * np.array(rand_xmax_scale)[:, np.newaxis]
# reco_ax = ax.inset_axes([0, -2.4, 1, 1])

ax[1].plot(
    fluctuated_bins.T,
    fluctuated_mean.T,
    lw=1,
    color="tab:blue",
    alpha=0.5,
)
ax[1].plot(
    depths[0, :],
    fluctuated_mean[0, :],
    lw=1,
    color="tab:blue",
    alpha=0.5,
    label=r"${\rm Mean\:Fluctuated,\:Reconstructed}$",
)

ax[1].fill_between(
    depths[0, :],
    mean - rms_err,
    mean + rms_err,
    facecolor="black",
    alpha=0.5,
    # hatch="////",
    zorder=5,
    # label="RMS"
)
ax[1].set(
    yscale="log",
    ylim=(1, 1e8),
    xlabel=r"${\rm\:slant\:depth\:(g/cm^2)}$",
    ylabel="N",
    xlim=(0, 2000),
)
ax[1].legend(loc="lower center")
#%% fluctuate by decay channel

decay_channels, shwrs_perchannel = np.unique(
    comp_charged[:, 1].astype("int"), return_counts=True
)
most_common_sort = np.flip(shwrs_perchannel.argsort())
decay_channels = decay_channels[most_common_sort]

shwrs_perchannel = shwrs_perchannel[most_common_sort]

branch_percent = shwrs_perchannel / np.sum(shwrs_perchannel)

# sum channels that contributed less than 3 percent to the decay
other_mask = branch_percent < 0.05
other_category = np.sum(shwrs_perchannel[other_mask])
decay_labels = [get_decay_channel(x) for x in decay_channels[~other_mask]]
decay_labels.append("other")
decay_codes = decay_channels[~other_mask]
shwrs_perchannel = np.append(shwrs_perchannel[~other_mask], other_category)
