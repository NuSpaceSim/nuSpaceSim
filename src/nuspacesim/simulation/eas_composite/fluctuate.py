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


def gaus(x, mu, sigma, amp):
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
gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids)
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
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(4, 3))
ax.plot(depths[0, :], comp_charged[:, 2:].T, lw=1, color="tab:red", alpha=0.2)
ax.set(yscale="log", ylim=(1, 1e8))

mean, rms_err = mean_shower(comp_charged[:, 2:])
xmax_idx = np.argmax(mean)
xmax_column = comp_charged[:, xmax_idx]
xmax = depths[0, 2:][xmax_idx]
bin_end = np.round(np.max(xmax_column / mean[xmax_idx]), 0)
hist_bins = np.linspace(0, bin_end, 25)


ax.plot(depths[0, :], mean, "k", alpha=1)
ax.fill_between(
    depths[0, :],
    mean - rms_err,
    mean + rms_err,
    facecolor="grey",
    alpha=0.5,
    # hatch="////",
    zorder=5,
    label="RMS Error",
)
ax.axvline(xmax, ls="--", color="grey")

dis_ax = ax.inset_axes([1.1, 0, 1, 1])

# histogram
cts, bin_edges = np.histogram(xmax_column / mean[xmax_idx], bins=hist_bins)
bin_ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2
dis_ax.hist(
    xmax_column / mean[xmax_idx],
    bins=hist_bins,
    color="tab:red",
    histtype="step",
    density=True,
)


cts, _ = np.histogram(xmax_column / mean[xmax_idx], bins=hist_bins, density=True)
# cts = dens_cts
# dis_ax.errorbar(
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
dis_ax.plot(
    theory_x,
    gauss_exp(theory_x, *params),
    ls="--",
    color="grey",
    # label=r"Prob($\chi^2$, dof) = {:.2f}".format(p_value),
    label=r"$ {:.2f}, {:.2f}, {:.2f}, {:.2f}$".format(reduced_ch2, *params),
)

dis_ax.set(xlim=(theory_x.min(), 3))
dis_ax.legend(title=r"$(\chi_\nu^2,\:\lambda,\:\sigma,\:\mu,\:{\rm amp})$")
#%% sample dist

r = exponnorm.rvs(1 / (lamb * sig), size=1000)


#%%
# filter based on common decay channels check distributions
channels = [300001, 300111, 200011]
labels = []
filt_shwrs = []
nmax_dist = []
mean_showers = []
rms_error_shower = []
multipliers = []
fluctuated = []

sample_grammage = 6000

for dc in channels:
    l = get_decay_channel(dc)
    _, filtered_n = decay_channel_filter(comp_charged, comp_charged, dc)
    filt_shwrs.append(filtered_n)
    labels.append(l)

    sample = MCVariedMean(
        filtered_n,
        depths[: filtered_n.shape[0], :],
        n_throws=100,
        hist_bins=30,
        sample_grammage=100,
    )
    mean, rms_error = mean_shower(filtered_n)
    mean_showers.append(mean[2:])
    rms_error_shower.append(rms_error[2:])
    _, _, _, dist = sample.sampling_nmax_once(return_rms_dist=True)
    # _, _, _, dist = sample.sample_specific_grammage(
    #     grammage=sample_grammage, return_rms_dist=True
    # )
    nmax_dist.append(dist)

    # for each comman decay channel, fluctuate it a lot
    mult = np.zeros(filtered_n.shape[0])

    fluctuated_per_channel = []

    for m, r in enumerate(mult):
        mc_rms_multiplier, _, _ = sample.sampling_nmax_once(return_rms_dist=False)
        # mc_rms_multiplier, _, _ = sample.sample_specific_grammage(
        #     grammage=sample_grammage, return_rms_dist=False
        # )
        mult[m] = mc_rms_multiplier
        fluctuated_per_channel.append(mean * mc_rms_multiplier)

    multipliers.append(mult)
    fluctuated.append(np.array(fluctuated_per_channel))


# def poisson_function(k, lamb):
#     # The parameter lamb will be used as the fit parameter
#     return poisson.pmf(k, lamb)

dist_params = []
rand_multipliers = []
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(4, 3))

bin_end = np.round(np.max(dist / np.mean(dist)), 0)
hist_bins = np.linspace(0, bin_end, 22)

for i, dist in enumerate(nmax_dist):

    cts, bin_edges = np.histogram(
        dist / np.mean(dist),
        bins=hist_bins,  # density=True
    )
    # if poisson noise is to be used to asses the quality of the fit,
    # density needs to be false

    bin_ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2

    params, pcov = curve_fit(gauss_exp, bin_ctrs, cts)
    gaus_params, gaus_pcov = curve_fit(gaus, bin_ctrs, cts)
    nonzero_mask = cts > 0
    print(cts)
    chi2 = np.sum(
        (cts[nonzero_mask] - gauss_exp(bin_ctrs, *params)[nonzero_mask]) ** 2
        / np.sqrt(cts)[nonzero_mask] ** 2
    )
    p_value = stats.chi2.sf(chi2, len(cts[nonzero_mask]))
    print(chi2)
    reduced_ch2 = chi2 / len(cts)

    # plot the theoretical fit, but 1 + the end
    ax.plot(
        np.linspace(0, bin_end + 0.5, 200),
        gauss_exp(np.linspace(0, bin_end + 0.5, 200), *params),
        # label=r"Prob($\chi^2$, dof) = {:.2f}".format(p_value),
        label=r"$\chi_\nu^2$ = {:.2f}".format(reduced_ch2),
    )
    dist_params.append(params)
    r = exponnorm.rvs(params[0] * params[1], size=1000)

    ax.hist(
        dist / np.mean(dist),
        alpha=0.5,
        # edgecolor="black",
        linewidth=0.5,
        label=labels[i],
        bins=hist_bins,
        histtype="step",
        lw=2,
        # density=True,
    )

    # ax.plot(np.linspace(0, 4, 200), gaus(np.linspace(0, 4, 200), *gaus_params), ls="--")

    ax.errorbar(bin_ctrs, cts, color="k", fmt=".", yerr=np.sqrt(cts))

    rand_multipliers.append(r[r > 0])

ax.legend(
    title="Composite Conex, Charged Component",
    ncol=2,
    bbox_to_anchor=[1, 1.4],
    # loc="left",
)
ax.set(
    # xlabel=f"sampled at {sample_grammage} g/cm$^{2}$",
    xlabel="sampled at Xmax ",
    ylabel="Number of Showers",
    # yscale="log",
)


# plt.savefig(
#     os.path.join(
#         "G:", "My Drive", "Research", "NASA", "full_conex_modeling", "rms_nmx_dist.pdf"
#     ),
#     bbox_inches="tight",
# )
#%%

fig, ax = plt.subplots(
    nrows=1, ncols=3, dpi=300, figsize=(9, 3), sharey=True, sharex=True
)
plt.subplots_adjust(wspace=0)
ax = ax.ravel()
c = ["tab:blue", "tab:orange", "tab:green"]
for i, l in enumerate(filt_shwrs):

    ax[i].plot(depths[: l.shape[0], :].T, l.T[2:], color=c[i], alpha=0.25)
    ax[i].plot(depths[:1, :].T, l[0, 2:], color=c[i], alpha=0.25, label=labels[i])

    ax[i].fill_between(
        depths[i, :],
        mean_showers[i] - rms_error_shower[i],
        mean_showers[i] + rms_error_shower[i],
        facecolor="grey",
        alpha=0.5,
        hatch="////",
        zorder=5,
        label="RMS Error",
    )

    ax[i].plot(depths[i, :], mean_showers[i], "k", label="Mean")
    ax[i].set(xlabel="Slant Depth (g cm$^{-2}$)")
    ax[i].set(ylim=(1, 9e7))
    ax[i].legend(title=f"{l.shape[0]} Showers")
ax[0].set(ylabel="N", yscale="log")

#%%
for i, multiplier in enumerate(rand_multipliers):
    nshowers = len(multiplier)
    for i2, m in enumerate(multiplier):

        if i2 == 0:

            ax[i + 3].plot(
                depths[i, :],
                mean_showers[i] * m,
                color=c[i],
                alpha=0.25,
                label="{} showers".format(nshowers),
            )
        else:
            ax[i + 3].plot(
                depths[i, :],
                mean_showers[i] * m,
                color=c[i],
                alpha=0.25,
            )

    ax[i + 3].plot(depths[i, :], mean_showers[i], "k", label="Mean")
    ax[i + 3].legend()
    ax[i + 3].set(xlabel="Slant Depth (g cm$^{-2}$)")

ax[4].set_title("Sampled from Guassian with Exponential Tail Distribution")
#%%
# for i, l in enumerate(fluctuated):

#     ax[i + 3].plot(depths[: l.shape[0], :].T, l.T[2:], color=c[i], alpha=0.25)
#     ax[i + 3].plot(
#         depths[:1, :].T,
#         l[0, 2:],
#         color=c[i],
#         alpha=0.25,
#         label="Fluctuated Mean Showers",
#     )

#     ax[i + 3].fill_between(
#         depths[i, :],
#         mean_showers[i] - rms_error_shower[i],
#         mean_showers[i] + rms_error_shower[i],
#         facecolor="grey",
#         alpha=0.5,
#         hatch="////",
#         zorder=5,
#     )

#     ax[i + 3].plot(depths[i, :], mean_showers[i], "k")
#     ax[i + 3].set(xlabel="Slant Depth (g cm$^{-2}$)")
#     ax[i + 3].legend()

# # ax[0].set(xlim=(0, 2000))
# ax[3].set(ylabel="N")

# plt.savefig(
#     os.path.join(
#         "G:", "My Drive", "Research", "NASA", "full_conex_modeling", "conex_fluct.pdf"
#     ),
#     bbox_inches="tight",
# )
# plt.savefig(
#     "/home/fabg/g_drive/Research/NASA/full_conex_modeling/conex_fluct_unlog.pdf",
#     bbox_inches="tight",
# )
