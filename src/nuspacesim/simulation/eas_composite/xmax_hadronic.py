import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.plt_routines import decay_channel_mult_plt
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py


from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter
from nuspacesim.simulation.eas_composite.comp_eas_conex import ConexCompositeShowers
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
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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
#%% shower decay channels

lepton_decay = [300001, 300002]
had_pionkaon_1bod = [200011, 210001]
# fmt: off
had_pi0 = [300111, 310101, 400211, 410111, 410101, 410201, 401111, 400111, 500131,
           500311, 501211, 501212, 510301, 510121, 510211, 510111, 510112, 600411,
           600231,
           ]
had_no_pi0 = [310001, 311001, 310011, 311002, 311003, 400031, 410021, 410011, 410012,
              410013, 410014, 501031, 501032, 510031, 600051,
              ]
# fmt: on
# initialize the EAS bin by bin generator
generator = ConexCompositeShowers(shower_comps=init, init_pid=pids, tau_table_start=0)

lepton_decay_eas = generator(
    n_comps=1000, channel=lepton_decay, return_table=False, no_subshwrs=True
)
pk_1bo_eas = generator(
    n_comps=1000, channel=had_pionkaon_1bod, return_table=False, no_subshwrs=True
)
pi0_eas = generator(n_comps=1000, channel=had_pi0, return_table=False, no_subshwrs=True)
no_pi0_eas = generator(
    n_comps=1000, channel=had_no_pi0, return_table=False, no_subshwrs=True
)

shwr_groups = [lepton_decay_eas, pk_1bo_eas, pi0_eas, no_pi0_eas]
decay_labels = [
    r"${\rm leptonic\:decay}$",
    r"${\rm  1\:body\:K,\:\pi^{+/-}}$",
    r"${\rm  hadronic\:with\:\pi_0}$",
    r"${\rm  hadronic\:no\:\pi_0}$",
]

cmap = plt.cm.get_cmap("inferno")(np.linspace(0, 1, 7))[1:]
sample_label = r"X_{\rm max}"

#%%
# data save
mean_perchan = []
rms_err_perchan = []

mean_xmaxs = []
min_xmaxs = []
max_xmaxs = []

xmaxs_perchan = []
nmaxs_perchan = []

rms_reco_params = []

fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(7, 3))
plt.subplots_adjust(wspace=0.2)

for ci, chnl in enumerate(shwr_groups):
    mean, rms_err = mean_shower(chnl[:, 2:])
    xmax_idx = np.argmax(mean)
    mean_xmax = depths[0, 0:][xmax_idx]
    mean_xmaxs.append(mean_xmax)
    xmaxs_idx = np.argmax(chnl[:, 2:], axis=1)

    # let's get the grammages where each shower peaks
    shower_xmaxs = np.take(
        depths[0, :], xmaxs_idx
    )  # depths array are the same for conex
    xmax_multipliers = shower_xmaxs / mean_xmax

    mean_perchan.append(mean)
    rms_err_perchan.append(rms_err)

    max_xmaxs.append(shower_xmaxs.max())
    min_xmaxs.append(shower_xmaxs.min())

    # save so we can do a heat map
    xmaxs_perchan.append(shower_xmaxs)
    nmaxs_perchan.append(np.max(chnl[:, 2:], axis=1))

    bin_end = 1.40  # np.round(np.max(xmax_column / mean[xmax_idx]), 0)
    hist_bins = np.linspace(0.8, bin_end, 30)

    # histogram from x max
    # cts, bin_edges = np.histogram(xmax_column / mean[xmax_idx], bins=hist_bins)

    xmaxs_cts, xmaxs_edges = np.histogram(
        xmax_multipliers, bins=hist_bins, density=True
    )
    xmax_bin_ctrs = (xmaxs_edges[:-1] + xmaxs_edges[1:]) / 2
    params, pcov = curve_fit(gauss_exp, xmax_bin_ctrs, xmaxs_cts)

    min_val = xmax_multipliers.min()
    max_val = xmax_multipliers.max()

    lamb = params[0]
    sig = np.abs(params[1])
    mu = params[2]

    print(lamb, sig, mu, min_val, max_val)
    rms_reco_params.append([lamb, sig, mu, min_val, max_val])

    nonzero_mask = xmaxs_cts > 0
    max_val = xmaxs_edges[1:][nonzero_mask][-1]
    chi2 = np.sum(
        (xmaxs_cts[nonzero_mask] - gauss_exp(xmax_bin_ctrs, *params)[nonzero_mask]) ** 2
        / gauss_exp(xmax_bin_ctrs, *params)[nonzero_mask]
    )
    p_value = stats.chi2.sf(chi2, len(xmaxs_cts[nonzero_mask]))
    reduced_ch2 = chi2 / len(xmaxs_cts)

    # plot the theoretical fit, but 1 + the end
    theory_x = np.linspace(0.75, bin_end, 200)

    # let's loop so that we can control the actual nuber of samples, not just mask it away
    n_samples = chnl[:, 2:].shape[0]

    rand_xmax_scale = []
    while len(rand_xmax_scale) != n_samples:

        r = exponnorm.rvs(1 / (lamb * sig), loc=mu, scale=sig)
        if (r > 0) and (r <= bin_end):
            rand_xmax_scale.append(r)

    # xmax_scale_perchan.append(rand_xmax_scale)

    # ax[0].hist(
    #     xmax_multipliers,
    #     bins=hist_bins,
    #     color=cmap[ci],
    #     histtype="step",
    #     lw=2,
    #     label=decay_labels[ci],
    #     alpha=0.90,
    # )

    ax[0].plot(
        depths[0, :],
        chnl[:, 2:].T,
        lw=1,
        color=cmap[ci],
        alpha=0.2,
        zorder=1,
    )
    # just to include the label
    ax[0].plot(
        depths[0, :],
        chnl[0, 2:],
        lw=1,
        color=cmap[ci],
        alpha=1,
        zorder=1,
        label=decay_labels[ci] + r"${{\rm\: | \: {:0.0f}}}$".format(mean_xmax),
    )
    # print(mean_xmax)
    ax[0].axvline(mean_xmax, lw=1, color=cmap[ci], alpha=1, zorder=3, ls="--")

    ax[1].hist(
        xmax_multipliers,
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
        label="$({:.2f}, {:.2f}, {:.2f}, {:.2f})$".format(chi2, *params),
    )

    # ax[2].hist(
    #     rand_xmax_scale,
    #     bins=hist_bins,
    #     color=cmap[ci],
    #     histtype="step",
    #     # hatch="\\\\",
    #     lw=3,
    #     density=True,
    #     alpha=0.9,
    #     label=r"${{\rm {}\:scaling\:values}}$".format(len(rand_xmax_scale)),
    # )


ax[0].set(
    ylabel=r"$N$",
    xlabel=r"${\rm slant \: depth \: (g\: cm^{-2})}$",
    xlim=(0, 2000),
    ylim=(100, 8e7),
    yscale="log",
)
ax[0].legend(
    title=r"${\rm Decay\:Channel\:|\:mean\:X_{\rm max}(g\:cm^{-2})}$",
    loc="lower center",
    fontsize=10,
    title_fontsize=10,
    bbox_to_anchor=(0.5, 1),
)
ax[1].set(
    xlabel=r"${{\rm shower\:{} \: / \: mean \: {} }}$".format(
        sample_label, sample_label
    ),
    ylabel="PDF",
    xlim=(0.8, 1.4),
)

ax[1].legend(
    title=r"${\rm Fit} \:(\chi^2,\:\lambda,\:\sigma,\:\mu)$",
    fontsize=10,
    title_fontsize=10,
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
)
# ax[2].legend(loc="upper right", title=r"${\rm Resampled}$", fontsize=8)

plt.savefig(
    "../../../../../g_drive/Research/NASA/xmax_rms_hadronic.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
#%% see if nmax and xmax correlate
fig, ax = plt.subplots(
    nrows=2, ncols=2, dpi=300, figsize=(5, 5), sharey=True, sharex=True
)
ax = ax.ravel()
plt.subplots_adjust(wspace=0, hspace=0)
for i, x in enumerate(xmaxs_perchan):

    # ax.scatter(x, np.log10(nmaxs_perchan[i]), s=1, color=cmap[i], alpha=0.5)
    cts = ax[i].hist2d(
        x,
        np.log10(nmaxs_perchan[i]),
        bins=(50, 50),
        range=[[590, 1100], [6.3, 8]],
        cmap="bone_r",
    )
    ax[i].text(
        0.95, 0.95, decay_labels[i], transform=ax[i].transAxes, ha="right", va="top"
    )
    # print(pearsonr(x, np.log10(nmaxs_perchan[i])))
    # cts, _, _ = np.histogram2d(
    #     x,
    #     np.log10(nmaxs_perchan[i]),
    #     bins=[50, 50],
    #     normed=False,
    #     range=[[590, 1100], [5, 9]],
    # )
    # cts = cts.T

    # lum = ax.imshow(
    #     cts,
    #     cmap="inferno",
    #     # interpolation="gaussian",
    #     origin="lower",
    #     extent=[590, 1100, 5, 9],
    #     # alpha=lum_alpha,
    #     aspect="auto"
    # )
    # ax.pcolor

ax[i].set(ylim=(6.3, 8))
fig.text(0.5, 0.05, r"${\rm shower} \: X_{\rm max} {\rm (g\:cm^{-2})}$", ha="center")
fig.text(
    0.02,
    0.5,
    r"$\log_{10}\: {\rm shower}\:{N_{\rm max}}$",
    va="center",
    rotation="vertical",
)
cbar_ax = ax[0].inset_axes([0.00, 1.1, 2, 0.05])
cbar = fig.colorbar(cts[3], cax=cbar_ax, pad=-1, orientation="horizontal")
cbar_ax.set_title(r"${\rm Number\: of\: Showers\:(1000\:per\:grouping)}$", size=8)

plt.savefig(
    "../../../../../g_drive/Research/NASA/xmax_nmax_correlation.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)

#%%

keys = ["leptonic", "one_body_kpi", "with_pi0", "no_pi0"]
fname = "xmax_rms_params"
with as_file(files("nuspacesim.data.eas_reco.rms_params") / f"{fname}.h5") as path:
    print(path)
    with h5py.File(path, "w") as f:
        for i, rms in enumerate(np.array(rms_reco_params)):

            f.create_dataset(
                keys[i],
                data=rms,
                dtype="f",
            )
            f.create_dataset(
                "mean_" + keys[i],
                data=mean_perchan[i],
                dtype="f",
            )

            f.create_dataset(
                "rms_" + keys[i],
                data=rms_err_perchan[i],
                dtype="f",
            )

        f.create_dataset(
            "slant_depth",
            data=depths[0, :],
            dtype="f",
        )
