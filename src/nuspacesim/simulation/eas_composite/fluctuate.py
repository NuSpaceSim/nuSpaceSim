import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
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

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "font.size": 8,
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
    rms_error = np.sqrt(np.nanmean((average - showers_n) ** 2, axis=0))
    rms = np.sqrt(np.nanmean((showers_n) ** 2, axis=0))
    std = np.nanstd(showers_n, axis=0)
    err_in_mean = np.nanstd(showers_n, axis=0) / np.sqrt(
        np.sum(~np.isnan(showers_n), 0)
    )
    rms_low = average - rms_error
    rms_high = average + rms_error
    return average, rms


#%%
tup_folder = "/home/fabg/g_drive/Research/NASA/Work/conex2r7_50-runs/"
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
# we can get the charged compoenent
elec_charged = elec_init.get_charged()
gamma_charged = gamma_init.get_charged()
pion_charged = pion_init.get_charged()
depths = elec_init.get_depths()

pids = [11, 22, 211]
init = [elec_charged, gamma_charged, pion_charged]
gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids)
comp_charged = gen_comp()
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
    mean_showers.append(mean)
    rms_error_shower.append(rms_error)
    _, _, _, dist = sample.sampling_nmax_once(return_rms_dist=True)
    nmax_dist.append(dist)

    # for each comman decay channel, fluctuate it a lot
    mult = np.zeros(100)

    fluctuated_per_channel = []

    for m, r in enumerate(mult):
        mc_rms_multiplier, _, _ = sample.sampling_nmax_once(return_rms_dist=False)
        mult[m] = mc_rms_multiplier
        fluctuated_per_channel.append(mean * mc_rms_multiplier)

    multipliers.append(mult)
    fluctuated.append(np.array(fluctuated_per_channel))

#!!! see how mean, fluctuated, rms error, and mean fluctuated compare.
#%%

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(4, 4))
for i, dist in enumerate(nmax_dist):
    ax.hist(
        dist / np.mean(dist),
        alpha=0.5,
        # edgecolor="black",
        linewidth=0.5,
        label=labels[i],
        bins=np.linspace(0, 3, 20),
        histtype="step",
        lw=2,
    )
ax.legend(title="Composite Conex, Charged Component", bbox_to_anchor=(0.1, 1))
ax.set(xlabel="Nmax/mean Nmax", ylabel="Number of Showers")

#%%


sample = MCVariedMean(
    filtered_n,
    comp_charged[: filtered_n.shape[0], :],
    n_throws=100,
    hist_bins=30,
    sample_grammage=100,
)


# rms_err_upper = mean_shwr + mc_rms * mean_shwr
# rms_err_lower = mean_shwr - mc_rms * mean_shwr
# abs_error = rms_err_upper - mean_shwr
average, rms = mean_shower(e_channel_n)

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300, figsize=(5, 5))
ax.plot(
    elec_depths[0, :].T,
    np.log10(e_channel_n[:, 2:].T),
    color="grey",
    alpha=0.2,
)

ax.plot(elec_depths[0, :], np.log10(average[2:]))
# ax.plot(elec_depths[0, :], np.log10(average[2:] - rms_error[2:]))
ax.set(ylim=(0, 8))
custom_lines = [
    Line2D([0], [0], color="grey", lw=4),
]
# Line2D([0], [0], color=cmap(.5), lw=4),
# Line2D([0], [0], color=cmap(1.), lw=4)]

ax.legend(custom_lines, [r"e$^{{+/-}}$ component, {}".format(decay_channel)])

fig.text(0.08, 0.5, r"log N", va="center", rotation="vertical")
fig.text(0.5, 0.08, r"Slant Depth (g cm$^{-2}$)", ha="center")
#%%
from scipy import optimize


def modified_gh(x, n_max, x_max, x_0, p1, p2, p3):

    particles = (
        n_max
        * np.nan_to_num(
            ((x - x_0) / (x_max - x_0))
            ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x ** 2)))
        )
    ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x ** 2))))

    return particles


def fit_quad_lambda(depth, comp_shower):
    r"""
    Gets fits for composite shower if supplied particle content and matching slant depths.
    Allows negative X0 and quadratic lambda.
    """

    nmax, xmax = bin_nmax_xmax(bins=depth, particle_content=comp_shower)
    fit_params, covariance = optimize.curve_fit(
        f=modified_gh,
        xdata=depth,
        ydata=comp_shower,
        p0=[nmax, xmax, 0, 80, -0.01, 1e-05],
        bounds=(
            [0, 0, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
    )
    theory_n = modified_gh(depth, *fit_params)
    print(fit_params)
    return theory_n


avg_fit = fit_quad_lambda(elec_depths[0, :], average[2:])
#%%
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300, figsize=(5, 5))
ax.plot(
    elec_depths[0, :].T,
    np.log10(e_channel_n[:, 2:].T),
    color="grey",
    alpha=0.2,
)
ax.plot(
    elec_depths[0, :].T,
    np.log10(e_channel_n_charged[:, 2:].T),
    color="red",
    alpha=0.2,
)
ax.plot(elec_depths[0, :], np.log10(average[2:]), color="tab:blue")
ax.plot(
    elec_depths[0, :],
    np.log10(np.outer(multipliers, average[2:]).T),
    # ls=":",
    color="tab:blue",
    alpha=0.1,
)

ax.plot(elec_depths[0, :], np.log10(avg_fit), "--k")
# ax.plot(elec_depths[0, :], np.log10(average[2:] - rms_error[2:]))
ax.set(ylim=(0, 8))
custom_lines = [
    Line2D([0], [0], color="grey", lw=1),
    Line2D([0], [0], color="tab:blue", lw=1),
    Line2D([0], [0], ls="--", color="k", lw=1),
    Line2D([0], [0], ls="--", color="red", lw=1),
]
# Line2D([0], [0], color=cmap(.5), lw=4),
# Line2D([0], [0], color=cmap(1.), lw=4)]

ax.legend(
    custom_lines,
    [
        r"e$^{{+/-}}$ component, {}".format(decay_channel),
        "mean, varied by sampling Nmax RMS",
        "Mean GH quadratic lambda",
        "charged component",
    ],
)

fig.text(0.08, 0.5, r"log N", va="center", rotation="vertical")
fig.text(0.5, 0.08, r"Slant Depth (g cm$^{-2}$)", ha="center")
plt.savefig("./ep_componenet_electronfinalstate_mean_ghfit.pdf")
