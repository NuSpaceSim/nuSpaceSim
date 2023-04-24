"""
generating composite showers using the profiles themselves from conex, not just the GH
100 PeV or 10^17 eV for 5 degree earth emergence angles
"""

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

from scipy.optimize import fsolve


try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files

tup_folder = "/home/fabg/g_drive/Research/NASA/Work/conex2r7_50-runs/"

# read in pythia decays


def generate_composite(
    e_init_charged,
    pi_init_charged,
    gamma_init_charged,
    e_init_elec,
    pi_init_elec,
    gamma_init_elec,
    e_init_gamma,
    pi_init_gamma,
    gamma_init_gamma,
    e_init_hadrons,
    pi_init_hadrons,
    gamma_init_hadrons,
    e_init_muons,
    pi_init_muons,
    gamma_init_muons,
    beta=5,
    shwr_per_file=1000,
    tautable_start=0,
):
    r"""Create composite showers by taking the shower components themselves, not just
    the GH Fits.

    Current composite showers require: electron, pion, and gamma initiated showers.

    Note, the slant depth bins are uniform, there are no negative X0, all things are
    physical; no gh fitting at all.

    Parameters
    ----------
    e_init: array
        electron initiated N
    pi_init: array
        pion initiated N
    gamma_init: array
        gamma initiated N
    beta: float
        earth emergence angle
    shwr_per_file: float
        number of events for each particle initiated shower
    tautable_start:float
        what row to start sampling the 10,000-row tau decay table used to scale N



    Returns
    -------
    composite_showers: array
        Shower content N for each generated shower.

    """

    with as_file(files("nuspacesim.data.pythia_tau_decays") / "tau_100_PeV.h5") as path:
        data = h5py.File(path, "r")
        tau_decays = np.array(data.get("tau_data"))

    # sub sample tau decay tables
    # generate mask to isolate each daughter energy scaling param from PYTHIA
    tau_tables = tau_decays[tautable_start:, :]
    muon_mask = tau_tables[:, 2] == 13
    electron_mask = tau_tables[:, 2] == 11
    # kaons and pions treated the same
    pion_kaon_mask = ((tau_tables[:, 2] == 211) | (tau_tables[:, 2] == -211)) | (
        (tau_tables[:, 2] == 321) | (tau_tables[:, 2] == -321)
    )
    gamma_mask = tau_tables[:, 2] == 22

    # each row has [event_num, decay_code,  energy scaling]
    electron_energies = tau_tables[electron_mask][:, [0, 1, -1]]
    pion_energies = tau_tables[pion_kaon_mask][:, [0, 1, -1]]
    gamma_energies = tau_tables[gamma_mask][:, [0, 1, -1]]

    # scale the charged components by the energy
    scaled_elec = (
        e_init_charged * electron_energies[:, -1][:shwr_per_file][:, np.newaxis]
    )
    scaled_elec = np.concatenate(
        (electron_energies[:, :2][:shwr_per_file], e_init_charged), axis=1
    )
    scaled_pion = pi_init_charged * pion_energies[:, -1][:shwr_per_file][:, np.newaxis]
    scaled_pion = np.concatenate(
        (pion_energies[:, :2][:shwr_per_file], pi_init_charged), axis=1
    )
    scaled_gamma = (
        gamma_init_charged * gamma_energies[:, -1][:shwr_per_file][:, np.newaxis]
    )
    scaled_gamma = np.concatenate(
        (gamma_energies[:, :2][:shwr_per_file], gamma_init_charged), axis=1
    )

    # muon_energies = tau_tables[muon_mask][:, [0, 1, -1]]
    # scaled_muon_hadrons = muon_hadrons * muon_energies[:, -1][:shwr_per_file][:, np.newaxis]
    # scaled_muon_hadrons = np.concatenate(
    #     (muon_energies[:, :2][:shwr_per_file], muon_hadrons), axis=1
    # )
    #!!!  sum the scaled individual component profiles ...
    ##so you should have a composite for the hadrons, one for the gamma, one for the e^+/-, one for the muons
    charged_single_shwrs = np.concatenate(
        (scaled_elec, scaled_gamma, scaled_pion), axis=0
    )

    # make composite showers
    single_shwrs = single_shwrs[single_shwrs[:, 0].argsort()]
    grps, idx, num_showers_in_evt = np.unique(
        single_shwrs[:, 0], return_index=True, return_counts=True, axis=0
    )
    unique_decay_codes = np.take(single_shwrs[:, 1], idx)
    composite_showers = np.column_stack(
        (
            grps,
            unique_decay_codes,
            np.add.reduceat(single_shwrs[:, 2:], idx),
        )
    )
    return composite_showers


# class Co
#%% read in the different showers initiated by different particles

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

elec_charged = elec_init.get_elec()
gamma_charged = gamma_init.get_elec()
pion_charged = pion_init.get_elec()
# corresponding depths
elec_depths = elec_init.get_depths()
gamma_depths = gamma_init.get_depths()
pion_depths = pion_init.get_depths()

composite = generate_composite(elec_charged, gamma_charged, pion_charged)
composite_charged = generate_composite(elec_charged, gamma_charged, pion_charged)
#%%
dc = 300001
decay_channel = get_decay_channel(dc)
_, e_channel_n, _, not_e_channel_n = decay_channel_filter(
    composite, composite, dc, get_discarded=True
)

_, e_channel_n_charged, _, not_e_channel_n = decay_channel_filter(
    composite, composite_charged, dc, get_discarded=True
)


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300, figsize=(5, 5))
ax.plot(
    elec_depths[0, :].T,
    np.log10(e_channel_n[:, 2:].T),
    color="grey",
    alpha=0.2,
)


#%%
from matplotlib.lines import Line2D

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


from nuspacesim.simulation.eas_composite.mc_mean_shwr import MCVariedMean

sampler = MCVariedMean(
    e_channel_n,
    elec_depths[:962, :],
    n_throws=100,
    hist_bins=30,
    sample_grammage=100,
)

multipliers = np.ones(100)
for m, rms_error in enumerate(multipliers):
    mc_rms_multiplier, _, _ = sampler.sampling_nmax_once()
    multipliers[m] = mc_rms_multiplier

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
