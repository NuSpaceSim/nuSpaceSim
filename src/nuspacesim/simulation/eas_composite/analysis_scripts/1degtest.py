import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py


with h5py.File("./100evts/lin_log_xmax_vs_energy.h5", "r") as f:
    muons = np.array(f["muons"])
    electron_positrons = np.array(f["electron_positron"])
    charged = np.array(f["charged"])
    gammas = np.array(f["gammas"])
    hadrons = np.array(f["hadrons"])

with h5py.File("./100evts/log_log_nmax_vs_energy.h5", "r") as f:
    energy_muons = np.array(f["muons"])
    energy_electron_positrons = np.array(f["electron_positron"])
    energy_charged = np.array(f["charged"])
    energy_gammas = np.array(f["gammas"])
    energy_hadrons = np.array(f["hadrons"])
#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), dpi=300)
ax.errorbar(
    muons[:, 0],
    muons[:, 1],
    muons[:, 2],
    fmt="o",
    capsize=5,
    alpha=0.8,
    label="muon",
)
ax.errorbar(
    electron_positrons[:, 0],
    electron_positrons[:, 1],
    electron_positrons[:, 2],
    fmt="o",
    capsize=5,
    alpha=0.8,
    label="electrons_positrons",
)

ax.errorbar(
    gammas[:, 0],
    gammas[:, 1],
    gammas[:, 2],
    fmt="o",
    capsize=5,
    alpha=0.8,
    label="gammas",
)

ax.errorbar(
    hadrons[:, 0],
    hadrons[:, 1],
    hadrons[:, 2],
    fmt="o",
    capsize=5,
    alpha=0.8,
    label="hadrons",
)

ax.errorbar(
    charged[:, 0],
    charged[:, 1],
    charged[:, 2],
    fmt="o",
    capsize=5,
    alpha=0.8,
    label="charged",
)

ax.set(
    xlabel="earth emergence angle (degrees)",
    ylabel="$\mathrm{elongation rate (\Delta Xmax / \Delta \log E)}$",
)
ax.legend(ncol=2, fontsize=8)

#%% try to reconstruct it using soleley the elenogation rate
tup_folder = "/home/fabg/conex_runs/100_showers"

log_17_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_100shwrs_1_degearthemergence_eposlhc_041793568_100.root",
    )
)
log_17_gh_depths = log_17_shwr_data.get_depths()
log_17_mus = log_17_shwr_data.get_muons()

log17_mean = np.nanmean(log_17_mus, axis=0)
log17_rms_error = np.sqrt(np.nanmean((log17_mean - log_17_mus) ** 2, axis=0))

log_18_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_18_eV_100shwrs_1_degearthemergence_eposlhc_988697912_100.root",
    )
)
log_18_gh_depths = log_18_shwr_data.get_depths()
log_18_mus = log_18_shwr_data.get_muons()
log18_mean = np.nanmean(log_18_mus, axis=0)
log18_rms_error = np.sqrt(np.nanmean((log18_mean - log_18_mus) ** 2, axis=0))


log_16_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_16_eV_100shwrs_1_degearthemergence_eposlhc_1843019347_100.root",
    )
)
log_16_gh_depths = log_16_shwr_data.get_depths()
log_16_mus = log_16_shwr_data.get_muons()
log16_mean = np.nanmean(log_16_mus, axis=0)
log16_rms_error = np.sqrt(np.nanmean((log16_mean - log_16_mus) ** 2, axis=0))


def muon_elongation_rate(log_e):
    y = muons[0, 1:][0] * log_e + muons[0, 1:][2]
    return y


def muon_decade_scaler(log_e):
    y = energy_muons[0, 1:][0] * log_e + energy_muons[0, 1:][2]
    return 10 ** y


fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(5, 5), gridspec_kw={"height_ratios": [3, 1]}, dpi=300
)
ax[0].plot(log_17_gh_depths[0, :], log17_mean, lw=1, label="lg 17 eV mean and rms")
ax[0].fill_between(
    log_17_gh_depths[0, :],
    log17_mean - log17_rms_error,
    log17_mean + log17_rms_error,
    alpha=0.5,
)
mus_nmax, mus_xmax = bin_nmax_xmax(log_17_gh_depths[0, :], log17_mean)


ax[0].plot(log_18_gh_depths[0, :], log18_mean, lw=1, label="lg 18 eV mean and rms")
ax[0].fill_between(
    log_18_gh_depths[0, :],
    log18_mean - log18_rms_error,
    log18_mean + log18_rms_error,
    alpha=0.5,
)
ax[0].plot(log_16_gh_depths[0, :], log16_mean, lw=1, label="lg 16 eV mean and rms")
ax[0].fill_between(
    log_16_gh_depths[0, :],
    log16_mean - log16_rms_error,
    log16_mean + log16_rms_error,
    alpha=0.5,
)

reco_xmax = muon_elongation_rate(18)
reco_nmax = muon_decade_scaler(18)
vertical_shift = muon_decade_scaler(18) / mus_nmax
reco_shift = reco_xmax - mus_xmax
reconstructed_bins = reco_shift + log_17_gh_depths[0, :]

ax[0].plot(
    reconstructed_bins,
    log17_mean * vertical_shift,
    "--k",
    label="reconstructed",
)
ax[1].plot(
    reconstructed_bins,
    (log17_mean * vertical_shift) / log18_mean,
    label="log 18",
    color="tab:orange",
)

reco_xmax = muon_elongation_rate(16)
reco_nmax = muon_decade_scaler(16)
vertical_shift = muon_decade_scaler(16) / mus_nmax
reco_shift = reco_xmax - mus_xmax
reconstructed_bins = reco_shift + log_17_gh_depths[0, :]

ax[0].plot(
    reconstructed_bins,
    log17_mean * vertical_shift,
    "--k",
    label="reconstructed",
)
ax[1].plot(
    reconstructed_bins,
    (log17_mean * vertical_shift) / log16_mean,
    label="log 16",
    color="tab:green",
)

ax[0].set(yscale="log", ylim=(10, 1e9), ylabel="$N$")
ax[0].legend(title=r"$\beta = 1$  degrees")
ax[1].set(yscale="log", xlabel="slant depth g/cm^2", ylabel="reco/actual")

ax[1].legend()
ax[1].axhline(y=1, ls="--", c="k")
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), dpi=300)
# for d, p in zip(log_17_gh_depths, log_17_mus):
#     ax.plot(d, p, color="red")
# for d, p in zip(log_18_gh_depths, log_18_mus):
#     ax.plot(d, p, color="blue")
# ax.set(yscale="log", ylim=(1, 1e7))
#%%
log_17_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_100shwrs_1_degearthemergence_eposlhc_041793568_100.root",
    )
)
log_17_gh_depths = log_17_shwr_data.get_depths()
log_17_charged = log_17_shwr_data.get_charged()

log17_mean = np.nanmean(log_17_charged, axis=0)
log17_rms_error = np.sqrt(np.nanmean((log17_mean - log_17_charged) ** 2, axis=0))

log_18_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_18_eV_100shwrs_1_degearthemergence_eposlhc_988697912_100.root",
    )
)
log_18_gh_depths = log_18_shwr_data.get_depths()
log_18_charged = log_18_shwr_data.get_charged()
log18_mean = np.nanmean(log_18_charged, axis=0)
log18_rms_error = np.sqrt(np.nanmean((log18_mean - log_18_charged) ** 2, axis=0))

log_16_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_16_eV_100shwrs_1_degearthemergence_eposlhc_1843019347_100.root",
    )
)
log_16_gh_depths = log_16_shwr_data.get_depths()
log_16_charged = log_16_shwr_data.get_charged()
log16_mean = np.nanmean(log_16_charged, axis=0)
log16_rms_error = np.sqrt(np.nanmean((log16_mean - log_16_charged) ** 2, axis=0))


def charged_elongation_rate(log_e):
    y = charged[0, 1:][0] * log_e + charged[0, 1:][2]
    return y


def charged_decade_scaler(log_e):
    y = energy_charged[0, 1:][0] * log_e + energy_charged[0, 1:][2]
    return 10 ** y


fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(5, 5), gridspec_kw={"height_ratios": [3, 1]}, dpi=300
)
ax.plot(log_17_gh_depths[0, :], log17_mean, lw=1, label="lg 17 eV mean and rms")
ax.fill_between(
    log_17_gh_depths[0, :],
    log17_mean - log17_rms_error,
    log17_mean + log17_rms_error,
    alpha=0.5,
)
charged_nmax, charged_xmax = bin_nmax_xmax(log_17_gh_depths[0, :], log17_mean)

ax.plot(log_18_gh_depths[0, :], log18_mean, lw=1, label="lg 18 eV mean and rms")
ax.fill_between(
    log_18_gh_depths[0, :],
    log18_mean - log18_rms_error,
    log18_mean + log18_rms_error,
    alpha=0.5,
)
reco_xmax = charged_elongation_rate(18)
reco_nmax = charged_decade_scaler(18)
vertical_shift = charged_decade_scaler(18) / charged_nmax
reco_shift = reco_xmax - charged_xmax
reconstructed_bins = reco_shift + log_17_gh_depths[0, :]
ax.plot(
    reconstructed_bins,
    log17_mean * vertical_shift,
    "--k",
)
ax[0].plot(
    reconstructed_bins,
    log17_mean * vertical_shift,
    "--k",
    label="reconstructed",
)
ax[1].plot(
    reconstructed_bins,
    (log17_mean * vertical_shift) / log17_mean,
    label="log 18",
    color="tab:orange",
)


ax.plot(log_16_gh_depths[0, :], log16_mean, lw=1, label="lg 16 eV mean and rms")
ax.fill_between(
    log_16_gh_depths[0, :],
    log16_mean - log16_rms_error,
    log16_mean + log16_rms_error,
    alpha=0.5,
)
reco_xmax = charged_elongation_rate(16)
reco_nmax = charged_decade_scaler(16)
vertical_shift = charged_decade_scaler(16) / charged_nmax
reco_shift = reco_xmax - charged_xmax
reconstructed_bins = reco_shift + log_17_gh_depths[0, :]
ax.plot(
    reconstructed_bins,
    log17_mean * vertical_shift,
    "--k",
    label="reconstructed",
)


ax.set(yscale="log", ylim=(10, 1e9), xlabel="slant depth g/cm^2", ylabel="$N$")
ax.legend(title=r"$\beta = 1$  degrees,  charged component", fontsize=8)

#%%
log_17_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_100shwrs_1_degearthemergence_eposlhc_041793568_100.root",
    )
)
log_17_gh_depths = log_17_shwr_data.get_depths()
log_17_elec_pos = log_17_shwr_data.get_elec_pos()

log17_mean = np.nanmean(log_17_elec_pos, axis=0)
log17_rms_error = np.sqrt(np.nanmean((log17_mean - log_17_elec_pos) ** 2, axis=0))

log_18_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_18_eV_100shwrs_1_degearthemergence_eposlhc_988697912_100.root",
    )
)
log_18_gh_depths = log_18_shwr_data.get_depths()
log_18_elec_pos = log_18_shwr_data.get_elec_pos()
log18_mean = np.nanmean(log_18_elec_pos, axis=0)
log18_rms_error = np.sqrt(np.nanmean((log18_mean - log_18_elec_pos) ** 2, axis=0))


log_16_shwr_data = ReadConex(
    os.path.join(
        tup_folder,
        "log_16_eV_100shwrs_1_degearthemergence_eposlhc_1843019347_100.root",
    )
)
log_16_gh_depths = log_16_shwr_data.get_depths()
log_16_elec_pos = log_16_shwr_data.get_elec_pos()
log16_mean = np.nanmean(log_16_elec_pos, axis=0)
log16_rms_error = np.sqrt(np.nanmean((log16_mean - log_16_elec_pos) ** 2, axis=0))


def electron_positrons_elongation_rate(log_e):
    y = electron_positrons[0, 1:][0] * log_e + electron_positrons[0, 1:][2]
    return y


def electron_positrons_decade_scaler(log_e):
    y = (
        energy_electron_positrons[0, 1:][0] * log_e
        + energy_electron_positrons[0, 1:][2]
    )
    return 10 ** y


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)
ax.plot(log_17_gh_depths[0, :], log17_mean, lw=1, label="lg 17 eV mean and rms")
ax.fill_between(
    log_17_gh_depths[0, :],
    log17_mean - log17_rms_error,
    log17_mean + log17_rms_error,
    alpha=0.5,
)
elec_pos_nmax, elec_pos_xmax = bin_nmax_xmax(log_17_gh_depths[0, :], log17_mean)


ax.plot(log_18_gh_depths[0, :], log18_mean, lw=1, label="lg 18 eV mean and rms")
ax.fill_between(
    log_18_gh_depths[0, :],
    log18_mean - log18_rms_error,
    log18_mean + log18_rms_error,
    alpha=0.5,
)
reco_xmax = electron_positrons_elongation_rate(18)
reco_nmax = electron_positrons_decade_scaler(18)
vertical_shift = electron_positrons_decade_scaler(18) / elec_pos_nmax
reco_shift = reco_xmax - elec_pos_xmax
reconstructed_bins = reco_shift + log_17_gh_depths[0, :]
ax.plot(
    reconstructed_bins,
    log17_mean * vertical_shift,
    "--k",
    # label="lg 17 eV reconstructed",
)


ax.plot(log_16_gh_depths[0, :], log16_mean, lw=1, label="lg 16 eV mean and rms")
ax.fill_between(
    log_16_gh_depths[0, :],
    log16_mean - log16_rms_error,
    log16_mean + log16_rms_error,
    alpha=0.5,
)
reco_xmax = electron_positrons_elongation_rate(16)
reco_nmax = electron_positrons_decade_scaler(16)
vertical_shift = electron_positrons_decade_scaler(16) / elec_pos_nmax
reco_shift = reco_xmax - elec_pos_xmax
reconstructed_bins = reco_shift + log_17_gh_depths[0, :]
ax.plot(
    reconstructed_bins,
    log17_mean * vertical_shift,
    "--k",
    label="reconstructed",
)

ax.set(yscale="log", ylim=(10, 1e9), xlabel="slant depth g/cm^2", ylabel="$N$")
ax.legend(
    title=r"$\beta = 1$  degrees,  elec_pos component", fontsize=8, loc="lower center"
)
