import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py


with h5py.File("./lin_log_xmax_vs_energy.h5", "r") as f:
    muons = np.array(f["muons"])
    electron_positrons = np.array(f["electron_positron"])
    charged = np.array(f["charged"])
    gammas = np.array(f["gammas"])
    hadrons = np.array(f["hadrons"])

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
tup_folder = "../conex_7_50_runs"

log_16_shwr_data = ReadConex(
    os.path.join(
        tup_folder, "log_16_eV_10shwrs_5_degearthemergence_eposlhc_1360643171_100.root"
    )
)
log_16_gh_depths = log_16_shwr_data.get_depths()
log_16_mus = log_16_shwr_data.get_muons()


log16_mean = np.nanmean(log_16_mus, axis=0)
log16_rms_error = np.sqrt(np.nanmean((log16_mean - log_16_mus) ** 2, axis=0))

log_18_shwr_data = ReadConex(
    os.path.join(
        tup_folder, "log_18_eV_10shwrs_5_degearthemergence_eposlhc_1456378716_100.root"
    )
)
log_18_gh_depths = log_18_shwr_data.get_depths()
log_18_mus = log_18_shwr_data.get_muons()

log18_mean = np.nanmean(log_18_mus, axis=0)
log18_rms_error = np.sqrt(np.nanmean((log18_mean - log_18_mus) ** 2, axis=0))


def muon_elongation_rate(log_e):
    y = muons[1, 1:][0] * log_e + muons[1, 1:][2]
    return y


def muon_decade_scaler(log_e):
    y = 0.91 * log_e - 9.61
    return 10 ** y


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), dpi=300)
ax.plot(log_16_gh_depths[0, :], log16_mean, lw=1, label="lg 16 eV mean and rms")
ax.fill_between(
    log_16_gh_depths[0, :],
    log16_mean - log16_rms_error,
    log16_mean + log16_rms_error,
    alpha=0.5,
)
mus_nmax, mus_xmax = bin_nmax_xmax(log_16_gh_depths[0, :], log16_mean)


ax.plot(log_18_gh_depths[0, :], log18_mean, lw=1, label="lg 18 eV mean and rms")
ax.fill_between(
    log_18_gh_depths[0, :],
    log18_mean - log18_rms_error,
    log18_mean + log18_rms_error,
    alpha=0.5,
)


reco_xmax = muon_elongation_rate(18)
reco_nmax = muon_decade_scaler(18)
vertical_shift = muon_decade_scaler(18) / mus_nmax
reco_shift = reco_xmax - mus_xmax
reconstructed_bins = reco_shift + log_16_gh_depths[0, :]

ax.plot(
    reconstructed_bins,
    log16_mean * vertical_shift,
    "--k",
    label="lg 16 eV reconstructed",
)

ax.set(yscale="log", ylim=(1, 1e7), xlabel="slant depth g/cm^2", ylabel="$N$")
ax.legend(title=r"$\beta = 5$  degrees")

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), dpi=300)
# for d, p in zip(log_16_gh_depths, log_16_mus):
#     ax.plot(d, p, color="red")
# for d, p in zip(log_18_gh_depths, log_18_mus):
#     ax.plot(d, p, color="blue")
# ax.set(yscale="log", ylim=(1, 1e7))
