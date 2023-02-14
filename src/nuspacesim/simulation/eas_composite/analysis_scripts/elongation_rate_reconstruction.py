import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py

# tup_folder = "/home/fabg/conex_runs/1000_showers"
# with h5py.File("./1000evts/lin_log_xmax_vs_energy.h5", "r") as f:
#     muons = np.array(f["muons"])
#     electron_positrons = np.array(f["electron_positron"])
#     charged = np.array(f["charged"])
#     gammas = np.array(f["gammas"])
#     hadrons = np.array(f["hadrons"])

# with h5py.File("./1000evts/log_log_nmax_vs_energy.h5", "r") as f:
#     energy_muons = np.array(f["muons"])
#     energy_electron_positrons = np.array(f["electron_positron"])
#     energy_charged = np.array(f["charged"])
#     energy_gammas = np.array(f["gammas"])
#     energy_hadrons = np.array(f["hadrons"])

tup_folder = "~/g_drive/Research/NASA/Work/conex2r7_50-runs/downward"
with h5py.File("./down_lin_log_xmax_vs_energy.h5", "r") as f:
    muons = np.array(f["muons"])
    electron_positrons = np.array(f["electron_positron"])
    charged = np.array(f["charged"])
    gammas = np.array(f["gammas"])
    hadrons = np.array(f["hadrons"])

with h5py.File("./down_log_log_nmax_vs_energy.h5", "r") as f:
    energy_muons = np.array(f["muons"])
    energy_electron_positrons = np.array(f["electron_positron"])
    energy_charged = np.array(f["charged"])
    energy_gammas = np.array(f["gammas"])
    energy_hadrons = np.array(f["hadrons"])

lg16_shwrs = "log_16_eV_1000shwrs_60_downward_eposlhc_272473279_100.root"
lg17_shwrs = "log_17_eV_1000shwrs_60_downward_eposlhc_1756896908_100.root"
lg18_shwrs = "log_18_eV_1000shwrs_60_downward_eposlhc_1791265245_100.root"
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

beta = 5


def muon_elongation_rate(log_e):
    y = muons[0, 1:][0] * log_e + muons[0, 1:][2]
    return y


def muon_decade_scaler(log_e):
    y = energy_muons[0, 1:][0] * log_e + energy_muons[0, 1:][2]
    return 10 ** y


def reco_shower(ref_shwr, reco_shwr, reco_e, elong, e_scaler, component):
    ref_ntuple = ReadConex(ref_shwr)
    ref_gh_depths = ref_ntuple.get_depths()

    reco_ntuple = ReadConex(reco_shwr)
    reco_gh_depths = reco_ntuple.get_depths()

    if component == "mu":
        ref_shwr_data = ref_ntuple.get_muons()
        reco_shwr_data = reco_ntuple.get_muons()
    elif component == "elec_pos":
        ref_shwr_data = ref_ntuple.get_elec_pos()
        reco_shwr_data = reco_ntuple.get_elec_pos()
    elif component == "charged":
        ref_shwr_data = ref_ntuple.get_charged()
        reco_shwr_data = reco_ntuple.get_charged()

    ref_mean = np.nanmean(ref_shwr_data, axis=0)
    ref_error = np.sqrt(np.nanmean((ref_mean - ref_shwr_data) ** 2, axis=0))
    ref_nmax, ref_xmax = bin_nmax_xmax(ref_gh_depths[0, :], ref_mean)

    reco_mean = np.nanmean(reco_shwr_data, axis=0)
    reco_error = np.sqrt(np.nanmean((reco_mean - reco_shwr_data) ** 2, axis=0))

    reco_xmax = elong(reco_e)
    # reco_nmax = e_scaler(reco_e)
    vertical_shift = e_scaler(reco_e) / ref_nmax
    reco_shift = reco_xmax - ref_xmax
    reco_bins = reco_shift + ref_gh_depths[0, :]

    return (
        (reco_bins, ref_mean * vertical_shift),
        (ref_gh_depths[0, :], ref_mean, ref_error),
        (reco_gh_depths[0, :], reco_mean, reco_error),
    )


reco_fit_18, ref_17, reco_18 = reco_shower(
    ref_shwr=os.path.join(
        tup_folder,
        lg17_shwrs,
    ),
    reco_shwr=os.path.join(
        tup_folder,
        lg18_shwrs,
    ),
    reco_e=18,
    elong=muon_elongation_rate,
    e_scaler=muon_decade_scaler,
    component="mu",
)

reco_fit_16, _, reco_16 = reco_shower(
    ref_shwr=os.path.join(
        tup_folder,
        lg17_shwrs,
    ),
    reco_shwr=os.path.join(
        tup_folder,
        lg16_shwrs,
    ),
    reco_e=16,
    elong=muon_elongation_rate,
    e_scaler=muon_decade_scaler,
    component="mu",
)

fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(5, 5), gridspec_kw={"height_ratios": [3, 1]}, dpi=300
)
ax[0].plot(ref_17[0], ref_17[1], lw=1, label="log 17 eV")
ax[0].fill_between(
    ref_17[0],
    ref_17[1] - ref_17[2],
    ref_17[1] + ref_17[2],
    alpha=0.5,
)
ax[0].plot(reco_18[0], reco_18[1], lw=1, label="log 18 eV")
ax[0].fill_between(
    reco_18[0],
    reco_18[1] - reco_18[2],
    reco_18[1] + reco_18[2],
    alpha=0.5,
)
ax[0].plot(reco_16[0], reco_16[1], lw=1, label="log 16 eV")
ax[0].fill_between(
    reco_16[0],
    reco_16[1] - reco_16[2],
    reco_16[1] + reco_16[2],
    alpha=0.5,
)

# get the reconstucted versions
ax[0].plot(reco_fit_18[0], reco_fit_18[1], "--k", label="reconstructed")
ax[1].plot(
    reco_fit_18[0], reco_fit_18[1] / reco_18[1], label="log 18", color="tab:orange"
)

ax[0].plot(reco_fit_16[0], reco_fit_16[1], "--k")
ax[1].plot(
    reco_fit_16[0], reco_fit_16[1] / reco_16[1], label="log 16", color="tab:green"
)


ax[0].set(yscale="log", ylim=(10, 1e9), ylabel="$N$")
ax[0].legend(title=r"$\beta = {:}$  degrees".format(beta))
ax[1].set(xlabel="slant depth g/cm^2", ylabel="reco/actual")
ax[1].legend(title="MUON", ncol=2)
ax[1].axhline(y=1, c="k", ls=":")

# plt.savefig(
#     os.path.expanduser(
#         "~/g_drive/Research/NASA/Work/muon_scaling_beta{}.pdf".format(beta)
#     ),
#     dpi=400,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
#%%


def charged_elongation_rate(log_e):
    y = charged[0, 1:][0] * log_e + charged[0, 1:][2]
    return y


def charged_decade_scaler(log_e):
    y = energy_charged[0, 1:][0] * log_e + energy_charged[0, 1:][2]
    return 10 ** y


reco_fit_18, ref_17, reco_18 = reco_shower(
    ref_shwr=os.path.join(
        tup_folder,
        lg17_shwrs,
    ),
    reco_shwr=os.path.join(
        tup_folder,
        lg18_shwrs,
    ),
    reco_e=18,
    elong=charged_elongation_rate,
    e_scaler=charged_decade_scaler,
    component="charged",
)

reco_fit_16, _, reco_16 = reco_shower(
    ref_shwr=os.path.join(
        tup_folder,
        lg17_shwrs,
    ),
    reco_shwr=os.path.join(
        tup_folder,
        lg16_shwrs,
    ),
    reco_e=16,
    elong=charged_elongation_rate,
    e_scaler=charged_decade_scaler,
    component="charged",
)

fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(5, 5), gridspec_kw={"height_ratios": [3, 1]}, dpi=300
)
ax[0].plot(ref_17[0], ref_17[1], lw=1, label="log 17 eV")
ax[0].fill_between(
    ref_17[0],
    ref_17[1] - ref_17[2],
    ref_17[1] + ref_17[2],
    alpha=0.5,
)
ax[0].plot(reco_18[0], reco_18[1], lw=1, label="log 18 eV")
ax[0].fill_between(
    reco_18[0],
    reco_18[1] - reco_18[2],
    reco_18[1] + reco_18[2],
    alpha=0.5,
)
ax[0].plot(reco_16[0], reco_16[1], lw=1, label="log 16 eV")
ax[0].fill_between(
    reco_16[0],
    reco_16[1] - reco_16[2],
    reco_16[1] + reco_16[2],
    alpha=0.5,
)

# get the reconstucted versions
ax[0].plot(reco_fit_18[0], reco_fit_18[1], "--k", label="reconstructed")
ax[1].plot(
    reco_fit_18[0], reco_fit_18[1] / reco_18[1], label="log 18", color="tab:orange"
)

ax[0].plot(reco_fit_16[0], reco_fit_16[1], "--k")
ax[1].plot(
    reco_fit_16[0], reco_fit_16[1] / reco_16[1], label="log 16", color="tab:green"
)


ax[0].set(yscale="log", ylim=(10, 1e9), ylabel="$N$")
ax[0].legend(title=r"$\beta = {:}$  degrees".format(beta))
ax[1].set(xlabel="slant depth g/cm^2", ylabel="reco/actual")
ax[1].legend(title="CHARGED", ncol=2)
ax[1].axhline(y=1, c="k", ls=":")

# plt.savefig(
#     os.path.expanduser(
#         "~/g_drive/Research/NASA/Work/charged_scaling_beta{}.pdf".format(beta)
#     ),
#     dpi=400,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
#%%


def elec_pos_elongation_rate(log_e):
    y = electron_positrons[0, 1:][0] * log_e + electron_positrons[0, 1:][2]
    return y


def elec_pos_decade_scaler(log_e):
    y = (
        energy_electron_positrons[0, 1:][0] * log_e
        + energy_electron_positrons[0, 1:][2]
    )
    return 10 ** y


reco_fit_18, ref_17, reco_18 = reco_shower(
    ref_shwr=os.path.join(
        tup_folder,
        lg17_shwrs,
    ),
    reco_shwr=os.path.join(
        tup_folder,
        lg18_shwrs,
    ),
    reco_e=18,
    elong=elec_pos_elongation_rate,
    e_scaler=elec_pos_decade_scaler,
    component="elec_pos",
)

reco_fit_16, _, reco_16 = reco_shower(
    ref_shwr=os.path.join(
        tup_folder,
        lg17_shwrs,
    ),
    reco_shwr=os.path.join(
        tup_folder,
        lg16_shwrs,
    ),
    reco_e=16,
    elong=elec_pos_elongation_rate,
    e_scaler=elec_pos_decade_scaler,
    component="elec_pos",
)

fig, ax = plt.subplots(
    nrows=2, ncols=1, figsize=(5, 5), gridspec_kw={"height_ratios": [3, 1]}, dpi=300
)
ax[0].plot(ref_17[0], ref_17[1], lw=1, label="log 17 eV")
ax[0].fill_between(
    ref_17[0],
    ref_17[1] - ref_17[2],
    ref_17[1] + ref_17[2],
    alpha=0.5,
)
ax[0].plot(reco_18[0], reco_18[1], lw=1, label="log 18 eV")
ax[0].fill_between(
    reco_18[0],
    reco_18[1] - reco_18[2],
    reco_18[1] + reco_18[2],
    alpha=0.5,
)
ax[0].plot(reco_16[0], reco_16[1], lw=1, label="log 16 eV")
ax[0].fill_between(
    reco_16[0],
    reco_16[1] - reco_16[2],
    reco_16[1] + reco_16[2],
    alpha=0.5,
)

# get the reconstucted versions
ax[0].plot(reco_fit_18[0], reco_fit_18[1], "--k", label="reconstructed")
ax[1].plot(
    reco_fit_18[0], reco_fit_18[1] / reco_18[1], label="log 18", color="tab:orange"
)

ax[0].plot(reco_fit_16[0], reco_fit_16[1], "--k")
ax[1].plot(
    reco_fit_16[0], reco_fit_16[1] / reco_16[1], label="log 16", color="tab:green"
)


ax[0].set(yscale="log", ylim=(10, 1e9), ylabel="$N$")
ax[0].legend(title=r"$\beta = {:}$  degrees".format(beta))
ax[1].set(xlabel="slant depth g/cm^2", ylabel="reco/actual")
ax[1].legend(title="ELECTRON/POSITRON", ncol=2)
ax[1].axhline(y=1, c="k", ls=":")

# plt.savefig(
#     os.path.expanduser(
#         "~/g_drive/Research/NASA/Work/ep_scaling_beta{}.pdf".format(beta)
#     ),
#     dpi=400,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
