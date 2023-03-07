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

from scipy.optimize import fsolve

#%%

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files

tup_folder = "/home/fabg/g_drive/Research/NASA/Work/conex2r7_50-runs/"

#%% read in the different showers initiated by different particles

# proton_init = ReadConex(
#     os.path.join(
#         tup_folder,
#         "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_1743428413_100.root",
#     )
# )
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
# muon_init = ReadConex(
#     os.path.join(
#         tup_folder,
#         "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_1801137428_13.root",
#     )
# )

gamma_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_1722203790_22.root",
    )
)
#%%


def altittude_to_depth(z):
    """
    # c     Calculate Grammage
    """
    z = np.array(z)
    X = np.empty_like(z)
    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25
    X[mask1] = np.power(((z[mask1] - 44.34) / -11.861), (1 / 0.19))
    X[mask2] = np.exp(np.divide(z[mask2] - 45.5, -6.34))
    X[mask3] = np.exp(np.subtract(13.841, np.sqrt(28.920 + 3.344 * z[mask3])))

    rho = np.empty_like(z)
    rho[mask1] = (
        -1.0e-5
        * (1 / 0.19)
        / (-11.861)
        * ((z[mask1] - 44.34) / -11.861) ** ((1.0 / 0.19) - 1.0)
    )
    rho[mask2] = np.multiply(-1e-5 * np.reciprocal(-6.34), X[mask2], dtype=np.float32)
    rho[mask3] = np.multiply(
        np.divide(0.5e-5 * 3.344, np.sqrt(28.920 + 3.344 * z[mask3])), X[mask3]
    )
    return X  # , rho


def depth_to_altitude(x):
    x = np.array(x)
    altitude_out = np.empty_like(x)

    # for altitudes z < 11
    altitude = (-11.861 * x ** 0.19) + 44.34
    mask1 = altitude < 11
    altitude_out[mask1] = altitude[mask1]

    # for altitudes z >= 11, z < 25
    altitude = -6.34 * np.log(x) + 45.5
    mask2 = (altitude >= 11) & (altitude < 25)
    altitude_out[mask2] = altitude[mask2]

    # for altitudes  z >= 25
    altitude = ((13.841 - np.log(x)) ** 2 - 28.920) / 3.344
    mask3 = altitude >= 25
    altitude_out[mask3] = altitude[mask3]

    return altitude_out


def slant_depth_to_depth(slant_depth, corrected_angles):
    depth = slant_depth * np.cos(np.radians(corrected_angles))
    return depth


def calc_alpha(obs_height, earth_emergence_angle):
    """
    calculate the angle of the detector’s line of sight (respect to the local zenith),
    alpha
    """

    def f(xy, r_earth=6371):

        x, y = xy
        z = np.array(
            [
                y
                - (np.cos(np.radians(earth_emergence_angle)) / (r_earth + obs_height)),
                y - (np.cos(x) / r_earth),
            ]
        )
        return z

    alpha_rads = fsolve(f, [0, 1])[0]
    print(fsolve(f, [0, 1])[1])
    alpha_degs = np.degrees(alpha_rads) % 360
    print(alpha_degs)
    if alpha_degs >= 180:
        alpha_degs = 360 - alpha_degs

    return alpha_degs


def depth_to_alt(beta, slant_depths):
    altitude_array = np.linspace(0, 150, 10000)  # determines the step in altitude
    depths = altittude_to_depth(altitude_array)
    lower_vertical_depths = depths[:-1]
    upper_vertical_depths = depths[1:]
    delta_vertical_depth = lower_vertical_depths - upper_vertical_depths

    obs_height = 33

    r_earth = 6371
    # calculate alpha given earth emergance angle and beta by setting a equal to 0
    # alpha_deg = calc_alpha(obs_height, earth_emergence_angle=beta)

    alpha_deg = np.degrees(
        np.arcsin((np.cos(np.radians(beta)) * r_earth) / (r_earth + obs_height))
    )
    beta_prime = np.degrees(
        np.arccos(
            (np.sin(np.radians(alpha_deg)) * (r_earth + obs_height))
            / (r_earth + altitude_array[1:])
        )
    )
    corrected_path_length = delta_vertical_depth / np.sin(np.radians(beta_prime))

    upper_slant_depth = np.cumsum(corrected_path_length)

    residuals = np.abs(upper_slant_depth - slant_depths[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1)
    out_alts = altitude_array[closest_match_idxs]

    return out_alts
    # high_res_slant_depth = np.linspace(0, max_alt, 1500)
    # interpolated_altitudes = np.interp(high_res_slant_depth, xp=slant_depths, fp=out_alts)

    # max_val = np.argmax(out_alts)


# plt.figure(dpi=300)
# plt.plot(altitude_array[1:], delta_vertical_depth, label="path length")
# plt.plot(altitude_array[1:], corrected_path_length, label="corrected path length")
# plt.xlabel("altitude (km)")
# plt.ylabel("$\Delta X$")
# plt.ylim(-5, 1000)
# plt.legend(title=r"$\beta = {}\degree$".format(beta))

# plt.figure(dpi=300)
# plt.plot(altitude_array[1:], beta_prime)
# plt.xlabel("altitude (km)")
# plt.ylabel(r"$\beta'$")
# plt.legend(title=r"$\beta = {}\degree$".format(beta))

# plt.figure(dpi=300)
# plt.plot(altitude_array[1:], upper_vertical_depths, label="vertical depth")
# plt.plot(
#     altitude_array[1:],
#     upper_slant_depth,
#     label="cumulative sum of corrected path length",
# )
# # plt.axvline(0.5)
# # plt.axhline(800)
# plt.xlabel("altitude (km)")
# plt.ylabel(r"$g / cm^{-2}$")
# plt.legend(title=r"$\beta = {}\degree$".format(beta))

#%% get the depths and focus on the charged componenets for now
# depths = proton_init.get_depths()
# char = proton_init.get_charged()
# muon_depths = muon_init.get_depths()
# muon_char = muon_init.get_charged()


elec_char = elec_init.get_charged()
gamma_char = gamma_init.get_charged()
pion_char = pion_init.get_charged()

elec_gamma = elec_init.get_gamma()
gamma_gamma = gamma_init.get_gamma()
pion_gamma = pion_init.get_gamma()

elec_elec_pos = elec_init.get_elec_pos()
gamma_elec_pos = gamma_init.get_elec_pos()
pion_elec_pos = pion_init.get_elec_pos()

elec_hadrons = elec_init.get_hadrons()
gamma_hadrons = gamma_init.get_hadrons()
pion_hadrons = pion_init.get_hadrons()

elec_muons = elec_init.get_muons()
gamma_muons = gamma_init.get_muons()
pion_muons = pion_init.get_muons()

elec_depths = elec_init.get_depths()
gamma_depths = gamma_init.get_depths()
pion_depths = pion_init.get_depths()

elec_dedx = elec_init.get_dedx()
gamma_dedx = gamma_init.get_dedx()
pion_dedx = pion_init.get_dedx()

# save one shower
header = " slant depth, altitude(km), dEdX, electron/positron, gamma, hadron, muon, "
data = np.vstack(
    (
        pion_depths[0, :],
        depth_to_alt(5, pion_depths[0, :]),
        pion_dedx[0, :],
        pion_elec_pos[0, :],
        pion_gamma[0, :],
        pion_hadrons[0, :],
        pion_muons[0, :],
    )
).T
np.savetxt("one_shower_beta5_100PeV_Pion_primary.txt", X=data, header=header)
# here are other componenets
# had = shwr_data.get_hadrons()
# mus = shwr_data.get_muons()
# el = shwr_data.get_elec_pos()
# gam = shwr_data.get_gamma()

#
#%% read in pythia decays

with as_file(files("nuspacesim.data.pythia_tau_decays") / "tau_100_PeV.h5") as path:
    data = h5py.File(path, "r")
    tau_decays = np.array(data.get("tau_data"))

# sub sample tau decay tables
row_strt = 100
tau_tables = tau_decays[row_strt:, :]

# generate mask to isolate each daughter energy scaling param from PYTHIA
electron_mask = tau_tables[:, 2] == 11
muon_mask = tau_tables[:, 2] == 13
# kaons and pions treated the same
pion_kaon_mask = ((tau_tables[:, 2] == 211) | (tau_tables[:, 2] == -211)) | (
    (tau_tables[:, 2] == 321) | (tau_tables[:, 2] == -321)
)

gamma_mask = tau_tables[:, 2] == 22

# each row has [event_num, decay_code,  energy scaling]
electron_energies = tau_tables[electron_mask][:, [0, 1, -1]]
muon_energies = tau_tables[muon_mask][:, [0, 1, -1]]
pion_energies = tau_tables[pion_kaon_mask][:, [0, 1, -1]]

gamma_energies = tau_tables[gamma_mask][:, [0, 1, -1]]

#%% scale each shower by the energy and tack on the event number and decay code

shwr_per_file = 1000

scaled_elec_char = elec_char * electron_energies[:, -1][:shwr_per_file][:, np.newaxis]
scaled_elec_char = np.concatenate(
    (electron_energies[:, :2][:shwr_per_file], elec_char), axis=1
)
# scaled_muon_char = muon_char * muon_energies[:, -1][:shwr_per_file][:, np.newaxis]
# scaled_muon_char = np.concatenate(
#     (muon_energies[:, :2][:shwr_per_file], muon_char), axis=1
# )
scaled_pion_char = pion_char * pion_energies[:, -1][:shwr_per_file][:, np.newaxis]
scaled_pion_char = np.concatenate(
    (pion_energies[:, :2][:shwr_per_file], pion_char), axis=1
)

scaled_gamma_char = gamma_char * gamma_energies[:, -1][:shwr_per_file][:, np.newaxis]
scaled_gamma_char = np.concatenate(
    (gamma_energies[:, :2][:shwr_per_file], pion_char), axis=1
)


# elec_depths = np.concatenate(
#     (electron_energies[:, :2][:shwr_per_file], elec_depths), axis=1
# )
# pion_depths = np.concatenate(
#     (pion_energies[:, :2][:shwr_per_file], pion_depths), axis=1
# )
# gamma_depths = np.concatenate(
#     (gamma_energies[:, :2][:shwr_per_file], gamma_depths), axis=1
# )

# muon_depths = np.concatenate(
#     (muon_energies[:, :2][:shwr_per_file], muon_depths), axis=1
# )
#%% make composite showers

single_shwrs = np.concatenate(
    (scaled_elec_char, scaled_gamma_char, scaled_pion_char), axis=0
)
single_shwrs = single_shwrs[single_shwrs[:, 0].argsort()]

# get unique event numbers, the index at which each event group starts
# and number of showers in each event
grps, idx, num_showers_in_evt = np.unique(
    single_shwrs[:, 0], return_index=True, return_counts=True, axis=0
)
unique_decay_codes = np.take(single_shwrs[:, 1], idx)

# sum each column up until the row index of where the new group starts
# and tack on the decay codes
composite_showers = np.column_stack(
    (
        grps,
        unique_decay_codes,
        np.add.reduceat(single_shwrs[:, 2:], idx),
    )
)
#%%
# clean up depths
# depths = np.concatenate((elec_depths, gamma_depths, pion_depths), axis=0)
# depth = depths[depths[:, 0].argsort()]
# sum_of_each_row = np.nansum(np.abs(depths[:, 2:]), axis=1)
# longest_shower_in_event_idxs = numpy_argmax_reduceat(sum_of_each_row, idx)
# composite_depths = np.take(depths, longest_shower_in_event_idxs, axis=0)
#%% plot
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
decay_channels = np.unique(composite_showers[:, 1])  # [0:2]

fig, ax = plt.subplots(nrows=4, ncols=8, dpi=400, figsize=(15, 8))
# plt.subplots_adjust(hspace=0)
ax = ax.ravel()

for i, dc in enumerate(decay_channels[:-1], start=0):

    x = depth_to_alt(5, pion_depths[0, :])
    y = composite_showers[composite_showers[:, 1] == dc]

    for shower in y:
        # iterate and plot each shower in that decay channel
        event_num = shower[0]
        decay_code = shower[1]
        ax[i].plot(
            x,
            # x[2:],
            np.log10(shower[2:]),
            alpha=0.3,
            linewidth=1.0,
            # label = str(event_num)+"|"+ str(decay_code)
        )

    decay_channel = get_decay_channel(dc)
    ax[i].text(0.1, 0.2, decay_channel, transform=ax[i].transAxes, va="top")
    # ax[i].set(xscale="log")
    # ax[i].title("{}".format(decay_channel))
    # plt.legend()

    # plt.ylabel('N Particles')
    # plt.xlabel('Slant Depth')

    # ax[i].set(yscale="log")
fig.text(
    0.1,
    0.5,
    r"log N",
    va="center",
    rotation="vertical",
)
# fig.text(
#     0.5,
#     0.08,
#     r"Slant Depth (g cm$^{-2}$)",
#     ha="center",
# )
fig.text(
    0.5,
    0.08,
    r"Altitude (km)",
    ha="center",
)
fig.text(
    0.5,
    0.90,
    r"Charged Components",
    ha="center",
)
plt.savefig(
    "./composite_charged_alt_unlog.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08,
)
#%%
# fig, ax = plt.subplots(
#     nrows=2, ncols=2, dpi=400, figsize=(6, 7), sharey=True, sharex=True
# )
# plt.subplots_adjust(hspace=0, wspace=0)
# ax[0, 0].plot(elec_depths.T, elec_char.T, lw=0.8, alpha=0.5)
# ax[0, 0].text(0.5, 0.8, "electron initiated", transform=ax[0, 0].transAxes, va="top")
# # ax[0, 1].plot(muon_depths.T, muon_char.T, lw=0.8, alpha=0.5)
# ax[0, 1].text(0.5, 0.8, "muon initiated", transform=ax[0, 1].transAxes, va="top")
# ax[1, 0].plot(pion_depths.T, pion_char.T, lw=0.8, alpha=0.5)
# ax[1, 0].text(0.5, 0.8, "pion initiated", transform=ax[1, 0].transAxes, va="top")

# ax[1, 1].plot(gamma_depths.T, gamma_char.T, lw=0.8, alpha=0.5)
# ax[1, 1].text(0.5, 0.8, "gamma initiated", transform=ax[1, 1].transAxes, va="top")

# ax[0, 0].set(yscale="log")

# decay_channel_mult_plt(bins=composite_depths, showers=composite_showers)
