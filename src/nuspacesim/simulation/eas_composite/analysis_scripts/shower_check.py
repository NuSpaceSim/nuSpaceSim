import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py

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

# tup_folder = r"G:\My Drive\Research\NASA\Work\conex2r7_50-runs\1000_evts"
tup_folder = "~/g_drive/Research/NASA/Work/conex2r7_50-runs/1000_evts"
# up_15 = "log_17_eV_1000shwrs_15_degearthemergence_eposlhc_215240113_100.root"
# up_15 = "log_16_eV_1000shwrs_15_degearthemergence_eposlhc_1728792184_100.root"
up_15 = "log_15_eV_1000shwrs_15_degearthemergence_eposlhc_1775946518_100.root"
up_15_data = ReadConex(os.path.join(tup_folder, up_15))

tup_folder = "~/g_drive/Research/NASA/Work/conex2r7_50-runs/downward"
# down_60 = "log_17_eV_1000shwrs_60_downward_eposlhc_1756896908_100.root"
# down_60 = "log_16_eV_1000shwrs_60_downward_eposlhc_272473279_100.root"
down_60 = "log_15_eV_1000shwrs_60_downward_eposlhc_842303273_100.root"
down_60_data = ReadConex(os.path.join(tup_folder, down_60))

down_mus = down_60_data.get_muons()
down_char = down_60_data.get_charged()
down_depths = down_60_data.get_depths()

up_mus = up_15_data.get_muons()
up_char = up_15_data.get_charged()
up_depths = up_15_data.get_depths()

fig, ax = plt.subplots(nrows=2, ncols=2, dpi=400, figsize=(6, 7), sharey=True)
ax[0, 0].plot(down_depths.T, down_char.T, lw=0.8, alpha=0.8)
ax[0, 1].plot(down_depths.T, down_mus.T, lw=0.8, alpha=0.8)

ax[1, 0].plot(up_depths.T, up_char.T, lw=0.8, alpha=0.8)
ax[1, 1].plot(up_depths.T, up_mus.T, lw=0.8, alpha=0.8)

titles = [
    "Charged 60 deg. down ",
    "Muon 60 deg. down ",
    "Charged 15 deg. up",
    "Muon 15 deg. up ",
]

for x, axis in enumerate(ax.flatten()):
    axis.set(
        yscale="log",
        xlabel="Slant Depth ($g/cm^2$)",
        ylim=(10, 1e9),
        ylabel="$N$",
        title=titles[x],
    )
