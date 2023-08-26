"""
Energy scaling; just point it to tup_folder.

The elongation rate tables will be saved at

nuSpaceSim/src/nuspacesim/data/eas_reco/energy_scaling

"""


import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files
# %%


# tup_folder = "../conex_7_50_runs"
tup_folder = "/home/fabg/gdrive_umd/Research/NASA/Work/conex2r7_50-runs/1000_evts"
# tup_folder = r"G:\My Drive\Research\NASA\Work\conex2r7_50-runs\downward"

ntuples = sorted(os.listdir(tup_folder))  # [1:]

energies = []
angles = []

mean_mus_nmax = []
mean_mus_xmax = []

mean_char_nmax = []
mean_char_xmax = []

mean_el_nmax = []
mean_el_xmax = []

mean_had_nmax = []
mean_had_xmax = []

mean_gam_nmax = []
mean_gam_xmax = []


for tup in ntuples:
    log_energy = int(tup.split("_")[1])
    beta = int(tup.split("_")[4])
    # print(beta)
    energies.append(log_energy)
    angles.append(beta)

    shwr_data = ReadConex(os.path.join(tup_folder, tup))
    # fits = shwr_data.gh_fits()
    # alts = shwr_data.get_alts()
    depths = shwr_data.get_depths()

    mus = shwr_data.get_muons()
    char = shwr_data.get_charged()
    el = shwr_data.get_elec()
    had = shwr_data.get_hadrons()
    gam = shwr_data.get_gamma()

    mus_nmaxs, mus_xmaxs = bin_nmax_xmax(depths, mus)
    char_nmaxs, char_xmaxs = bin_nmax_xmax(depths, char)
    el_nmaxs, el_xmaxs = bin_nmax_xmax(depths, el)
    had_nmaxs, had_xmaxs = bin_nmax_xmax(depths, had)
    gam_nmaxs, gam_xmaxs = bin_nmax_xmax(depths, gam)

    mean_mus_nmax.append(mus_nmaxs.mean())
    mean_mus_xmax.append(mus_xmaxs.mean())

    mean_char_nmax.append(char_nmaxs.mean())
    mean_char_xmax.append(char_xmaxs.mean())

    mean_el_nmax.append(el_nmaxs.mean())
    mean_el_xmax.append(el_xmaxs.mean())

    mean_had_nmax.append(had_nmaxs.mean())
    mean_had_xmax.append(had_xmaxs.mean())

    mean_gam_nmax.append(gam_nmaxs.mean())
    mean_gam_xmax.append(gam_xmaxs.mean())

energies = np.array(energies)
angles = np.array(angles)
mean_mus_nmax = np.array(mean_mus_nmax)
mean_mus_xmax = np.array(mean_mus_xmax)
mean_char_nmax = np.array(mean_char_nmax)
mean_char_xmax = np.array(mean_char_xmax)
mean_el_nmax = np.array(mean_el_nmax)
mean_el_xmax = np.array(mean_el_xmax)
mean_had_nmax = np.array(mean_had_nmax)
mean_had_xmax = np.array(mean_had_xmax)
mean_gam_nmax = np.array(mean_gam_nmax)
mean_gam_xmax = np.array(mean_gam_xmax)


# %%
def lin_func(x, m, b):
    return (m * x) + b


plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)

ptype = ["muons", "charged", "e-+", "hadrons", "gammas"]
mtype = ["^", "s", "x", "o", "+"]

three_angles = []
particle_types = []
slope = []
slope_uncertainty = []
intercept = []
intercept_uncertainty = []

fig, ax = plt.subplots(
    ncols=3,
    nrows=2,
    sharex=True,
    sharey=True,
    figsize=(9, 5),
    dpi=300,
)
ax = ax.ravel()
plt.subplots_adjust(wspace=0, hspace=0)
for angle_idx, a in enumerate(sorted(list(set(angles)))):
    angle = a
    mask = angles == angle
    masked_energies = energies[mask]
    masked_mus_nmax = mean_mus_nmax[mask]
    masked_mus_xmax = mean_mus_xmax[mask]
    masked_char_nmax = mean_char_nmax[mask]
    masked_char_xmax = mean_char_xmax[mask]
    masked_el_nmax = mean_el_nmax[mask]
    masked_el_xmax = mean_el_xmax[mask]
    masked_had_nmax = mean_had_nmax[mask]
    masked_had_xmax = mean_had_xmax[mask]
    masked_gam_nmax = mean_gam_nmax[mask]
    masked_gam_xmax = mean_gam_xmax[mask]

    particle_xmaxs = [
        masked_mus_xmax,
        masked_char_xmax,
        masked_el_xmax,
        masked_had_xmax,
        masked_gam_xmax,
    ]
    particle_nmaxs = [
        masked_mus_nmax,
        masked_char_nmax,
        masked_el_nmax,
        masked_had_nmax,
        masked_gam_nmax,
    ]

    for idx, p in enumerate(ptype):
        params, uncertainty = curve_fit(
            f=lin_func,
            xdata=masked_energies,
            ydata=np.log10(particle_nmaxs[idx]),
        )
        uncertainties = np.sqrt(np.diag(uncertainty))

        theory_x = np.linspace(10, 21, 100)
        ax[angle_idx].plot(theory_x, lin_func(theory_x, *params), ls="--", alpha=0.8)

        if angle_idx == 0:
            mus_scatter = ax[angle_idx].scatter(
                masked_energies,
                np.log10(particle_nmaxs[idx]),
                marker=mtype[idx],
                label=r"${{\rm {} }} \: ({:.1f}, {:.1f})$".format(p, *params),
                alpha=0.5,
            )
        else:
            mus_scatter = ax[angle_idx].scatter(
                masked_energies,
                np.log10(particle_nmaxs[idx]),
                marker=mtype[idx],
                label=r"$({:.1f}, {:.1f})$".format(*params),
                alpha=0.5,
            )

        three_angles.append(a)
        particle_types.append(p)
        slope.append(params[0])
        slope_uncertainty.append(uncertainties[0])
        intercept.append(params[1])
        intercept_uncertainty.append(uncertainties[1])

    if angle_idx == 0:
        ax[angle_idx].legend(
            title=r"${\rm component\: (slope,\:intercept)}$",
            fontsize=8,
            title_fontsize=8,
        )
    else:
        ax[angle_idx].legend(fontsize=8, title_fontsize=8, ncols=2)

    ax[angle_idx].text(
        0.95,
        0.05,
        r"${{\rm \beta = {} \degree}}$".format(angle),
        transform=ax[angle_idx].transAxes,
        ha="right",
        va="bottom",
    )

    ax[angle_idx].set(ylim=(2, 15), xlim=(13.5, 20.5))
    ax[angle_idx].grid(ls="--")


fig.text(0.5, 0.08, r"${\rm \log_{10} \: Energy \:({\rm eV})}$", ha="center")
fig.text(
    0.07,
    0.5,
    r"$\log_{10}\: {\rm mean}\:N_{\rm max}$",
    va="center",
    rotation="vertical",
)

# fig.text(
#     0.1,
#     0.90,
#     tup_folder,
#     ha="left"
# )

# fig.text(
#     0.025,
#     0.5,
#     r"log mean Nmax (g/cm$^2$)",
#     va="center",
#     rotation="vertical",
# )
plt.savefig(
    "../../../../../../gdrive_umd/Research/NASA/energy_scaling.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.0,
)
# %% save

three_angles = np.array(three_angles)
particle_types = np.array(particle_types)
slope = np.array(slope)
slope_uncertainty = np.array(slope_uncertainty)
intercept = np.array(intercept)
intercept_uncertainty = np.array(intercept_uncertainty)

ptypes = ["muons", "charged", "e-+", "hadrons", "gammas"]
earth_emer_angles = sorted(list(set(angles)))
fname = tup_folder.split("/")[-1]
with as_file(files("nuspacesim.data.eas_reco.energy_scaling") / f"{fname}.h5") as path:
    print(path)
    with h5py.File(path, "w") as f:
        for t in ptypes:
            # aggregate across earth emergence angles
            particle_data = []

            for ang in earth_emer_angles:
                ang_data = np.concatenate(
                    (
                        np.array([ang]),
                        slope[(three_angles == ang) & (particle_types == t)],
                        slope_uncertainty[
                            (three_angles == ang) & (particle_types == t)
                        ],
                        intercept[(three_angles == ang) & (particle_types == t)],
                        intercept_uncertainty[
                            (three_angles == ang) & (particle_types == t)
                        ],
                    )
                )
                print(ang_data)
                particle_data.append(ang_data)

            f.create_dataset(
                t,
                data=np.array(particle_data),
                dtype="f",
            )
