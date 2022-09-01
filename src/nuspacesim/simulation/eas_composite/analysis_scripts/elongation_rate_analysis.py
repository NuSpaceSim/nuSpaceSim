import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py


tup_folder = "../conex_7_50_runs"
ntuples = sorted(os.listdir(tup_folder))[:-1]

energies = []
angles = []

mean_mus_nmax = []
mean_mus_xmax = []

mean_char_nmax = []
mean_char_xmax = []

mean_had_nmax = []
mean_had_xmax = []

mean_gam_nmax = []
mean_gam_xmax = []


for tup in ntuples:
    log_energy = int(tup.split("_")[1])
    beta = int(tup.split("_")[4])

    energies.append(log_energy)
    angles.append(beta)

    shwr_data = ReadConex(os.path.join(tup_folder, tup))
    # fits = shwr_data.gh_fits()
    # alts = shwr_data.get_alts()
    depths = shwr_data.get_depths()

    mus = shwr_data.get_muons()
    char = shwr_data.get_charged()
    el = shwr_data.get_elec_pos()
    had = shwr_data.get_hadrons()
    gam = shwr_data.get_gamma()

    mus_nmaxs, mus_xmaxs = bin_nmax_xmax(depths, mus)
    char_nmaxs, char_xmaxs = bin_nmax_xmax(depths, char)
    # el_nmaxs, el_xmaxs = bin_nmax_xmax(depths, el)
    had_nmaxs, had_xmaxs = bin_nmax_xmax(depths, had)
    gam_nmaxs, gam_xmaxs = bin_nmax_xmax(depths, gam)

    mean_mus_nmax.append(mus_nmaxs.mean())
    mean_mus_xmax.append(mus_xmaxs.mean())

    mean_char_nmax.append(char_nmaxs.mean())
    mean_char_xmax.append(char_xmaxs.mean())

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
mean_had_nmax = np.array(mean_had_nmax)
mean_had_xmax = np.array(mean_had_xmax)
mean_gam_nmax = np.array(mean_gam_nmax)
mean_gam_xmax = np.array(mean_gam_xmax)


def lin_func(x, m, b):
    return (m * x) + b


# plot_type = "log_log_nmax_vs_xmax"
# plot_type = "lin_log_xmax_vs_energy"
plot_type = "log_log_nmax_vs_energy"

ptype = ["muons", "charged", "hadrons", "gammas"]
mtype = ["^", "x", "s", "o"]
if plot_type == "log_log_nmax_vs_xmax":
    # plot mean nmax as a function of mean xmax with a color bar indicating energy.

    three_angles = []
    particle_types = []
    slope = []
    slope_uncertainty = []
    intercept = []
    intercept_uncertainty = []

    for a in sorted(list(set(angles))):
        angle = a
        mask = angles == angle
        masked_energies = energies[mask]
        masked_mus_nmax = mean_mus_nmax[mask]
        masked_mus_xmax = mean_mus_xmax[mask]
        masked_char_nmax = mean_char_nmax[mask]
        masked_char_xmax = mean_char_xmax[mask]
        masked_had_nmax = mean_had_nmax[mask]
        masked_had_xmax = mean_had_xmax[mask]
        masked_gam_nmax = mean_gam_nmax[mask]
        masked_gam_xmax = mean_gam_xmax[mask]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), dpi=300)
        cmap = plt.cm.get_cmap("plasma")
        particle_xmaxs = [
            masked_mus_xmax,
            masked_char_xmax,
            masked_had_xmax,
            masked_gam_xmax,
        ]
        particle_nmaxs = [
            masked_mus_nmax,
            masked_char_nmax,
            masked_had_nmax,
            masked_gam_nmax,
        ]

        for idx, p in enumerate(ptype):
            params, uncertainty = curve_fit(
                f=lin_func,
                xdata=np.log10(particle_xmaxs[idx]),
                ydata=np.log10(particle_nmaxs[idx]),
            )
            uncertainties = np.sqrt(np.diag(uncertainty))
            theory_x = np.linspace(2.5, 3.2, 100)
            ax.plot(
                theory_x, lin_func(theory_x, *params), ls="--", color="grey", alpha=0.8
            )
            mus_scatter = ax.scatter(
                np.log10(particle_xmaxs[idx]),
                np.log10(particle_nmaxs[idx]),
                marker=mtype[idx],
                c=masked_energies,
                cmap=cmap,
                label=r"{} ($\log_{{10}}(y)$ = {:.2f} $\log_{{10}}(x) $+ {:.2f})".format(
                    p, *params
                ),
            )

            three_angles.append(a)
            particle_types.append(p)
            slope.append(params[0])
            slope_uncertainty.append(uncertainties[0])
            intercept.append(params[1])
            intercept_uncertainty.append(uncertainties[1])

        cbar = plt.colorbar(mus_scatter, pad=0)
        cbar.set_label(label="log Energy (eV)")

        ax.set_xlabel("log10 mean xmax $(g/cm^2)$")
        ax.set_ylabel("log10 mean nmax (N)")
        # ax.set_xscale("log")
        # ax.set_yscale("log")

        # ax.set_xlim(100, 3000)
        ax.legend(
            title=r"$\beta = {} \degree$".format(angle),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.4),
            fontsize=8,
        )


if plot_type == "lin_log_xmax_vs_energy":

    three_angles = []
    particle_types = []
    slope = []
    slope_uncertainty = []
    intercept = []
    intercept_uncertainty = []

    for a in sorted(list(set(angles))):
        angle = a
        mask = angles == angle
        masked_energies = energies[mask]
        masked_mus_nmax = mean_mus_nmax[mask]
        masked_mus_xmax = mean_mus_xmax[mask]
        masked_char_nmax = mean_char_nmax[mask]
        masked_char_xmax = mean_char_xmax[mask]
        masked_had_nmax = mean_had_nmax[mask]
        masked_had_xmax = mean_had_xmax[mask]
        masked_gam_nmax = mean_gam_nmax[mask]
        masked_gam_xmax = mean_gam_xmax[mask]

        particle_xmaxs = [
            masked_mus_xmax,
            masked_char_xmax,
            masked_had_xmax,
            masked_gam_xmax,
        ]
        particle_nmaxs = [
            masked_mus_nmax,
            masked_char_nmax,
            masked_had_nmax,
            masked_gam_nmax,
        ]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), dpi=300)
        for idx, p in enumerate(ptype):

            params, uncertainty = curve_fit(
                f=lin_func,
                xdata=masked_energies,
                ydata=particle_xmaxs[idx],
            )
            uncertainties = np.sqrt(np.diag(uncertainty))
            theory_x = np.linspace(14, 21, 100)
            ax.plot(theory_x, lin_func(theory_x, *params), ls="--", alpha=0.8)
            mus_scatter = ax.scatter(
                masked_energies,
                particle_xmaxs[idx],
                marker=mtype[idx],
                label=r"{} (y = {:.2f} $\log_{{10}}(x)$ + {:.2f})".format(p, *params),
            )

            three_angles.append(a)
            particle_types.append(p)
            slope.append(params[0])
            slope_uncertainty.append(uncertainties[0])
            intercept.append(params[1])
            intercept_uncertainty.append(uncertainties[1])

        ax.set_xlabel("log10 Energy $(eV)$")
        ax.set_ylabel("mean Xmax (N)")
        # ax.set_xscale("log")
        # ax.set_yscale("log")

        # ax.set_xlim(100, 3000)
        ax.legend(
            title=r"$\beta = {} \degree$".format(angle),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.4),
            fontsize=8,
        )


if plot_type == "log_log_nmax_vs_energy":
    three_angles = []
    particle_types = []
    slope = []
    slope_uncertainty = []
    intercept = []
    intercept_uncertainty = []

    for a in sorted(list(set(angles))):
        angle = a
        mask = angles == angle
        masked_energies = energies[mask]
        masked_mus_nmax = mean_mus_nmax[mask]
        masked_mus_xmax = mean_mus_xmax[mask]
        masked_char_nmax = mean_char_nmax[mask]
        masked_char_xmax = mean_char_xmax[mask]
        masked_had_nmax = mean_had_nmax[mask]
        masked_had_xmax = mean_had_xmax[mask]
        masked_gam_nmax = mean_gam_nmax[mask]
        masked_gam_xmax = mean_gam_xmax[mask]

        particle_xmaxs = [
            masked_mus_xmax,
            masked_char_xmax,
            masked_had_xmax,
            masked_gam_xmax,
        ]
        particle_nmaxs = [
            masked_mus_nmax,
            masked_char_nmax,
            masked_had_nmax,
            masked_gam_nmax,
        ]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), dpi=300)
        for idx, p in enumerate(ptype):

            params, uncertainty = curve_fit(
                f=lin_func,
                xdata=masked_energies,
                ydata=np.log10(particle_nmaxs[idx]),
            )
            uncertainties = np.sqrt(np.diag(uncertainty))

            theory_x = np.linspace(14, 21, 100)
            ax.plot(theory_x, lin_func(theory_x, *params), ls="--", alpha=0.8)
            mus_scatter = ax.scatter(
                masked_energies,
                np.log10(particle_nmaxs[idx]),
                marker=mtype[idx],
                label=r"{} ($\log_{{10}}(y)$ = {:.2f} $\log_{{10}}(x)$ + {:.2f})".format(
                    p, *params
                ),
            )

            three_angles.append(a)
            particle_types.append(p)
            slope.append(params[0])
            slope_uncertainty.append(uncertainties[0])
            intercept.append(params[1])
            intercept_uncertainty.append(uncertainties[1])

        ax.set_xlabel("log10 Energy $(eV)$")
        ax.set_ylabel("log10  mean Nmax (N)")

        # ax.set_xlim(100, 3000)
        ax.legend(
            title=r"$\beta = {} \degree$".format(angle),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.4),
            fontsize=8,
        )

#%% save


three_angles = np.array(three_angles)
particle_types = np.array(particle_types)
slope = np.array(slope)
slope_uncertainty = np.array(slope_uncertainty)
intercept = np.array(intercept)
intercept_uncertainty = np.array(intercept_uncertainty)

muon_1deg_data = np.concatenate(
    (
        np.array([1]),
        slope[(three_angles == 1) & (particle_types == "muons")],
        slope_uncertainty[(three_angles == 1) & (particle_types == "muons")],
        intercept[(three_angles == 1) & (particle_types == "muons")],
        intercept_uncertainty[(three_angles == 1) & (particle_types == "muons")],
    )
)

muon_5deg_data = np.concatenate(
    (
        np.array([5]),
        slope[(three_angles == 5) & (particle_types == "muons")],
        slope_uncertainty[(three_angles == 5) & (particle_types == "muons")],
        intercept[(three_angles == 5) & (particle_types == "muons")],
        intercept_uncertainty[(three_angles == 5) & (particle_types == "muons")],
    )
)

muon_35deg_data = np.concatenate(
    (
        np.array([35]),
        slope[(three_angles == 35) & (particle_types == "muons")],
        slope_uncertainty[(three_angles == 35) & (particle_types == "muons")],
        intercept[(three_angles == 35) & (particle_types == "muons")],
        intercept_uncertainty[(three_angles == 35) & (particle_types == "muons")],
    )
)

charged_1deg_data = np.concatenate(
    (
        np.array([1]),
        slope[(three_angles == 1) & (particle_types == "charged")],
        slope_uncertainty[(three_angles == 1) & (particle_types == "charged")],
        intercept[(three_angles == 1) & (particle_types == "charged")],
        intercept_uncertainty[(three_angles == 1) & (particle_types == "charged")],
    )
)

charged_5deg_data = np.concatenate(
    (
        np.array([5]),
        slope[(three_angles == 5) & (particle_types == "charged")],
        slope_uncertainty[(three_angles == 5) & (particle_types == "charged")],
        intercept[(three_angles == 5) & (particle_types == "charged")],
        intercept_uncertainty[(three_angles == 5) & (particle_types == "charged")],
    )
)

charged_35deg_data = np.concatenate(
    (
        np.array([35]),
        slope[(three_angles == 35) & (particle_types == "charged")],
        slope_uncertainty[(three_angles == 35) & (particle_types == "charged")],
        intercept[(three_angles == 35) & (particle_types == "charged")],
        intercept_uncertainty[(three_angles == 35) & (particle_types == "charged")],
    )
)

hadrons_1deg_data = np.concatenate(
    (
        np.array([1]),
        slope[(three_angles == 1) & (particle_types == "hadrons")],
        slope_uncertainty[(three_angles == 1) & (particle_types == "hadrons")],
        intercept[(three_angles == 1) & (particle_types == "hadrons")],
        intercept_uncertainty[(three_angles == 1) & (particle_types == "hadrons")],
    )
)

hadrons_5deg_data = np.concatenate(
    (
        np.array([5]),
        slope[(three_angles == 5) & (particle_types == "hadrons")],
        slope_uncertainty[(three_angles == 5) & (particle_types == "hadrons")],
        intercept[(three_angles == 5) & (particle_types == "hadrons")],
        intercept_uncertainty[(three_angles == 5) & (particle_types == "hadrons")],
    )
)

hadrons_35deg_data = np.concatenate(
    (
        np.array([35]),
        slope[(three_angles == 35) & (particle_types == "hadrons")],
        slope_uncertainty[(three_angles == 35) & (particle_types == "hadrons")],
        intercept[(three_angles == 35) & (particle_types == "hadrons")],
        intercept_uncertainty[(three_angles == 35) & (particle_types == "hadrons")],
    )
)


gammas_1deg_data = np.concatenate(
    (
        np.array([1]),
        slope[(three_angles == 1) & (particle_types == "gammas")],
        slope_uncertainty[(three_angles == 1) & (particle_types == "gammas")],
        intercept[(three_angles == 1) & (particle_types == "gammas")],
        intercept_uncertainty[(three_angles == 1) & (particle_types == "gammas")],
    )
)

gammas_5deg_data = np.concatenate(
    (
        np.array([5]),
        slope[(three_angles == 5) & (particle_types == "gammas")],
        slope_uncertainty[(three_angles == 5) & (particle_types == "gammas")],
        intercept[(three_angles == 5) & (particle_types == "gammas")],
        intercept_uncertainty[(three_angles == 5) & (particle_types == "gammas")],
    )
)

gammas_35deg_data = np.concatenate(
    (
        np.array([35]),
        slope[(three_angles == 35) & (particle_types == "gammas")],
        slope_uncertainty[(three_angles == 35) & (particle_types == "gammas")],
        intercept[(three_angles == 35) & (particle_types == "gammas")],
        intercept_uncertainty[(three_angles == 35) & (particle_types == "gammas")],
    )
)

with h5py.File("{}.h5".format(plot_type), "w") as f:

    f.create_dataset(
        "muons",
        data=np.vstack((muon_1deg_data, muon_5deg_data, muon_35deg_data)),
        dtype="f",
    )

    f.create_dataset(
        "charged",
        data=np.vstack((charged_1deg_data, charged_5deg_data, charged_35deg_data)),
        dtype="f",
    )

    f.create_dataset(
        "hadrons",
        data=np.vstack((hadrons_1deg_data, hadrons_5deg_data, hadrons_35deg_data)),
        dtype="f",
    )

    f.create_dataset(
        "gammas",
        data=np.vstack((gammas_1deg_data, gammas_5deg_data, gammas_35deg_data)),
        dtype="f",
    )
