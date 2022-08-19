import re
import numpy as np
import matplotlib.pyplot as plt
from nuspacesim.simulation.eas_composite.shower_long_profiles import (
    ShowerParameterization,
)
from nuspacesim.simulation.eas_composite.x_to_z_lookup import (
    depth_to_alt_lookup,
    depth_to_alt_lookup_v2,
)
from nuspacesim.simulation.eas_composite.depth_to_altitude import (
    depth_to_altitude,
    slant_depth_to_depth,
)
from nuspacesim.simulation.eas_optical.atmospheric_models import (
    cummings_atmospheric_density,
)
import matplotlib as mpl

#%%
"""
reading corsika 77420 binary files for gh hillas and actual particle content
"""
corsika_angle = 95
observing_height = 5
start_depth = 1030
azimuthal_angle = 85
start_z = float(depth_to_altitude(np.array([start_depth])))  # km
shower_type = "Upward Proton Primary"
direction = "up"
in_file = (
    "./corsika-77420/new_runs/up_proton_1e8gev_theta95deg_start1030gcm2_obs5km.txt"
)


def read_corsika_binary(in_file):
    with open(in_file) as f:
        shower_nums = []
        gh_params = []
        showers = []
        shower_content = []
        for line in f:
            shower_content.append(line)
            if "LONGITUDINAL ENERGY DEPOSIT IN " in line or "AV. DEVIATION IN" in line:
                try:
                    shower_nums.append(int(line.split()[-1]))
                except:
                    pass
                showers.append(shower_content)
                shower_content = []
            if "PARAMETERS" in line:
                gh_params.append(np.array(line.split()[2:], dtype=np.float64))
    distributions = showers[::2]
    deposits = showers[1::2]
    cleaned_distributions = []
    for shwr in distributions:
        one_shower = []
        for line in shwr:
            if r"D" in line or line.isspace():  # get common letter in the string
                pass
            else:
                one_shower.append(np.array(line.split(), dtype=np.float64))
        cleaned_distributions.append(np.array(one_shower))
    # only return distributions not necessarily energy deposits.
    return np.array(shower_nums), cleaned_distributions, gh_params


shower_nums, cleaned_distributions, gh_params = read_corsika_binary(in_file)

showers = []

for num, dists, gh in zip(shower_nums, cleaned_distributions, gh_params):
    data = np.array(dists)
    # see the data files for what each column contains

    depths = data[:, 0]
    gammas = data[:, 1]
    positrons = data[:, 2]
    electrons = data[:, 3]
    mu_plus = data[:, 4]
    muons = data[:, 5]
    hadrons = data[:, 6]
    charged = data[:, 7]

    showers.append((num, gh, depths, charged, positrons, electrons, muons, hadrons))


# plt.figure(figsize=(6, 4), dpi=300)
# plt.scatter(depths, positrons, label="positrons", s=1)
# plt.scatter(depths, electrons, label="electrons", s=1)
# plt.scatter(depths, charged, label="charged", s=1)

# plt.title("Shower {}".format(num))
# plt.yscale("log")
# plt.legend()

#%%

# for shwr in showers:
#     plt.figure(figsize=(6, 4), dpi=300)
#     shwr_num = shwr[0]
#     gh_params = shwr[1]
#     corsika_depths = shwr[2]
#     corsika_charge = shwr[3]
#     corsika_positron = shwr[4]
#     corsika_electrons = shwr[5]
#     corsika_muons = shwr[6]
#     # corsika_combined = corsika_charge + corsika_positron + corsika_electrons

#     nmax = gh_params[0]
#     x0 = gh_params[1]
#     xmax = gh_params[2]
#     p1 = gh_params[3]
#     p2 = gh_params[4]
#     p3 = gh_params[5]

#     shower = ShowerParameterization(
#         table_decay_e=1,
#         event_tag=1,
#         decay_tag=1,
#     )

#     depth, shower_content = shower.gaisser_hillas(
#         n_max=nmax,
#         x_max=xmax,
#         x_0=x0,
#         p1=p1,
#         p2=p2,
#         p3=p3,
#         shower_end=6000,
#         grammage=1,
#     )

#     plt.scatter(corsika_depths, corsika_charge, label="Charged", s=2, alpha=0.5)
#     plt.scatter(corsika_depths, corsika_electrons, label=r"e^{-}", s=2, alpha=0.5)
#     plt.scatter(corsika_depths, corsika_positron, label=r"e^{+}", s=2, alpha=0.5)

#     plt.scatter(corsika_depths, corsika_muons, label=r"$\mu^{-}$", s=2, alpha=0.5)

#     plt.plot(depth, shower_content, "--k", label="GH Fit")

#     # upward_proton_20evts_95degzenith_15kmobs_1030gcm2starting_1e8gev.txt"
#     plt.title(r"Upward Shower, Proton Primary  ")
#     plt.xlabel(r"$g \: cm^{-2}$")
#     plt.ylabel(r"$N$")

#     # plt.xlim(-100, 2000)
#     # plt.ylim(1, 5e8)
#     plt.yscale("log")

#     y_minor = mpl.ticker.LogLocator(
#         base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
#     )
#     ax = plt.gca()
#     ax.yaxis.set_minor_locator(y_minor)
#     plt.tick_params(axis="both", which="both")
#     plt.tick_params(axis="y", which="minor")
#     # plt.grid(True, which="both", linestyle="--")

#     plt.legend()

#%%


for shwr in showers:

    shwr_num = shwr[0]
    gh_params = shwr[1]
    corsika_depths = shwr[2]
    corsika_charge = shwr[3]
    corsika_positron = shwr[4]
    corsika_electron = shwr[5]
    corsika_muon = shwr[6]
    corsika_hadron = shwr[7]
    # corsika_combined = corsika_charge + corsika_positron + corsika_electrons

    nmax = gh_params[0]
    x0 = gh_params[1]
    xmax = gh_params[2]
    p1 = gh_params[3]
    p2 = gh_params[4]
    p3 = gh_params[5]

    shower = ShowerParameterization(
        table_decay_e=1,
        event_tag=1,
        decay_tag=1,
    )

    depth, shower_content = shower.gaisser_hillas(
        n_max=nmax,
        x_max=xmax,
        x_0=x0,
        p1=p1,
        p2=p2,
        p3=p3,
        shower_end=np.max(corsika_depths),
        grammage=1,
    )
    fig, ax = plt.subplots(
        nrows=5,
        ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": [6, 2, 2, 2, 2]},
        figsize=(6, 10),
        dpi=300,
    )
    ax[0].plot(depth, shower_content, "--k", label="GH Fit")
    print(shwr_num)
    ax[0].scatter(corsika_depths, corsika_charge, label="Charged", s=4, alpha=1)

    ax[0].scatter(corsika_depths, corsika_electron, label=r"e$^{-}$", s=4, alpha=1)
    ax[0].scatter(corsika_depths, corsika_positron, label=r"e$^{+}$", s=4, alpha=1)
    ax[0].scatter(corsika_depths, corsika_muon, label=r"$\mu^{-}$", s=4, alpha=1)
    ax[0].scatter(corsika_depths, corsika_hadron, label=r"hadron", s=4, alpha=1)
    # ax[0].set_title(r"Upward Shower, Proton Primary")
    ax[0].set_ylabel(r"$N$")
    ax[3].set_xlabel(r"slant depth (g/cm$^2$)")
    ax[0].set_yscale("log")
    ax[0].set_ylim(bottom=1)
    ax[0].legend(
        title=(
            "{}"
            "\n"
            "Starting Grammage = {} $\mathrm{{g \: cm^{{-2}}}}$"
            "\n"
            r"$\theta_{{\mathrm{{zenith}}}} = {} \degree,"
            "\: z_{{obs}} = {} \mathrm{{km}}$,"
        ).format(shower_type, start_depth, corsika_angle, observing_height),
        ncol=2,
        loc="lower left",
    )

    # muon over electron ratio
    ax[1].scatter(corsika_depths, corsika_muon / corsika_electron, s=4, color="salmon")
    ax[1].set_ylabel(r"$\mu^{-} / e^{-}$")
    ax[1].set_ylim(top=1)
    # ax[1].set_yscale("log")

    altitudes = depth_to_alt_lookup_v2(
        slant_depths=corsika_depths,
        angle=corsika_angle,
        starting_alt=start_z,
        direction=direction,
    )
    vert_depth = slant_depth_to_depth(corsika_depths, azimuthal_angle)
    alt = depth_to_altitude(vert_depth)

    ax[2].plot(corsika_depths, altitudes, color="seagreen", label="new")
    # ax[2].scatter(
    #     corsika_depths, np.flip(alt), s=4, color="darkorange", label="rough vert depth"
    # )  # !!! flipped if upward
    ax[2].set_ylabel(r"Altitude (km)")
    # ax[2].set_ylim(top=50)
    ax[2].legend()

    ax[3].scatter(corsika_depths, np.flip(vert_depth), s=4, color="brown")
    ax[3].set_ylabel(r"Vert. Depth (km)")

    atm_dense = cummings_atmospheric_density(altitudes)
    ax[4].scatter(corsika_depths, atm_dense, s=4, color="mediumpurple")
    # ax[3].set_ylim(top=0.0007)
    ax[4].set_ylabel(r"$\mathrm{\rho_{atm}} \: \mathrm{(g \: cm^{-3})}$")
    # ax[3].set_yscale("log")

    ax[0].set_xlim(left=corsika_depths.min())
    plt.subplots_adjust(hspace=0)
    # plt.legend()
