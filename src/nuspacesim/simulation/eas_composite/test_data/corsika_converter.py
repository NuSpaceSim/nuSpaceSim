import re
import numpy as np
import matplotlib.pyplot as plt
from nuspacesim.simulation.eas_composite.shower_long_profiles import (
    ShowerParameterization,
)
import matplotlib as mpl

#%%
"""
reading corsika 77420 binary files for gh hillas and actual particle content
"""


in_file = "./corsika-77420/downward_proton_05evts_85degzenith_05kmobs_0gcm2starting_1e8gev.txt"


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
    positrons = data[:, 2]
    electrons = data[:, 3]
    charged = data[:, 7]

    showers.append((num, gh, depths, charged, positrons, electrons))


# plt.figure(figsize=(6, 4), dpi=300)
# plt.scatter(depths, positrons, label="positrons", s=1)
# plt.scatter(depths, electrons, label="electrons", s=1)
# plt.scatter(depths, charged, label="charged", s=1)

# plt.title("Shower {}".format(num))
# plt.yscale("log")
# plt.legend()

#%%

for shwr in showers:
    plt.figure(figsize=(6, 4), dpi=300)
    shwr_num = shwr[0]
    gh_params = shwr[1]
    corsika_depths = shwr[2]
    corsika_charge = shwr[3]
    corsika_positron = shwr[4]
    corsika_electrons = shwr[5]
    corsika_combined = corsika_charge + corsika_positron + corsika_electrons

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
        shower_end=6000,
        grammage=1,
    )

    plt.scatter(corsika_depths, corsika_charge, label="charged", s=2, alpha=0.5)
    plt.scatter(corsika_depths, corsika_positron, label="e-", s=2, alpha=0.5)
    plt.scatter(corsika_depths, corsika_positron, label="e+", s=2, alpha=0.5)

    plt.scatter(corsika_depths, corsika_combined, label="combined", s=2, alpha=0.5)

    plt.plot(depth, shower_content, "--k", label="GH Fit")
    # plt.title("Shower {}".format(shwr_num))
    # upward_proton_20evts_95degzenith_15kmobs_1030gcm2starting_1e8gev.txt"
    plt.title(r"Downward Proton, $\theta_{zenith} = 85\degree$, 5 km obs ")
    plt.xlabel(r"$g \: cm^{-2}$")
    plt.ylabel(r"$N$")

    plt.xlim(-100, 2000)
    # plt.ylim(1, 5e8)
    plt.yscale("log")

    y_minor = mpl.ticker.LogLocator(
        base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
    )
    ax = plt.gca()
    ax.yaxis.set_minor_locator(y_minor)
    plt.tick_params(axis="both", which="both")
    plt.tick_params(axis="y", which="minor")
    # plt.grid(True, which="both", linestyle="--")

    plt.legend()
