import uproot
import numpy as np
from nuspacesim.simulation.eas_composite.shower_long_profiles import (
    ShowerParameterization,
)
import matplotlib.pyplot as plt

#%%
def conex_to_text(file_path: str, output_file: str, num_showers=1):
    ntuple = uproot.open(file_path)

    params = []
    for i in range(0, num_showers):

        shwr = ntuple["Shower;{}".format(i + 1)]
        lg_10_e = shwr["lgE"].array(library="np")
        zenith_ang_deg = shwr["zenith"].array(library="np")
        azimuth_ang_deg = shwr["azimuth"].array(library="np")

        gh_n_max = shwr["Nmax"].array(library="np")
        gh_x_max = shwr["Xmax"].array(library="np")
        gh_x0 = shwr["X0"].array(library="np")
        gh_p1 = shwr["p1"].array(library="np")
        gh_p2 = shwr["p2"].array(library="np")
        gh_p3 = shwr["p3"].array(library="np")

        params.append(
            np.array(
                [
                    float(i + 1),
                    float(lg_10_e),
                    float(zenith_ang_deg),
                    float(azimuth_ang_deg),
                    float(gh_n_max),
                    float(gh_x_max),
                    float(gh_x0),
                    float(gh_p1),
                    float(gh_p2),
                    float(gh_p3),
                ]
            )
        )

    header = (
        "\t Shower Number \t"
        "\t lg_10(E) \t"
        "\t zenith(deg) \t"
        "\t azimuth(deg) \t"
        "\t GH Nmax \t"
        "\t GH Xmax \t"
        "\t GH X0 \t"
        "\t\t quad GH p1 \t"
        "\t quad GH p2 \t"
        "\t quad GH p3 "
    )

    save_data = params
    np.savetxt(output_file, X=save_data, header=header)


#%%
fname = "conex_eposlhc_100degzenith_000000001_100.root"
ntuple = uproot.open(fname)
shwr = ntuple["Shower;{}".format(1)]
lg_10_e = shwr["lgE"].array(library="np")
zenith_ang_deg = shwr["zenith"].array(library="np")
azimuth_ang_deg = shwr["azimuth"].array(library="np")

gh_n_max = shwr["Nmax"].array(library="np")
gh_x_max = shwr["Xmax"].array(library="np")
gh_x0 = shwr["X0"].array(library="np")
gh_p1 = shwr["p1"].array(library="np")
gh_p2 = shwr["p2"].array(library="np")
gh_p3 = shwr["p3"].array(library="np")

slt_depth = shwr["X"].array(library="np")
height_km = shwr["H"].array(library="np") / 1e3
height_first_interact_km = shwr["Hfirst"].array(library="np") / 1e3
electrons = shwr["N"].array(library="np")
gammas = shwr["Gamma"].array(library="np")
hadrons = shwr["Hadrons"].array(library="np")

shower = ShowerParameterization(
    table_decay_e=1,
    event_tag=1,
    decay_tag=1,
)


def modified_gh(x, n_max, x_max, x_0, p1, p2, p3):

    particles = (
        n_max
        * np.nan_to_num(
            ((x - x_0) / (x_max - x_0))
            ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x ** 2)))
        )
    ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x ** 2))))
    return particles


depths = np.linspace(0, slt_depth[0].max(), 1000)
shower_content = modified_gh(depths, gh_n_max, gh_x_max, gh_x0, gh_p1, gh_p2, gh_p3)

#%%


from nuspacesim.simulation.eas_composite.x_to_z_lookup import (
    depth_to_alt_lookup,
    depth_to_alt_lookup_v2,
)

# altitudes = depth_to_alt_lookup_v2(
#     slant_depths=np.round(slt_depth[0], 4),
#     angle=80,
#     starting_alt=0,
#     direction="up",
#     s=int(1e5),
# )


plt.figure(figsize=(4, 3), dpi=100)
plt.scatter(slt_depth[0], electrons[0], label="charged", s=1)
plt.scatter(slt_depth[0], gammas[0], label="gammas", s=1)
plt.scatter(slt_depth[0], hadrons[0], label="hadrons", s=1)
plt.plot(depths, shower_content, "--k", label="conex gh fit")


plt.xlabel(
    "slant depth (g/cm^2) \n {} \n altitude of first interaction {:.0f} km".format(
        fname, height_first_interact_km[0]
    )
)
plt.ylabel("N")
plt.yscale("log")
plt.ylim(bottom=1)
plt.xlim(left=slt_depth[0].min(), right=slt_depth[0].max())


alt = plt.twiny()
alt.set_xlim(left=height_km[0].min(), right=height_km[0].max())
# alt.plot(altitudes, electrons[0], color="red", label="calculated altitude")
alt.set_xlabel("height (km)")

#%%

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


def unpack():
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
    return showers


showers = unpack()

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

    gh_depths = np.linspace(corsika_depths.min(), corsika_depths.max(), 1000)
    corsika_n = modified_gh(gh_depths, nmax, xmax, x0, p1, p2, p3)

# fig, ax = plt.subplots(
#     nrows=1,
#     ncols=1,
#     sharex=True,
#     # gridspec_kw={"height_ratios": [6, 2, 2, 2, 2]},
#     # figsize=(6, 10),
#     dpi=300,
# )
plt.plot(gh_depths, corsika_n, "--k", label="Corsika GH Fit")
plt.plot(corsika_depths, corsika_charge, "--", label="Charged", alpha=1)
plt.plot(corsika_depths, corsika_electron, "--", label=r"e$^{-}$", alpha=1)
plt.plot(corsika_depths, corsika_positron, "--", label=r"e$^{+}$", alpha=1)
plt.plot(corsika_depths, corsika_muon, "--", label=r"$\mu^{-}$", alpha=1)
plt.plot(corsika_depths, corsika_hadron, "--", label=r"hadron", alpha=1)
