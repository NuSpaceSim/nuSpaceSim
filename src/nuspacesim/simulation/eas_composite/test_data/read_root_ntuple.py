import uproot
import numpy as np
from nuspacesim.simulation.eas_composite.shower_long_profiles import (
    ShowerParameterization,
)
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
import matplotlib.pyplot as plt


def conexgh_to_text(file_path: str, output_file: str, num_showers=1):
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


class ReadConex:
    def __init__(self, file_name: str, shower_header_name=2):
        self.file_name = file_name
        self.ntuple = uproot.open(self.file_name)
        self.shwr = self.ntuple["Shower;{}".format(shower_header_name)]
        # lg_10_e = shwr["lgE"].array(library="np")
        # zenith_ang_deg = shwr["zenith"].array(library="np")
        # azimuth_ang_deg = shwr["azimuth"].array(library="np")

    def modified_gh(self, x, n_max, x_max, x_0, p1, p2, p3):

        particles = (
            n_max
            * np.nan_to_num(
                ((x - x_0) / (x_max - x_0))
                ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x ** 2)))
            )
        ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x ** 2))))
        return particles

    def gh_fits(self):
        slt_depths = self.shwr["X"].array(library="np")
        gh_n_max = self.shwr["Nmax"].array(library="np")
        gh_x_max = self.shwr["Xmax"].array(library="np")
        gh_x0 = self.shwr["X0"].array(library="np")
        gh_p1 = self.shwr["p1"].array(library="np")
        gh_p2 = self.shwr["p2"].array(library="np")
        gh_p3 = self.shwr["p3"].array(library="np")

        shower_fits = np.zeros((int(slt_depths.size), int(slt_depths[0].size)))
        for i, x in enumerate(slt_depths):
            gh = self.modified_gh(
                x=x,
                n_max=gh_n_max[i],
                x_max=gh_x_max[i],
                x_0=gh_x0[i],
                p1=gh_p1[i],
                p2=gh_p2[i],
                p3=gh_p3[i],
            )
            shower_fits[i, :] = gh
        return shower_fits

    def get_depths(self):
        slt_depth = self.shwr["X"].array(library="np")
        x = np.zeros((int(slt_depth.size), int(slt_depth[0].size)))
        for i, depths in enumerate(slt_depth):
            x[i, :] = depths
        return x

    def get_alts(self):
        height_km = self.shwr["H"].array(library="np") / 1e3
        z = np.zeros((int(height_km.size), int(height_km[0].size)))
        for i, height in enumerate(height_km):
            z[i, :] = height
        return z

    def get_charged(self):
        charged = self.shwr["N"].array(library="np")
        c = np.zeros((int(charged.size), int(charged[0].size)))
        for i, charged_particles in enumerate(charged):
            c[i, :] = charged_particles
        return c

    def get_elec_pos(self):
        elect_pos = self.shwr["Electrons"].array(library="np")
        e = np.zeros((int(elect_pos.size), int(elect_pos[0].size)))
        for i, elec in enumerate(elect_pos):
            e[i, :] = elec
        return e

    def get_gamma(self):
        gammas = self.shwr["Gamma"].array(library="np")
        g = np.zeros((int(gammas.size), int(gammas[0].size)))
        for i, gamm in enumerate(gammas):
            g[i, :] = gamm
        return g

    def get_hadrons(self):
        hadrons = self.shwr["Hadrons"].array(library="np")
        h = np.zeros((int(hadrons.size), int(hadrons[0].size)))
        for i, had in enumerate(hadrons):
            h[i, :] = had
        return h

    def get_muons(self):
        muons = self.shwr["Mu"].array(library="np")
        m = np.zeros((int(muons.size), int(muons[0].size)))
        for i, mu in enumerate(muons):

            m[i, :] = mu
        return m


# labels = ["1eV_5deg", "100PeV_5deg", "10EeV_5deg"]
# fnames = [
#     "./conex_7_50_runs/1eV_10shwrs_5degearthemergence_eposlhc_1456378716_100.root",
#     "./conex_7_50_runs/100PeV_10shwrs_5degearthemergence_eposlhc_1324240290_100.root",
#     "./conex_7_50_runs/10EeV_10shwrs_5degearthemergence_eposlhc_660227399_100.root",
# ]
colors = ["tab:red", "tab:blue", "tab:green"]
labels = ["1deg", "5deg", "35deg"]
fnames = [
    "./conex_7_50_runs/100PeV_10shwrs_1degearthemergence_eposlhc_1284567109_100.root",
    "./conex_7_50_runs/100PeV_10shwrs_5degearthemergence_eposlhc_1324240290_100.root",
    "./conex_7_50_runs/100PeV_10shwrs_35degearthemergence_eposlhc_1404866740_100.root",
]

plt.figure(figsize=(4, 3), dpi=300)
for label, fname, color in zip(labels, fnames, colors):
    conex = ReadConex(fname)
    fits = conex.gh_fits()
    depths = conex.get_depths()
    alts = conex.get_alts()
    mus = conex.get_muons()
    el = conex.get_elec_pos()
    charges = conex.get_charged()

    for i, x in enumerate(depths):
        if i == 1:
            plt.plot(x, charges[i], label=label, c=color, alpha=0.5, lw=1)

        else:
            plt.plot(x, charges[i], c=color, alpha=0.5, lw=1)

#!!! showers with different inclinations can't be on the same twin axis.
# plt.xlabel("slant depth (g/cm^2) \n {}  ".format(fname))
plt.xlabel("slant depth (g/cm^2) \n {}  ".format("charged"))
plt.ylabel("N")
plt.yscale("log")
plt.ylim(bottom=1)
plt.xlim(left=x.min(), right=x.max())
plt.legend(ncol=2)


alt = plt.twiny()
alt.set_xlim(left=alts.min(), right=alts.max())
alt.set_xlabel("height (km)")
#%%
colors = ["tab:red", "tab:blue", "tab:green"]
labels = ["1deg", "5deg", "35deg"]
fnames = [
    "./conex_7_50_runs/100PeV_10shwrs_1degearthemergence_eposlhc_1284567109_100.root",
    "./conex_7_50_runs/100PeV_10shwrs_5degearthemergence_eposlhc_1324240290_100.root",
    "./conex_7_50_runs/100PeV_10shwrs_35degearthemergence_eposlhc_1404866740_100.root",
]


def pwr_law(x, a, coeff):
    return coeff * x ** a


plt.figure(figsize=(4, 3), dpi=300)

# muons_x = []
# muons
# gammas_to_fit = []

for label, fname, color in zip(labels, fnames, colors):
    conex = ReadConex(fname)
    fits = conex.gh_fits()
    depths = conex.get_depths()
    alts = conex.get_alts()
    mus = conex.get_muons()
    charges = conex.get_charged()
    el = conex.get_elec_pos()
    had = conex.get_hadrons()
    gam = conex.get_gamma()
    # line plot
    # for i, x in enumerate(depths):
    #     if i == 1:
    #         plt.plot(x, charges[i], label=label, c=color, alpha=0.5, lw=1)

    #     else:
    #         plt.plot(x, charges[i], c=color, alpha=0.5, lw=1)

    particle_comps = [mus, el, had, gam]
    particle_type_labels = [" mouns", " electrons", " hadrons", " gammas"]
    mtype = ["^", "x", "s", "o"]
    for l, ptype in enumerate(particle_comps):
        bin_nmaxs, bin_xmaxs = bin_nmax_xmax(depths, ptype)
        plt.scatter(
            bin_xmaxs.mean(),
            bin_nmaxs.mean(),
            c=color,
            marker=mtype[l],
            label=label + particle_type_labels[l],
        )


plt.title("Taking the mean of 10 shower components for each point \n ")
# plt.yscale("log")
# plt.xscale("log")
# plt.xlim(100, 3000)
# plt.ylim(5e5, 8e9)
plt.legend(fontsize=5)
plt.xlabel("bin xmax")
plt.ylabel("bin nmax")
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

# from nuspacesim.simulation.eas_composite.x_to_z_lookup import (
#     depth_to_alt_lookup,
#     depth_to_alt_lookup_v2,
# )

# altitudes = depth_to_alt_lookup_v2(
#     slant_depths=np.round(slt_depth[0], 4),
#     angle=80,
#     starting_alt=0,
#     direction="up",
#     s=int(1e4),
# )
# #%%
# plt.figure(figsize=(4, 3), dpi=300)
# plt.plot(gh_depths, corsika_n, "--k", label="Corsika GH Fit")
# plt.plot(corsika_depths, corsika_charge, "--", label="Charged", alpha=1)
# plt.plot(corsika_depths, corsika_electron, "--", label=r"e$^{-}$", alpha=1)
# plt.plot(corsika_depths, corsika_positron, "--", label=r"e$^{+}$", alpha=1)
# plt.plot(corsika_depths, corsika_muon, "--", label=r"$\mu^{-}$", alpha=1)
# plt.plot(corsika_depths, corsika_hadron, "--", label=r"hadron", alpha=1)
# plt.yscale("log")

# alt_ax = plt.twiny()

# # alt.plot(altitudes, electrons[0], color="red", label="calculated altitude")
# alt_ax.set_xlim(left=altitudes.min(), right=altitudes.max())
# alt_ax.set_xlabel("altiude (km)")
