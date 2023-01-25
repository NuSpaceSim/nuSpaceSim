import uproot
import numpy as np


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
    r"""
    Reads ntuple contents such as the depth and particle components.

    Examples
    --------
    tup_folder = "./conex_7_50_runs"
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
        depths = shwr_data.get_depths()
        mus = shwr_data.get_muons()
        char = shwr_data.get_charged()
        el = shwr_data.get_elec_pos()
        had = shwr_data.get_hadrons()
        gam = shwr_data.get_gamma()
        mus_nmaxs, mus_xmaxs = bin_nmax_xmax(depths, mus)
        char_nmaxs, char_xmaxs = bin_nmax_xmax(depths, char)
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
    """

    def __init__(self, file_name: str, shower_header_name=101):
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
