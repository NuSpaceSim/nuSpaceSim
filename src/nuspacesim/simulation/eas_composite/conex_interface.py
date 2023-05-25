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

    Parameters
    ----------
    file_name: str
        name of the ntuple file
    shower_header_name: int
        default, 101. header name depends on the run. will spit out error to change
        it to a certain recognized key.

    Returns
    -------
    conex_read : obj
        can use various functions after initiating this object

    Examples
    --------

    """

    def __init__(self, file_name: str, shower_header_name=101):
        self.file_name = file_name
        self.ntuple = uproot.open(self.file_name)
        try:
            self.shwr = self.ntuple["Shower;{}".format(shower_header_name)]
        except:
            self.shwr = self.ntuple["Shower;{}".format(102)]
        print("> Reading", file_name.replace("\\", "/").split("/")[-1])
        # lg_10_e = shwr["lgE"].array(library="np")
        # zenith_ang_deg = shwr["zenith"].array(library="np")
        # azimuth_ang_deg = shwr["azimuth"].array(library="np")

    def modified_gh(self, x, n_max, x_max, x_0, p1, p2, p3):
        r"""Gaisser-Hillas Parametrization with a quadratic lambda.
        Qudratic coefficients p1, p2, p3, non-physical.


        Parameters
        ----------
        x: float
            the grammage bins g/cm^2
        n_max: float
            peak number of particles
        x_max: float
            grammage where the shower peaks
        x_0: float
            shower start, can be negative, fitting param, non-physical

        Returns
        -------
        N: array
            particle content


        """
        particles = (
            n_max
            * np.nan_to_num(
                ((x - x_0) / (x_max - x_0))
                ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x ** 2)))
            )
        ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x ** 2))))
        return particles

    def gh_fits(self):
        r"""
        loop through and get each the GH particle profiles by evaluating using the params
        """
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

    def get_gh_params(self):
        """
        get the gaisser hillas profiles
        """

        gh_n_max = self.shwr["Nmax"].array(library="np")
        gh_x_max = self.shwr["Xmax"].array(library="np")
        gh_x0 = self.shwr["X0"].array(library="np")
        gh_p1 = self.shwr["p1"].array(library="np")
        gh_p2 = self.shwr["p2"].array(library="np")
        gh_p3 = self.shwr["p3"].array(library="np")

        params = np.vstack((gh_n_max, gh_x_max, gh_x0, gh_p1, gh_p2, gh_p3)).T

        return params

    def get_dedx(self):
        """get dedx array; each row is for each shower"""
        de_dx = self.shwr["dEdX"].array(library="np")
        dx = np.zeros((int(de_dx.size), int(de_dx[0].size)))
        for i, depths in enumerate(de_dx):
            dx[i, :] = depths
        return dx

    def get_depths(self):
        """get depths array; each row is for each shower"""
        slt_depth = self.shwr["X"].array(library="np")
        x = np.zeros((int(slt_depth.size), int(slt_depth[0].size)))
        for i, depths in enumerate(slt_depth):
            x[i, :] = depths
        return x

    def get_alts(self):
        """get starting alts array; each number is for each shower"""
        height_km = self.shwr["H"].array(library="np") / 1e3
        z = np.zeros((int(height_km.size), int(height_km[0].size)))
        for i, height in enumerate(height_km):
            z[i, :] = height
        return z

    def get_charged(self):
        """get charged component array; each row is for each shower"""
        charged = self.shwr["N"].array(library="np")
        c = np.zeros((int(charged.size), int(charged[0].size)))
        for i, charged_particles in enumerate(charged):
            c[i, :] = charged_particles
        return c

    def get_elec(self):
        """get electron component array; each row is for each shower"""
        elect_pos = self.shwr["Electrons"].array(library="np")
        e = np.zeros((int(elect_pos.size), int(elect_pos[0].size)))
        for i, elec in enumerate(elect_pos):
            e[i, :] = elec
        return e

    def get_gamma(self):
        """get gamma component array; each row is for each shower"""
        gammas = self.shwr["Gamma"].array(library="np")
        g = np.zeros((int(gammas.size), int(gammas[0].size)))
        for i, gamm in enumerate(gammas):
            g[i, :] = gamm
        return g

    def get_hadrons(self):
        """get hadron component array; each row is for each shower"""
        hadrons = self.shwr["Hadrons"].array(library="np")
        h = np.zeros((int(hadrons.size), int(hadrons[0].size)))
        for i, had in enumerate(hadrons):
            h[i, :] = had
        return h

    def get_muons(self):
        """get muon component array; each row is for each shower"""
        muons = self.shwr["Mu"].array(library="np")
        m = np.zeros((int(muons.size), int(muons[0].size)))
        for i, mu in enumerate(muons):

            m[i, :] = mu
        return m
