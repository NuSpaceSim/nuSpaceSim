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
fname = "conex_eposlhc_000000002_100.root"
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
electrons = shwr["N"].array(library="np")
gammas = shwr["Gamma"].array(library="np")
hadrons = shwr["Hadrons"].array(library="np")

shower = ShowerParameterization(
    table_decay_e=1,
    event_tag=1,
    decay_tag=1,
)

depth, shower_content = shower.gaisser_hillas(
    n_max=gh_n_max,
    x_max=gh_x_max,
    x_0=gh_x0,
    p1=gh_p1,
    p2=gh_p1,
    p3=gh_p1,
    shower_end=np.max(slt_depth[0]),
    grammage=1,
)
#%%
plt.figure(figsize=(4, 3), dpi=100)
plt.plot(height_km[0], electrons[0], label="charged")
plt.plot(height_km[0], gammas[0], label="gammas")
plt.plot(height_km[0], hadrons[0], label="hadrons")
# plt.plot(depth, shower_content, "--k", label="conex gh fit")
plt.yscale("log")
plt.title(fname)
plt.legend()
