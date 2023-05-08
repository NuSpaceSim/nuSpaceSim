#%%
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_conex import ConexCompositeShowers
import os


def modified_gh(x, n_max, x_max, x_0, p1, p2, p3):

    particles = (
        n_max
        * np.nan_to_num(
            ((x - x_0) / (x_max - x_0))
            ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x**2)))
        )
    ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x**2))))

    return particles


def fit_quad_lambda(depth, comp_shower):
    r"""
    Gets fits for composite shower if supplied particle content and matching slant depths.
    Allows negative X0 and quadratic lambda.
    """

    nmax, xmax = bin_nmax_xmax(bins=depth, particle_content=comp_shower)
    fit_params, covariance = optimize.curve_fit(
        f=modified_gh,
        xdata=depth,
        ydata=comp_shower,
        p0=[nmax, xmax, 0, 80, -0.01, 1e-05],
        bounds=(
            [0, 0, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
    )
    theory_n = modified_gh(depth, *fit_params)
    print(fit_params)
    return theory_n


tup_folder = "/home/fabg/g_drive/Research/NASA/Work/conex2r7_50-runs/"
tup_folder = "C:/Users/144/Desktop/g_drive/Research/NASA/Work/conex2r7_50-runs"
# we can read in the showers with different primaries
elec_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_2033993834_11.root",
    )
)
pion_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_730702871_211.root",
    )
)
gamma_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_1722203790_22.root",
    )
)
# we can get the charged compoenent
elec_charged = elec_init.get_charged()
gamma_charged = gamma_init.get_charged()
pion_charged = pion_init.get_charged()
depths = elec_init.get_depths()

pids = [11, 22, 211]
init = [elec_charged, gamma_charged, pion_charged]
gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids)
comp_charged = gen_comp()
#%%
