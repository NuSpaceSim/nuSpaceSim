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


# tup_folder = "/home/fabg/g_drive/Research/NASA/Work/conex2r7_50-runs/"
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
def gaisser_hillas(x, n_max, x_max, x_0, gh_lambda):

    particles = (
        n_max
        * np.nan_to_num(((x - x_0) / (x_max - x_0)) ** ((x_max - x_0) / gh_lambda))
    ) * (np.exp((x_max - x) / gh_lambda))

    return particles


sample_x = depths[0, :]
sample_y = comp_charged[0, 2:]
nmax, xmax = bin_nmax_xmax(bins=sample_x, particle_content=sample_y)

fit_params, covariance = optimize.curve_fit(
    f=gaisser_hillas,
    xdata=sample_x,
    ydata=sample_y,
    p0=[nmax, xmax, 0, 70],
    # bounds=([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
)

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(4, 3))
ax.plot(sample_x, sample_y)
ax.plot(sample_x, gaisser_hillas(sample_x, *fit_params))
ax.set(yscale="log", ylim=(1, 1e8), xlim=(0, 2000))
#%%


start = 3000
order = 15
mask = sample_x > start
z = np.polyfit(sample_x[mask], sample_y[mask], 12)
p = np.poly1d(z)

# power law


def pwr_law(x, a, b):
    return a * x**b


pfit, pcov = optimize.curve_fit(f=pwr_law, xdata=sample_x[mask], ydata=sample_y[mask])

# bounds=([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),


gh_theory = gaisser_hillas(sample_x, *fit_params)

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(5, 4))


ax.plot(sample_x, sample_y, lw=3, label="composite")
ax.plot(sample_x[mask], p(sample_x)[mask], label="{} order poly".format(order))
ax.plot(sample_x[~mask], gh_theory[~mask], label="GH")
ax.plot(
    sample_x[mask],
    pwr_law(sample_x[mask], *pfit),
    label=r"$\alpha$ = {:.2f}".format(pfit[1]),
)
ax.set(yscale="log", ylim=(1, 1e8))

inset = ax.inset_axes([0.2, 0.1, 0.3, 0.3])
inset.plot(sample_x, sample_y, lw=3)
inset.plot(sample_x[mask], p(sample_x)[mask])
inset.plot(sample_x[~mask], gh_theory[~mask])
inset.plot(sample_x[mask], pwr_law(sample_x[mask], *pfit))
inset.set(xlim=(1300, 2000), ylim=(1e5, 1e7), yscale="log")

ax.indicate_inset_zoom(inset)

inset = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
inset.plot(sample_x, sample_y, lw=3)
inset.plot(sample_x[mask], p(sample_x)[mask])
inset.plot(sample_x[~mask], gh_theory[~mask])
inset.plot(sample_x[mask], pwr_law(sample_x[mask], *pfit))
inset.set(xlim=(10200, 11000), ylim=(4e3, 1e5), yscale="log")

ax.indicate_inset_zoom(inset)

ax.legend()
