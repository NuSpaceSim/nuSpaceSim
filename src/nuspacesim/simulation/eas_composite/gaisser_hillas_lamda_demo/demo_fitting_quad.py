"""
Demo for fitting fluctuated shower by using corsika's lambda parameter.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# relative imports are not for scripts, absolute imports here
from nuspacesim.simulation.eas_composite.composite_eas import CompositeShowers
from nuspacesim.simulation.eas_composite.comp_eas_utils import (
    bin_nmax_xmax,
    decay_channel_filter,
    separate_showers,
)
from nuspacesim.simulation.eas_composite.mc_mean_shwr import MCVariedMean
from nuspacesim.simulation.eas_composite.plt_routines import (
    mean_rms_plt,
    decay_channel_mult_plt,
    recursive_plt,
    get_decay_channel,
)

# initialize shower parameters
make_composites_00km = CompositeShowers(
    alt=0, shower_end=1e4, grammage=1, tau_table_start=3000
)

# call to make the composite showers and return particles as a function of depth
comp_showers_00km, comp_depths_00km = make_composites_00km(filter_errors=False)

# trim ends and seperate out by shower type: reached the threshold or not
full, trimmed, shallow = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km,
    composite_depths=comp_depths_00km,
    separate_showers=True,
)

# TODO: make grammage and slant depth x,y for all-- in same order

# each particle type is returned in a tuple, unpack them
full_x = full[1]
full_y = full[0]

trimmed_x = trimmed[1]
trimmed_y = trimmed[0]

# stack them for convenience, not differentiating between cut off or not.
stacked_x = np.vstack((full_x, trimmed_x))
stacked_y = np.vstack((full_y, trimmed_y))


# take a set of showers and fluctate them by sampling variability
# here were throwing darts at a distribution
# 400 throws at a distribution with 30 bins
# also, the sample mean will be returned with grammage resolution of 20 g/cm2
mc = MCVariedMean(
    slant_depths=full_x,  # !!! only full showers
    composite_showers=full_y,
    n_throws=400,
    hist_bins=30,
    sample_grammage=1,
)

# throw darts using intialized parameters, return a scaling factor,
# the sampled grammage, and the samples shower mean

mc_rms, sample_grm, sample_shwr, variability_dist = mc.sampling_nmax_once(
    return_rms_dist=True
)


# def polynomial(x, a, b, c, d, e, f):
#     y = a + b * x ** 1 + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5
#     return y
# a = np.polyfit(sample_grm, shower, 5)
# plt.plot(sample_grm, polynomial(sample_grm, *a))

# plot the sample_shwr (the mean of the input showers) and scale it accordingly
plt.figure(figsize=(8, 6), dpi=200)
plt.scatter(
    sample_grm,
    sample_shwr * mc_rms,
    color="violet",
    zorder=20,
    label="All shower mean",
)

plt.ylim(bottom=1e0)
plt.title("Sampling RMS Distribution At Mean XMax")
plt.xlabel("slant depth (g cm$^{-2}$)")
plt.ylabel("$N$")
plt.yscale("log")
plt.tick_params(axis="both", which="both", direction="in", top="on", right="on")
plt.grid(True, which="both", linestyle="--")


#%% fit them
# TODO: do this for the rebounding/ non rebounding bifurcation.
from scipy import optimize


def modified_gh(x, n_max, x_max, x_0, p1, p2, p3):

    particles = (
        n_max
        * np.nan_to_num(
            ((x - x_0) / (x_max - x_0))
            ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x ** 2)))
        )
    ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x ** 2))))

    return particles


def gaisser_hillas(x, n_max, x_max, x_0, gh_lambda):

    particles = (
        n_max
        * np.nan_to_num(((x - x_0) / (x_max - x_0)) ** ((x_max - x_0) / gh_lambda))
    ) * (np.exp((x_max - x) / gh_lambda))

    return particles


parameters = []


mask = (sample_grm <= 4000) & (sample_shwr != np.nan)  # mask to fudge fits
grammage_to_fit = sample_grm[mask]
sample_shwr_to_fit = sample_shwr[mask]
num_fit_params = 4

guess_n_max, guess_x_max = bin_nmax_xmax(sample_grm, sample_shwr)
fit_params, covariance = optimize.curve_fit(
    f=modified_gh,
    xdata=grammage_to_fit,
    ydata=sample_shwr_to_fit,
    p0=[guess_n_max, guess_x_max, 0, 10, -0.01, 1e-05],
    bounds=(
        [0, 0, -np.inf, -np.inf, -np.inf, -np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    ),
)
print("Best fit n_max, x_max, x_0, p1, p2, p3")
print(*fit_params)
theory_n = modified_gh(sample_grm[mask], *fit_params)

chisquare = np.sum((sample_shwr_to_fit - theory_n) ** 2 / theory_n)
dof = np.size(theory_n) - num_fit_params
reduced_chisquare = chisquare / dof
p_value = stats.chi2.sf(chisquare, dof)

parameters.append(fit_params)
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(
    grammage_to_fit,
    theory_n,
    label=r"Fit: $\chi_{{\nu}}^2$ = {:.2e}, $P(\chi^2, \nu)$ = {:.2f}".format(
        reduced_chisquare, p_value
    ),
)
plt.plot(
    sample_grm,
    sample_shwr,
    color="violet",
    zorder=20,
    linewidth=2,
    # label="Uncut sample_shwrs, Not Reaching 1%",
    label="All sample_shwrs",
)
plt.ylim(bottom=1e0)
plt.title("Showers Fitted Up to 8000 ")
plt.xlabel("slant depth (g cm$^{-2}$)")
plt.ylabel("$N$")
plt.yscale("log")
plt.tick_params(axis="both", which="both", direction="in", top="on", right="on")
plt.grid(True, which="both", linestyle="--")
plt.legend()


#%% save the best fit parameters
# header = (
#     "\t Shower Number \t"
#     "\t lg_10(E) \t"
#     "\t zenith(deg) \t"
#     "\t azimuth(deg) \t"
#     "\t GH Nmax \t"
#     "\t GH Xmax \t"
#     "\t GH X0 \t"
#     "\t quad GH p1 \t"
#     "\t quad GH p2 \t"
#     "\t quad GH p3 "
# )

# header = (
#     "\t Shower Number \t"
#     "\t lg_10(E) \t"
#     "\t zenith(deg) \t"
#     "\t azimuth(deg) \t"
#     "\t GH Nmax \t"
#     "\t GH Xmax \t"
#     "\t GH X0 \t"
#     "\t GH Lambda \t"
# )

# number_fluc_shwrs = np.shape(np.array(parameters))[0]
# extra_info = np.vstack(
#     (
#         np.arange(number_fluc_shwrs),
#         17 * np.ones(number_fluc_shwrs),
#         95 * np.ones(number_fluc_shwrs),
#         np.zeros(number_fluc_shwrs),
#     )
# ).T
# save_data = np.hstack((extra_info, np.array(parameters)))
# np.savetxt("fluctuated_10_shwrs_0_km_gh.txt", X=save_data, header=header)

#%% save the fluctuated shower themselves
# header = "\t First line: grammage; Following lines are fluctuated showers\t"
# save_data = np.vstack((sample_grm, fluctuated_showers))
# np.savetxt(
#     "fluctuated_full_fluctuated_mean_1683_showers.txt", X=save_data, header=header
# )
