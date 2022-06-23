"""
Demo for generating fluctuated showers by sampling the nmax variability distribution
using a rudimentary dartboard MC.
"""

import matplotlib.pyplot as plt
import numpy as np


# relative imports are not for scripts, absolute imports here
from nuspacesim.simulation.eas_composite.composite_eas import CompositeShowers
from nuspacesim.simulation.eas_composite.mc_mean_shwr import MCVariedMean

# initialize shower parameters
make_composites_00km = CompositeShowers(
    alt=0, shower_end=8000, grammage=1, tau_table_start=3000
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
# here were throwing darts at a distribution of particle values at xmax
# 400 throws at a distribution with 30 bins
# also, the sample mean will be returned with grammage resolution of 20 g/cm2
mc = MCVariedMean(
    slant_depths=stacked_x,
    composite_showers=stacked_y,
    n_throws=400,
    hist_bins=30,
    sample_grammage=20,
)
# throw darts using intialized parameters, return a scaling factor,
# the sampled grammage, and the samples shower mean
mc_rms, sample_grm, sample_shwr, variability_dist = mc.sampling_nmax_once(
    return_rms_dist=True
)

# # generate a bunch of showers
# fluctuated_showers = []
# grammages = []

# for i in range(1):

#     mc_rms, sample_grm, sample_shwr, variability_dist = mc.sampling_nmax_once(
#         return_rms_dist=True
#     )
#     fluctuated_showers.append(mc_rms * sample_shwr)
#     grammages.append(sample_grm)

# fluctuated_showers = np.array(fluctuated_showers)
# grammages = np.array(grammages)

# # save the fluctuated shower themselves
# header = "\t First line: grammage; Following lines are fluctuated showers\t"
# save_data = np.vstack((grammages[0], fluctuated_showers))
# np.savetxt(
#     "fluctuated_full_fluctuated_mean_1683_showers.txt", X=save_data, header=header
# )

# plot the sample_shwr (the mean of the input showers) and scale it accordingly
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(
    sample_grm,
    sample_shwr * mc_rms,
    color="violet",
    zorder=20,
    label="All shower mean",
)

plt.ylim(bottom=1e0, top=0.5e8)
plt.title("Sampling RMS Distribution At Mean XMax")
plt.xlabel("slant depth (g cm$^{-2}$)")
plt.ylabel("$N$")
plt.yscale("log")
plt.tick_params(axis="both", which="both", direction="in", top="on", right="on")
plt.grid(True, which="both", linestyle="--")
