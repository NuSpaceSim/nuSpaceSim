import matplotlib.pyplot as plt
import numpy as np

# relative imports are not for scripts, absolute imports here
from nuspacesim.simulation.eas_composite.composite_eas import CompositeShowers
from nuspacesim.simulation.eas_composite.mc_mean_shwr_sampler import MCVariedMean
from nuspacesim.simulation.eas_composite.plt_routines import (
    mean_rms_plt,
    decay_channel_mult_plt,
    recursive_plt,
)

#%%

make_composites_00km = CompositeShowers(
    alt=0, shower_end=8e3, grammage=10, tau_table_start=3000
)

comp_showers_00km, comp_depths_00km = make_composites_00km(filter_errors=False)

trimmed_showers_00km, _ = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km,
    composite_depths=comp_depths_00km,
    separate_showers=False,
)

full, trimmed, _ = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km,
    composite_depths=comp_depths_00km,
    separate_showers=True,
)

decay_channels = np.unique(comp_depths_00km[:, 1])
#%%
plt.figure(figsize=(8, 6), dpi=200)


for depths, showers in zip(trimmed[1], trimmed[0]):

    event_num = depths[0]
    decay_code = depths[1]

    plt.plot(
        depths[2:],
        showers[2:],
        alpha=0.2,
        # s=.2,
        label=str(event_num) + "|" + str(decay_code),
    )

plt.yscale("log")

#%%Sampling Per Slant Depth
sampler = MCVariedMean(
    trimmed_showers_00km,
    comp_depths_00km,
    n_throws=400,
    hist_bins=30,
    sample_grammage=100,
)
mc_rms, sample_grm, sample_shwr = sampler.sampling_per_depth()

rms_err_upper = sample_shwr + mc_rms * sample_shwr
rms_err_lower = sample_shwr - mc_rms * sample_shwr
abs_error = rms_err_upper - sample_shwr

plt.figure(figsize=(20, 6), dpi=200)
plt.suptitle("Sampling Per Slant Depth")

plt.subplot(1, 3, 1)
plt.errorbar(sample_grm, sample_shwr, abs_error, fmt=".")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content")
plt.yscale("log")

plt.subplot(1, 3, 2)
plt.errorbar(sample_grm, sample_shwr * mc_rms, fmt=".")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content*Random Multiplier")
plt.yscale("log")

plt.subplot(1, 3, 3)
plt.errorbar(sample_grm, sample_shwr * mc_rms, fmt=".")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content*Random Multiplier")

#%%Sampling Nmax Per Slant Depth

sampler = MCVariedMean(
    trimmed_showers_00km,
    comp_depths_00km,
    n_throws=400,
    hist_bins=30,
    sample_grammage=100,
)
mc_rms, sample_grm, sample_shwr = sampler.sampling_nmax_per_depth()

rms_err_upper = sample_shwr + mc_rms * sample_shwr
rms_err_lower = sample_shwr - mc_rms * sample_shwr
abs_error = rms_err_upper - sample_shwr

plt.figure(figsize=(20, 6), dpi=200)
plt.suptitle("Sampling Nmax Per Slant Depth")

plt.subplot(1, 3, 1)
plt.errorbar(sample_grm, sample_shwr, abs_error, fmt=".", c="orange")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content")
plt.yscale("log")

plt.subplot(1, 3, 2)
plt.errorbar(sample_grm, sample_shwr * mc_rms, fmt=".", c="orange")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content*Random Multiplier")
plt.yscale("log")

plt.subplot(1, 3, 3)
plt.errorbar(sample_grm, sample_shwr * mc_rms, fmt=".", c="orange")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content*Random Multiplier")
#%%Sampling Nmax Once
sampler = MCVariedMean(
    trimmed_showers_00km,
    comp_depths_00km,
    n_throws=400,
    hist_bins=30,
    sample_grammage=100,
)
mc_rms, sample_grm, sample_shwr = sampler.sampling_nmax_once()

rms_err_upper = sample_shwr + mc_rms * sample_shwr
rms_err_lower = sample_shwr - mc_rms * sample_shwr
abs_error = rms_err_upper - sample_shwr

plt.figure(figsize=(20, 6), dpi=200)
plt.suptitle("Sampling Nmax Once")

plt.subplot(1, 3, 1)
plt.errorbar(sample_grm, sample_shwr, abs_error, fmt=".", c="red")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content")
plt.yscale("log")

plt.subplot(1, 3, 2)
plt.errorbar(sample_grm, sample_shwr * mc_rms, fmt=".", c="red")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content*Random Multiplier")
plt.yscale("log")

plt.subplot(1, 3, 3)
plt.errorbar(sample_grm, sample_shwr * mc_rms, fmt=".", c="red")
recursive_plt(comp_depths_00km, comp_showers_00km)
plt.ylabel("Particle Content*Random Multiplier")
#%% overplot
plt.figure(figsize=(8, 6), dpi=200)


recursive_plt(comp_depths_00km, comp_showers_00km)

plt.errorbar(
    sample_grm, sample_shwr, abs_error, fmt=".", c="black", label="mean and sampled rms"
)

# plt.fill_between(
#     sample_grm,
#     rms_err_lower,
#     rms_err_upper,
#     alpha=0.4,
#     facecolor="black",
#     zorder=20,
# )

plt.title("Sampling Nmax Once")
plt.xlabel("slant depth g cm$^{-2}$")
plt.ylabel("$N$")
plt.yscale("log")
plt.legend()
#%%

sample_shower_column = trimmed_showers_00km[:, 500::500].T
sample_depth_column = comp_depths_00km[:, 500::500].T

plt.figure(figsize=(8, 6), dpi=200)

bin_00km, mean_00km, rms_low, rms_high = mean_rms_plt(
    showers=trimmed_showers_00km,
    bins=comp_depths_00km,
    label="0 km",
    facecolor="tab:red",
)

max_shwr_col = trimmed_showers_00km[:, np.argmax(mean_00km)].T
max_dpth_col = comp_depths_00km[:, np.argmax(mean_00km)].T

# mean and rms plot params
# plt.xlim(right=2000)
plt.yscale("linear")
# plt.axvline(max_dpth_col[1136])


plt.figure(figsize=(8, 6), dpi=200)
# x = showers / np.nanmean(showers)
plt.hist(
    max_shwr_col / np.mean(max_shwr_col),
    alpha=0.5,
    edgecolor="black",
    linewidth=0.5,
    label="{:g} g/cm^2".format(max_dpth_col[0]),
    bins=30,
)


# plt.scatter(bin_ctr, freq, c='k')
plt.title("Distribution of Composite values")
plt.xlabel("Particle Content/ Avg particle Content (N)")
# plt.xlabel('Particle Content (N)')
plt.ylabel("# of composite showers")
# plt.xscale('log')
plt.legend()
#%%sample_grm, sample_shwr, abs_error
decay_channel_mult_plt(
    bins=comp_depths_00km,
    showers=comp_showers_00km,
    smpl_rms_plt=True,
    sampl_grm=sample_grm,
    sampl_lwr_err=rms_err_lower,
    sampl_upr_err=rms_err_upper,
)


#%% Dart board monte carlo
# n = 10
# bins = 30


# #%%
# col_depth, col_mean, hit_rms = mc_drt_rms(
#     50, 100, col_depths=max_dpth_col, col_showers=max_shwr_col
# )

# #%% Iterating through the composite shower to assign rms for each depth

# sample_shower_column = trimmed_showers_00km[::, 500::1].T
# sample_depth_column = comp_depths_00km[::, 500::1].T

# shwr_depth = np.ones(np.shape(sample_shower_column)[0])
# shwr_mean = np.ones(np.shape(sample_shower_column)[0])
# sample_shwr_rms = np.ones(np.shape(sample_shower_column)[0])

# for i, (depths, showers) in enumerate(
#     zip(sample_depth_column, sample_shower_column)
# ):

#     showers = showers[~np.isnan(showers)]

#     col_depth, col_mean, hit_rms = mc_drt_rms(
#         200, 30, col_depths=depths, col_showers=showers
#     )
#     shwr_depth[i] = col_depth
#     shwr_mean[i] = col_mean
#     sample_shwr_rms[i] = hit_rms


# #%%

# plt.plot(shwr_depth, shwr_mean)
# rms_err_upper = shwr_mean + sample_shwr_rms * shwr_mean
# rms_err_lower = shwr_mean - sample_shwr_rms * shwr_mean
# abs_error = rms_err_upper - shwr_mean
# plt.figure(figsize=(8, 6), dpi=200)
# plt.errorbar(shwr_depth, shwr_mean, abs_error, fmt=".")
# plt.ylabel("Shower Mean * Sampled Shower Normalization  Factor")
# # plt.yscale('log')
# #%% Sampling max over and over again


# max_sample_shwr_rms = np.ones(np.shape(sample_shower_column)[0])

# for i, (depths, showers) in enumerate(
#     zip(sample_depth_column, sample_shower_column)
# ):

#     _, _, hit_rms = mc_drt_rms(
#         200, 30, col_depths=depths, col_showers=max_shwr_col
#     )

#     max_sample_shwr_rms[i] = hit_rms


# plt.plot(shwr_depth, shwr_mean)
# rms_err_upper = shwr_mean + max_sample_shwr_rms * shwr_mean
# rms_err_lower = shwr_mean - max_sample_shwr_rms * shwr_mean
# around_max_abs_error = rms_err_upper - shwr_mean
# plt.figure(figsize=(8, 6), dpi=200)
# plt.errorbar(shwr_depth, shwr_mean, around_max_abs_error, fmt=".")
# plt.ylabel("Shower Mean * Sampled Shower Normalization  Factor")
# # plt.yscale('log')

# #%%

# plt.figure(figsize=(8, 6), dpi=200)
# plt.scatter(
#     shwr_depth,
#     shwr_mean * max_sample_shwr_rms,
#     label="Sampled Shower/ShowerMean",
#     s=5,
# )
# plt.xlabel("Slant Depth")
# plt.ylabel("Shower Mean * Sampled Shower Normalization  Factor")
# # plt.ylabel('Particle Content/ Avg particle Content (N)')
# plt.legend()

# #%% Decay channel

#
