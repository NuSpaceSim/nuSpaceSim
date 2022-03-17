import matplotlib.pyplot as plt
import numpy as np
from composite_ea_showers import CompositeShowers
from fitting_composite_eas import FitCompositeShowers
from plt_routines import mean_rms_plot, decay_channel_mult_plt
from scipy import stats

#%%

make_composites_00km = CompositeShowers(
    alt=0, shower_end=5e3, grammage=1, tau_table_start=3000
)

comp_showers_00km, comp_depths_00km = make_composites_00km(filter_errors=False)

trimmed_showers_00km, test_depths = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km,
    composite_depths=comp_depths_00km,
    separate_showers=False,
)

decay_channels = np.unique(comp_depths_00km[:, 1])
#%%

# sample_shower_column = trimmed_showers_00km[:,500::500].T
# sample_depth_column = comp_depths_00km[:,500::500].T

plt.figure(figsize=(8, 6), dpi=200)
bin_00km, mean_00km, rms_low, rms_high = mean_rms_plot(
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
    max_shwr_col,
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

#%% Dart board monte carlo
n = 10
bins = 30


def mc_drt_rms(n, bins, col_depths, col_showers, plot_darts=False):
    """
    Given the particle content at a given slant depth for a set of composite showers,
    Returns the mean and associated MC uncertainty from at that point, sampled from RMS curve.
    """

    # get normalized using mean
    col_mean = np.nanmean(col_showers)
    col_showers = col_showers / col_mean
    col_depth = col_depths[0]  # assuming depths are stacked properly

    freq, bin_edgs = np.histogram(col_showers, bins=bins)
    bin_ctr = 0.5 * (bin_edgs[1:] + bin_edgs[:-1])
    bin_size = bin_ctr[1] - bin_ctr[0]

    # get random values within the plotting window
    rdom_x_ax = np.random.uniform(
        low=0.0, high=(np.max(bin_ctr) + bin_size), size=n
    )
    rdom_y_ax = np.random.uniform(low=0.0, high=(np.max(freq) + 2), size=n)

    x_residuals = np.abs(rdom_x_ax - bin_ctr[:, np.newaxis])
    clst_x_idx = np.argmin(x_residuals, axis=0)  # x_residuals < bin_size
    smlst_resid = np.take_along_axis(
        x_residuals.T, clst_x_idx[:, None], axis=1
    )
    within_bin_msk = smlst_resid < bin_size

    test = clst_x_idx[:, None][within_bin_msk]
    # print(test)
    accepted_bin_idxs = np.unique(clst_x_idx[:, None][within_bin_msk])
    hit_y_axis_values = freq[accepted_bin_idxs]
    # print(accepted_bin_idxs,hit_y_axis_values )

    # brute force loop, not sure how to vectorize yet
    accepted_his_idxs = []
    accepted_his_vals = []
    for y_dart, bin_idx in zip(rdom_y_ax, test):

        if y_dart - freq[bin_idx] < 0:
            accepted_his_idxs.append(bin_idx)
            # print(freq[bin_idx])
            accepted_his_vals.append(freq[bin_idx])

    # accepted_his_idxs = np.unique(accepted_his_idxs)
    # accepted_his_vals = np.unique(accepted_his_vals)
    # filtered_freq = freq[accepted_his_idxs]
    # print(accepted_his_idxs, filtered_freq)

    new_frequencies = np.zeros(np.size(freq))
    # print(accepted_his_idxs)
    new_frequencies[accepted_his_idxs[0]] = accepted_his_vals[0]
    hit_rms = bin_ctr[accepted_his_idxs]

    if plot_darts is True:
        plt.figure(figsize=(8, 6), dpi=200)
        plt.hist(
            col_showers,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
            label=r"${:g} g/cm^2$".format(col_depths[0]),
            bins=bins,
        )
        plt.scatter(bin_ctr, freq, s=2, c="k")
        plt.scatter(rdom_x_ax, rdom_y_ax, s=2, c="r")
        # plt.xlim(right=8e7)
        # plt.ylim(top=np.max(freq)+2)
        plt.xlabel("Particle Content/ Avg particle Content (N)")
        plt.title("bins = {}, n = {}".format(bins, n))
        plt.legend()
        plt.figure(figsize=(8, 6), dpi=200)
        plt.bar(
            bin_ctr,
            new_frequencies,
            bin_size,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
            label=r"${:g} g/cm^2$ MC Reconstructed".format(col_depth),
        )
        plt.scatter(rdom_x_ax, rdom_y_ax, s=2, c="r")

        plt.title("bins = {}, n = {}".format(bins, n))
        plt.xlabel("Particle Content/ Avg particle Content (N)")
        # plt.xlim(right=8e7)
        # plt.ylim(top=np.max(freq)+2)
        plt.legend()

    return col_depth, col_mean, hit_rms[0]


#%%
col_depth, col_mean, hit_rms = mc_drt_rms(
    50, 100, col_depths=max_dpth_col, col_showers=max_shwr_col
)

#%% Iterating through the composite shower to assign rms for each depth

sample_shower_column = trimmed_showers_00km[::, 500::1].T
sample_depth_column = comp_depths_00km[::, 500::1].T

shwr_depth = np.ones(np.shape(sample_shower_column)[0])
shwr_mean = np.ones(np.shape(sample_shower_column)[0])
sample_shwr_rms = np.ones(np.shape(sample_shower_column)[0])

for i, (depths, showers) in enumerate(
    zip(sample_depth_column, sample_shower_column)
):

    showers = showers[~np.isnan(showers)]

    col_depth, col_mean, hit_rms = mc_drt_rms(
        200, 30, col_depths=depths, col_showers=showers
    )
    shwr_depth[i] = col_depth
    shwr_mean[i] = col_mean
    sample_shwr_rms[i] = hit_rms


#%%

plt.plot(shwr_depth, shwr_mean)
rms_err_upper = shwr_mean + sample_shwr_rms * shwr_mean
rms_err_lower = shwr_mean - sample_shwr_rms * shwr_mean
abs_error = rms_err_upper - shwr_mean
plt.figure(figsize=(8, 6), dpi=200)
plt.errorbar(shwr_depth, shwr_mean, abs_error, fmt=".")
plt.ylabel("Shower Mean * Sampled Shower Normalization  Factor")
# plt.yscale('log')
#%% Sampling max over and over again


max_sample_shwr_rms = np.ones(np.shape(sample_shower_column)[0])

for i, (depths, showers) in enumerate(
    zip(sample_depth_column, sample_shower_column)
):

    _, _, hit_rms = mc_drt_rms(
        200, 30, col_depths=depths, col_showers=max_shwr_col
    )

    max_sample_shwr_rms[i] = hit_rms


plt.plot(shwr_depth, shwr_mean)
rms_err_upper = shwr_mean + max_sample_shwr_rms * shwr_mean
rms_err_lower = shwr_mean - max_sample_shwr_rms * shwr_mean
around_max_abs_error = rms_err_upper - shwr_mean
plt.figure(figsize=(8, 6), dpi=200)
plt.errorbar(shwr_depth, shwr_mean, around_max_abs_error, fmt=".")
plt.ylabel("Shower Mean * Sampled Shower Normalization  Factor")
# plt.yscale('log')

#%%

plt.figure(figsize=(8, 6), dpi=200)
plt.scatter(
    shwr_depth,
    shwr_mean * max_sample_shwr_rms,
    label="Sampled Shower/ShowerMean",
    s=5,
)   
plt.xlabel("Slant Depth")
plt.ylabel("Shower Mean * Sampled Shower Normalization  Factor")
# plt.ylabel('Particle Content/ Avg particle Content (N)')
plt.legend()

#%% Decay channel

decay_channel_mult_plt(bins=comp_depths_00km, showers=trimmed_showers_00km)
