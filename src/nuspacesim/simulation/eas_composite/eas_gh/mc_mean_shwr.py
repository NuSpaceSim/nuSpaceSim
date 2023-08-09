import numpy as np
import matplotlib.pyplot as plt
from nuspacesim.simulation.eas_composite.plt_routines import mean_rms_plt
import warnings


class MCVariedMean:
    r"""Dart Board Monte Carlo Method"""

    def __init__(
        self,
        composite_showers,
        slant_depths,
        n_throws=100,
        hist_bins=30,
        sample_grammage=1,
    ):
        print(np.shape(composite_showers)[0], "Showers have been inputed.")
        # catch warnings when taking mean of a vertical NaN slice
        # e.g., for trimmed showers padded with NaNs, and taking the mean for a
        # given slant depth where all have been cut

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            self.tags = composite_showers[:, 0:2]
            self.showers = composite_showers
            self.depths = slant_depths
            self.n_throws = n_throws
            self.hist_bins = hist_bins
            self.n_showers = np.shape(composite_showers)[0]
            # find mean and depth of all composite showers
            self.mean_depth, self.mean, _, _ = mean_rms_plt(
                showers=self.showers,
                bins=self.depths,
            )

            # find where the average peaks, get that column for all showers @ that grammg.
            max_idx = np.nanargmax(self.mean)
            self.nmax_shwr_col = self.showers[:, max_idx].T
            self.xmax_dpth_col = self.depths[:, max_idx].T

            # print(max_idx)
            # print("X-max", self.xmax_dpth_col[1])
            # controll how much linear sampling is done
            left_pad_width = 400
            self.sample_dpth_col = self.depths[::, left_pad_width::sample_grammage]
            self.sample_shwr_col = self.showers[::, left_pad_width::sample_grammage]

            self.output_depth, self.output_mean, _, _ = mean_rms_plt(
                showers=np.hstack((self.tags, self.sample_shwr_col)),
                bins=np.hstack((self.tags, self.sample_dpth_col)),
            )

    # print(self.sample_dpth_col.shape[1])

    def mc_drt_rms(self, col_depths, col_showers, plot_darts=False):
        """
        Given the particle content at a given slant depth for a set of composite showers,
        Returns the mean and associated MC uncertainty from that point, sampled from
        a histogram showing the variance.
        """
        with warnings.catch_warnings():
            # take care of taking mean of NaN slice
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_mean = np.nanmean(col_showers)  # get normalized using mean

        col_showers = col_showers / col_mean
        col_depth = col_depths[0]  # assuming depths are stacked properly

        freq, bin_edgs = np.histogram(col_showers, bins=self.hist_bins)
        bin_ctr = 0.5 * (bin_edgs[1:] + bin_edgs[:-1])
        bin_size = bin_ctr[1] - bin_ctr[0]

        # get random values within the plotting window
        rdom_x_ax = np.random.uniform(
            low=0.0, high=(np.max(bin_ctr) + bin_size), size=self.n_throws
        )
        rdom_y_ax = np.random.uniform(
            low=0.0, high=(np.max(freq) + 2), size=self.n_throws
        )

        x_residuals = np.abs(rdom_x_ax - bin_ctr[:, np.newaxis])
        clst_x_idx = np.argmin(x_residuals, axis=0)  # x_residuals < bin_size
        smlst_resid = np.take_along_axis(x_residuals.T, clst_x_idx[:, None], axis=1)
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
        try:
            new_frequencies = np.zeros(np.size(freq))
            # print(accepted_his_idxs)
            new_frequencies[accepted_his_idxs[0]] = accepted_his_vals[0]
            hit_rms = bin_ctr[accepted_his_idxs]
        except RuntimeError:
            print("Need to throw more darts.")

        if plot_darts is True:
            plt.figure(figsize=(8, 6), dpi=200)
            plt.scatter(rdom_x_ax, rdom_y_ax, s=2, c="k", label="Throws")
            plt.hist(
                col_showers,
                alpha=0.5,
                edgecolor="black",
                linewidth=0.5,
                label=r"{:g} g/cm$^2$ mean Xmax".format(col_depths[0]),
                bins=self.hist_bins,
            )
            # plt.scatter(bin_ctr, freq, s=2, c="k")

            # plt.xlim(right=8e7)
            # plt.ylim(top=np.max(freq)+2)
            plt.xlabel("Nmax Particle Content/Avg Nmax for All Showers")
            plt.ylabel("Counts")
            # plt.title("bins = {}, n = {}".format(self.hist_bins, self.n_throws))

            plt.bar(
                bin_ctr,
                new_frequencies,
                bin_size,
                alpha=0.5,
                edgecolor="black",
                hatch="///",
                linewidth=0.5,
                label=r"First Hit Multiplier = {:.2f}".format(hit_rms[0]),
            )

            # plt.xlim(right=8e7)
            # plt.ylim(top=np.max(freq)+2)
            plt.legend(
                title="{:} showers, bins = {}, n = {}".format(
                    self.n_showers, self.hist_bins, self.n_throws
                ),
                # loc="upper right",
            )
            plt.show()

        return col_depth, col_mean, hit_rms[0]

    def sampling_per_depth(self, return_rms_dist=False):
        r"""
        Iterating through the composite shower to assign rms for each depth
        by sampling each point in that depth.
        """
        print("Sampling the shower variability at each grammage per sample grammage.")
        # left_pad_width = 500
        # shwr_depth = np.ones(np.shape(self.sample_shwr_col)[0])
        # shwr_mean = np.ones(np.shape(self.sample_shwr_col)[0])
        sample_shwr_rms = np.ones(self.sample_dpth_col.shape[1])

        for i, (depths, showers) in enumerate(
            zip(self.sample_dpth_col.T, self.sample_shwr_col.T)
        ):

            showers = showers[~np.isnan(showers)]

            _, _, hit_rms = self.mc_drt_rms(col_depths=depths, col_showers=showers)
            # shwr_depth[i] = col_depth
            # shwr_mean[i] = col_mean
            sample_shwr_rms[i] = hit_rms
            # print(hit_rms)

        return sample_shwr_rms, self.output_depth.T, self.output_mean.T

    def sampling_nmax_per_depth(self, return_rms_dist=False):
        r"""
        Iterating through the composite shower to assign rms for each depth
        by sampling showers near nmax per slant depth. Non-uniform multipliers
        """
        print("Sampling the shower variability at x_max per sample grammage.")
        max_sample_shwr_rms = np.ones(self.sample_dpth_col.shape[1])

        for i, (depths, showers) in enumerate(
            zip(self.sample_dpth_col.T, self.sample_shwr_col.T)
        ):

            _, _, hit_rms = self.mc_drt_rms(
                col_depths=depths,
                col_showers=self.nmax_shwr_col,
            )
            # print(hit_rms)
            max_sample_shwr_rms[i] = hit_rms

        return max_sample_shwr_rms, self.output_depth.T, self.output_mean.T

    def sampling_nmax_once(self, return_rms_dist=False):
        r"""
        Sample the rms distribution once around nmax and return multipliers from
        """
        print("Sampling the shower variability at x_max.")
        _, _, hit_rms = self.mc_drt_rms(
            col_depths=self.xmax_dpth_col,
            col_showers=self.nmax_shwr_col,
            # plot_darts=True,
        )
        # print(self.nmax_shwr_col)

        if return_rms_dist is True:
            return (
                hit_rms,
                self.output_depth.T,
                self.output_mean.T,
                self.nmax_shwr_col.T,
            )
        else:
            return hit_rms, self.output_depth.T, self.output_mean.T

    def sample_specific_grammage(self, grammage, return_rms_dist=False):
        r"""
        Sample the rms distribution once at a specfic grammage
        """
        print(f"Sampling the shower variability at {grammage}")

        depth_column_idx = np.argmin(np.abs(self.mean_depth - grammage))
        depth_column = self.depths[:, depth_column_idx].T  # this should be = grammage
        shower_column = self.showers[:, depth_column_idx].T
        _, _, hit_rms = self.mc_drt_rms(
            col_depths=depth_column,
            col_showers=shower_column,
            # plot_darts=True,
        )
        # print(self.nmax_shwr_col)

        if return_rms_dist is True:
            return (
                hit_rms,
                self.output_depth.T,
                self.output_mean.T,
                shower_column,
            )
        else:
            return hit_rms, self.output_depth.T, self.output_mean.T