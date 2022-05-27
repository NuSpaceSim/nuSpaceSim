import numpy as np
import matplotlib.pyplot as plt
import warnings
from .comp_eas_utils import get_decay_channel


def decay_channel_mult_plt(
    bins,
    showers,
    smpl_rms_plt=False,
    sampl_grm=None,
    sampl_lwr_err=None,
    sampl_upr_err=None,
):
    r"""
    Mosiac of longitudinal profiles for each decay channel in a set of composoite showers.

    Sample usage:
    -------------
    decay_channel_mult_plt(bins=comp_depths_00km, showers=trimmed_showers_00km)
    """

    decay_channels = np.unique(bins[:, 1])  # [0:2]
    plt.figure(figsize=(25, 15), dpi=200)

    for i, dc in enumerate(decay_channels, start=1):

        x = bins[bins[:, 1] == dc]
        y = showers[showers[:, 1] == dc]

        plt.subplot(4, 7, i)
        for depth, shower in zip(x, y):
            # iterate and plot each shower in that decay channel
            event_num = depth[0]
            decay_code = depth[1]
            plt.plot(
                depth[2:],
                shower[2:],
                alpha=0.3,
                linewidth=1.0,
                # label = str(event_num)+"|"+ str(decay_code)
            )

        decay_channel = get_decay_channel(dc)
        plt.title(decay_channel)
        # plt.legend()

        # plt.ylabel('N Particles')
        # plt.xlabel('Slant Depth')
        plt.yscale("log")

        if smpl_rms_plt is True:

            plt.fill_between(
                sampl_grm,
                sampl_lwr_err,
                sampl_upr_err,
                alpha=0.4,
                facecolor="black",
                zorder=20,
            )


def mean_rms_plt(bins, showers, plot_mean_rms=False, remove_tags=True, **kwargs):

    comp_showers = np.copy(showers[:, 2:])
    bin_lengths = np.nansum(np.abs(bins[:, 2:]), axis=1)

    longest_shower_idx = np.argmax(bin_lengths)
    longest_shower_bin = bins[longest_shower_idx, 2:]

    with warnings.catch_warnings():
        # take average along each bin, ignoring nans
        warnings.simplefilter("ignore", category=RuntimeWarning)

        average_composites = np.nanmean(comp_showers, axis=0)

        # test = average_composites  - comp_showers
        # take the square root of the mean of the difference between the average
        # and each particle content of each shower for one bin, squared
        rms_error = np.sqrt(
            np.nanmean((average_composites - comp_showers) ** 2, axis=0)
        )
        rms = np.sqrt(np.nanmean((comp_showers) ** 2, axis=0))
        std = np.nanstd(comp_showers, axis=0)
        err_in_mean = np.nanstd(comp_showers, axis=0) / np.sqrt(
            np.sum(~np.isnan(comp_showers), 0)
        )
    rms_low = average_composites - rms_error
    rms_high = average_composites + rms_error
    if plot_mean_rms is True:
        plt.figure(figsize=(8, 6))
        plt.plot(longest_shower_bin, average_composites, "--k", label="mean")
        # plt.plot(longest_shower, rms_error ,  '--r', label='rms error')
        # plt.plot(longest_shower, rms ,  '--g', label='rms')
        # plt.plot(longest_shower, std ,  '--y', label='std')
        # plt.plot(longest_shower, err_in_mean ,  '--b', label='error in mean')

        plt.fill_between(
            longest_shower_bin,
            rms_low,
            rms_high,
            alpha=0.2,
            # facecolor='crimson',
            interpolate=True,
            **kwargs,
        )

        plt.title("Mean and RMS Error")
        plt.ylabel("Number of Particles")
        # plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("Shower stage")
        # plt.yscale('log')
        plt.grid(True, which="both", linestyle="--")
        plt.ylim(bottom=1)
        # plt.xlim(right=1500)
        plt.legend()
    # plt.show()

    return longest_shower_bin, average_composites, rms_low, rms_high


def recursive_plt(composite_dpths, composite_shwrs, lbl="None", **kwargs):
    r"""Loops through each paired list and plots them on the same axes"""
    for depths, showers in zip(composite_dpths, composite_shwrs):

        event_num = depths[0]
        decay_code = depths[1]

        plt.plot(
            depths[2:],
            showers[2:],
            alpha=0.2,
            **kwargs,
        )

    plt.plot(
        composite_dpths[1, 2:],
        composite_shwrs[1, 2:],
        alpha=0.2,
        label=lbl,
        zorder=0,
        **kwargs,
    )
