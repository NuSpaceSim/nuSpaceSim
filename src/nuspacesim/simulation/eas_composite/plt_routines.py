import numpy as np
import matplotlib.pyplot as plt


def get_decay_channel(decay_code):
    r"""
    PYTHIA 8 Decay Codes to Corresponding Decay Codes
    """
    decay_dict = {
        200011: r"$\tau \rightarrow \nu_\tau + \pi$",
        210001: r"$\tau \rightarrow \nu_\tau + K$",
        300001: r"$\tau \rightarrow \nu_\tau + e + \nu_e$",
        300002: r"$\tau \rightarrow \nu_\tau + \mu + nu_mu$",
        300111: r"$\tau \rightarrow \nu_\tau + \pi_0 + \pi$",
        310001: r"$\tau \rightarrow \nu_\tau + K_0 + K$",
        311001: r"$\tau \rightarrow \nu_\tau + \eta + K$",
        310011: r"$\tau \rightarrow \nu_\tau + \pi_0 + \overline{K_0}$",
        310101: r"$\tau \rightarrow \nu_\tau + \pi_0 + K$",
        311002: r"$\tau \rightarrow \nu_\tau + \Omega + K$",
        311003: r"$\tau \rightarrow \nu_\tau + \eta + K^*(892)$",
        400211: r"$\tau \rightarrow \nu_\tau + 2\pi_0 + \pi$",
        400031: r"$\tau \rightarrow \nu_\tau + \pi^+ + \pi^- +\pi$",
        410111: r"$\tau \rightarrow \nu_\tau + \pi_0 + \pi + \overline{K_0}$",
        410021: r"$\tau \rightarrow \nu_\tau + \pi^+ + \pi^- + K$",
        410011: r"$\tau \rightarrow \nu_\tau + \pi + K^+ + K^- $",
        410101: r"$\tau \rightarrow \nu_\tau + \pi_0 + K_0 + K$",
        410012: r"$\tau \rightarrow \nu_\tau + \pi + KS + KL$",
        410201: r"$\tau \rightarrow \nu_\tau + 2\pi_0 + K$",
        410013: r"$\tau \rightarrow \nu_\tau + \pi + KL + KL$",
        410014: r"$\tau \rightarrow \nu_\tau + \pi + KS + KS$",
        401111: r"$\tau \rightarrow \nu_\tau + \eta + \pi_0 + \pi$",
        400111: r"$\tau \rightarrow \nu_\tau + \gamma + \pi_0 + \pi$",
        500131: r"$\tau \rightarrow \nu_\tau + \pi_0 + \pi^+ + \pi^- + \pi$",
        500311: r"$\tau \rightarrow \nu_\tau + 3\pi_0 + \pi$",
        501031: r"$\tau \rightarrow \nu_\tau + \pi^+ + \pi^- + \pi + \eta$",
        501211: r"$\tau \rightarrow \nu_\tau + 2\pi_0 + \pi + \eta$",
        501212: r"$\tau \rightarrow \nu_\tau + 2\pi_0 + \pi + \Omega$",
        501032: r"$\tau \rightarrow \nu_\tau + \pi^+ + \pi^- + \pi + \Omega$",
        510301: r"$\tau \rightarrow \nu_\tau + \3\pi_0 + K$",
        510121: r"$\tau \rightarrow \nu_\tau + \pi_0 + \pi^+ + \pi^- + K$",
        510211: r"$\tau \rightarrow \nu_\tau + 2\pi_0 + \overline{K_0} + \pi$",
        510031: r"$\tau \rightarrow \nu_\tau + \overline{K_0} + \pi^+ + \pi^- + \pi$",
        510111: r"$\tau \rightarrow \nu_\tau + \pi_0 + K_0 + \overline{K_0} + \pi$",
        510112: r"$\tau \rightarrow \nu_\tau + \pi_0 + K^+ + K^- + \pi$",
        600231: r"$\tau \rightarrow \nu_\tau + 2\pi_0 + \pi^+ + \pi^- + \pi$",
        600411: r"$\tau \rightarrow \nu_\tau + 4\pi_0 + \pi$",
        600051: r"$\tau \rightarrow \nu_\tau + \pi^+ + \pi^- + \pi^+ + \pi^- + \pi$",
    }
    return decay_dict[decay_code]


def mean_rms(bins, showers, plot_mean_rms=False, remove_tags=True, **kwargs):

    comp_showers = np.copy(showers[:, 2:])
    bin_lengths = np.nansum(np.abs(bins[:, 2:]), axis=1)

    longest_shower_idx = np.argmax(bin_lengths)
    longest_shower_bin = bins[10, 2:]
    # take average a long each bin, ignoring nans
    average_composites = np.nanmean(comp_showers, axis=0)

    # test = average_composites  - comp_showers
    # take the square root of the mean of the difference between the average
    # and each particle content of each shower for one bin, squared
    rms_error = np.sqrt(np.nanmean((average_composites - comp_showers) ** 2, axis=0))
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
            **kwargs
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
