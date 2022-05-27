import numpy as np


def bin_nmax_xmax(bins, particle_content):
    r"""
    given an array of Slant Depths and Particle Content values for the same particle
    (can be any number of events, but need to be same size), returns the Nmax and Xmax Values

    per row (if composite showers and bins are inputted, per event)

    intended to use for nmax and xmax distribution analysis
    """

    try:
        bin_nmax = np.nanmax(particle_content, axis=1)
        bin_nmax_pos = np.nanargmax(particle_content, axis=1)
        bin_xmaxs = bins[np.arange(len(bins)), bin_nmax_pos]
    except:
        bin_nmax = np.nanmax(particle_content)
        bin_nmax_pos = np.nanargmax(particle_content)
        bin_xmaxs = bins[bin_nmax_pos]

    return bin_nmax, bin_xmaxs


def decay_channel_filter(
    shwr_dpths,
    shwr_n,
    decay_channel,
    nth_digit=None,
    digit_flag=None,
    get_discarded=None,
):
    r"""Filter out specific decay channels or decay channel type"""
    if nth_digit is not None and digit_flag is not None:

        n_mask = shwr_dpths[:, 1][nth_digit - 1] == digit_flag

        out_shwr_dpths = shwr_dpths[n_mask]
        out_shwr_n = shwr_n[n_mask]

        out_not_shwr_dpths = shwr_dpths[~n_mask]
        out_not_shwr_n = shwr_n[~n_mask]
    else:
        decay_mask = shwr_dpths[:, 1] == decay_channel

        out_shwr_dpths = shwr_dpths[decay_mask]
        out_shwr_n = shwr_n[decay_mask]

        out_not_shwr_dpths = shwr_dpths[~decay_mask]
        out_not_shwr_n = shwr_n[~decay_mask]

    if get_discarded is not None:
        return out_shwr_dpths, out_shwr_n, out_not_shwr_dpths, out_not_shwr_n
    else:
        return out_shwr_dpths, out_shwr_n


def get_decay_channel(decay_code):
    r"""
    PYTHIA 8 Decay Codes to Corresponding Decay Codes
    https://drive.google.com/file/d/1MVj0FhWNI-075oZQwM8NWSwateT0xqJH/view
    Decay Code Format — 6 digit number
    — 1st number = number of daughters (range: 2 - 6)
    — 2nd number = kaon flag (0 or 1)
    — 3rd number = eta/omega flag (0 or 1)
    — 4th number = number of pi0s (range: 0 - 4)
    — 5th number = number of charged pions (range: 0 - 5)
    — 6th number = number to differentiate between decays with similarities
    in the other numbers
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


def numpy_argmax_reduceat(arr, group_idxs):
    r"""Get the indeces of maximum values within a grouping.

    Parameters
    ----------
    arr: array
        flattened array of all the values to find the maximum of each group
    group_idxs: int
        indeces of the start of each group, can be found by np.unique

    Returns
    -------
    max_in_grp_idx: array
        indeces of the maximum of each grouping

    Based on: https://stackoverflow.com/a/41835843
    """
    n = arr.max() + 1
    id_arr = np.zeros(arr.size, dtype=int)
    id_arr[group_idxs[1:]] = 1
    shift = n * id_arr.cumsum()
    sortidx = (arr + shift).argsort()
    grp_shifted_argmax = np.append(group_idxs[1:], arr.size) - 1
    max_in_grp_idx = sortidx[grp_shifted_argmax]
    return max_in_grp_idx


def separate_showers(shwr_dpths, shwr_n, sep_dpth, sep_n):
    r"""
    Bifurcate shower based on an arbitrary particle content threshold.
    """
    dpth_idx = int(np.argwhere(shwr_dpths[0, :] == sep_dpth))
    shwr_content_at_depth = shwr_n[:, dpth_idx]

    above_n_mask = shwr_content_at_depth > sep_n
    below_n_mask = shwr_content_at_depth <= sep_n

    above_depths = shwr_dpths[above_n_mask]
    below_depths = shwr_dpths[below_n_mask]
    above_showers = shwr_n[above_n_mask]
    below_showers = shwr_n[below_n_mask]

    return below_depths, below_showers, above_depths, above_showers
