import numpy as np


def bin_nmax_xmax(bins, particle_content):
    r"""
    given an array of Slant Depths and Particle Content values for the same particle
    (can be any number of events, but need to be same size), returns the Nmax and Xmax Values

    per row (if composite showers and bins are inputted, per event)

    intended to use for nmax and xmax distribution analysis
    """

    try:
        bin_nmax = np.amax(particle_content, axis=1)
        bin_nmax_pos = np.nanargmax(particle_content, axis=1)
        bin_xmaxs = bins[np.arange(len(bins)), bin_nmax_pos]
    except:
        bin_nmax = np.amax(particle_content)
        bin_nmax_pos = np.nanargmax(particle_content)
        bin_xmaxs = bins[bin_nmax_pos]

    return bin_nmax, bin_xmaxs


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
