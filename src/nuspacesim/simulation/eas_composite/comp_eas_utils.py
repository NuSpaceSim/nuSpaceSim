import numpy as np
from scipy.optimize import fsolve


def altittude_to_depth(z):
    """
    Calculate Overabundance as a function of altitude.
    Taken from Alex's transciption of John's code.
    """
    z = np.array(z)
    X = np.empty_like(z)
    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25
    X[mask1] = np.power(((z[mask1] - 44.34) / -11.861), (1 / 0.19))
    X[mask2] = np.exp(np.divide(z[mask2] - 45.5, -6.34))
    X[mask3] = np.exp(np.subtract(13.841, np.sqrt(28.920 + 3.344 * z[mask3])))

    rho = np.empty_like(z)
    rho[mask1] = (
        -1.0e-5
        * (1 / 0.19)
        / (-11.861)
        * ((z[mask1] - 44.34) / -11.861) ** ((1.0 / 0.19) - 1.0)
    )
    rho[mask2] = np.multiply(-1e-5 * np.reciprocal(-6.34), X[mask2], dtype=np.float32)
    rho[mask3] = np.multiply(
        np.divide(0.5e-5 * 3.344, np.sqrt(28.920 + 3.344 * z[mask3])), X[mask3]
    )
    return X  # , rho


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


def calc_alpha(obs_height, earth_emergence_angle):
    """
    Calculates the angle of the detector’s line of sight (respect to the local zenith),
    alpha
    """

    def f(xy, r_earth=6371):

        x, y = xy
        z = np.array(
            [
                y
                - (np.cos(np.radians(earth_emergence_angle)) / (r_earth + obs_height)),
                y - (np.cos(x) / r_earth),
            ]
        )
        return z

    alpha_rads = fsolve(f, [0, 1])[0]
    print(fsolve(f, [0, 1])[1])
    alpha_degs = np.degrees(alpha_rads) % 360
    print(alpha_degs)
    if alpha_degs >= 180:
        alpha_degs = 360 - alpha_degs

    return alpha_degs


def decay_channel_filter(
    shwr_dpths,
    shwr_n,
    decay_channel,
    nth_digit=None,
    digit_flag=None,
    get_discarded=None,
):
    r"""Filter out specific decay channels or decay channel type"""

    # If you want all decay channels that have the nth_digit equal to the digit_flag
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


def depth_to_altitude(x):
    x = np.array(x)
    altitude_out = np.empty_like(x)

    # for altitudes z < 11
    altitude = (-11.861 * x ** 0.19) + 44.34
    mask1 = altitude < 11
    altitude_out[mask1] = altitude[mask1]

    # for altitudes z >= 11, z < 25
    altitude = -6.34 * np.log(x) + 45.5
    mask2 = (altitude >= 11) & (altitude < 25)
    altitude_out[mask2] = altitude[mask2]

    # for altitudes  z >= 25
    altitude = ((13.841 - np.log(x)) ** 2 - 28.920) / 3.344
    mask3 = altitude >= 25
    altitude_out[mask3] = altitude[mask3]

    return altitude_out


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
        300002: r"$\tau \rightarrow \nu_\tau + \mu + \nu_\mu$",
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
    if shwr_dpths.shape[1] < sep_dpth:
        print("showers don't propagate past beyond sep_depth")
    dpth_idx = int(np.argwhere(shwr_dpths[0, :] == sep_dpth))
    shwr_content_at_depth = shwr_n[:, dpth_idx]

    above_n_mask = shwr_content_at_depth > sep_n
    below_n_mask = shwr_content_at_depth <= sep_n

    above_depths = shwr_dpths[above_n_mask]
    below_depths = shwr_dpths[below_n_mask]
    above_showers = shwr_n[above_n_mask]
    below_showers = shwr_n[below_n_mask]

    return below_depths, below_showers, above_depths, above_showers


def shabita_depth_to_altitude(x):
    """
    Calculate Overabundance (vertical depth) as a function of altitude.
    Taken from T.K. Gaisser's book
    """

    x = np.array(x)
    altitude_out = np.empty_like(x)
    mask1 = x < 25
    mask2 = (x >= 25) & (x < 230)
    mask3 = x >= 230
    altitude_out[mask1] = (
        47.05 - 6.9 * np.log(x[mask1]) + 0.299 * np.log(x[mask1] / 10) ** 2
    )
    altitude_out[mask2] = 45.5 - 6.34 * np.log(x[mask2])
    altitude_out[mask3] = 44.34 - 11.861 * (x[mask3]) ** 0.19
    return altitude_out


def slant_depth_to_alt(
    earth_emergence_ang,
    slant_depths,
    obs_height=33,
    alt_stop=100,
    alt_smpl=1e4,
    sci_plots=False,
):
    r"""
    Optimized for upward going showers.
    Translates slant depth to altitude via simple monte-carlo.

    Parameters
    ----------
    earth_emergence_ang : float
        Earth emergence angle, beta, in degrees
    slant_depths : array
        The array of slant depths to be translated
    obs_height : int
        The observing height of the satelite in km, default is 33 km.
    alt_stop : int
        Stopping altitude of the atmosphere (km), tied to obs_height.
        Default is 100 km
    alt_smpl : int
        Sampling rate of the constructed altitude lookup vector.
        Higher is better, but slower.
        Default is 1000 bins.
    sci_plots : bool
        If True, plot the performance of the translator.

    Returns
    ----------
    out_alts : the altitude (km) corresponding to the slant depth array

    Examples
    --------
    test = slant_depth_to_alt(
        earth_emergence_ang=5, slant_depths=np.linspace(0, 10000, 1000), sci_plots=True
    )

    """
    r_earth = 6371  # km

    # define an altitude array to use as a look-up vector.
    altitude_array = np.geomspace(1e-6, alt_stop, int(alt_smpl))

    # for given altitude, calculate vertical depth
    depths = altittude_to_depth(altitude_array)
    lower_vertical_depths = depths[:-1]
    upper_vertical_depths = depths[1:]

    # calculate path length
    delta_vertical_depth = lower_vertical_depths - upper_vertical_depths

    # calculate alpha given earth emergance angle and beta by setting "a" equal to 0
    alpha_deg = np.degrees(
        np.arcsin(
            (np.cos(np.radians(earth_emergence_ang)) * r_earth) / (r_earth + obs_height)
        )
    )
    # beta prime is an array of how the earth emergence angle changes via z
    beta_prime = np.degrees(
        np.arccos(
            (np.sin(np.radians(alpha_deg)) * (r_earth + obs_height))
            / (r_earth + altitude_array[1:])
        )
    )
    # take the path length and correct it by beta_prime, which evolves via z
    corrected_path_length = delta_vertical_depth / np.sin(np.radians(beta_prime))

    upper_slant_depth = np.cumsum(corrected_path_length)

    # note, the linspace nature of the altitude array may yield inaccurate results
    # when the slant depth changes drastically
    residuals = np.abs(upper_slant_depth - slant_depths[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1)
    out_alts = altitude_array[closest_match_idxs]

    if out_alts[-1] == out_alts[-2]:
        print("> Note, you have hit the top of the atmosphere.")
        print("> Either change  the top of the atmosphere alt_stop (default: 100 km), ")
        print("  or cut off the shower at a smaller slant depth.")

    if sci_plots is True:
        # see how much the path length is corrected
        import matplotlib.pyplot as plt

        plt.figure(dpi=300)
        plt.plot(altitude_array[1:], delta_vertical_depth, label="path length")
        plt.plot(
            altitude_array[1:], corrected_path_length, label="corrected path length"
        )
        plt.xlabel("altitude (km)")
        plt.ylabel("$\Delta X$")
        # plt.ylim(-5, corrected_path_length.max() + 10)
        plt.legend(title=r"$\beta = {}\degree$".format(earth_emergence_ang))

        # plot how the local emergence angle evolves with altitude
        plt.figure(dpi=300)
        plt.plot(
            altitude_array[1:],
            beta_prime,
            label=r"$\beta = {}\degree$".format(earth_emergence_ang),
        )
        plt.xlabel("altitude (km)")
        plt.ylabel(r"$\beta'$")
        plt.legend(title=r"$\beta = {}\degree$".format(earth_emergence_ang))

        # plot the altitude and slant depth as well as the performance of
        # the look up table
        plt.figure(dpi=300)
        plt.plot(
            upper_vertical_depths, altitude_array[1:], lw=3, label="vertical depth"
        )
        plt.plot(upper_slant_depth, altitude_array[1:], lw=3, label="slant depth")
        plt.plot(
            slant_depths,
            out_alts,
            ls=":",
            c="red",
            alpha=0.8,
            label="reconstructed slant depth",
            zorder=2,
        )
        plt.axvline(1030, ls="--", c="k", label=" 1030 g/cm2")

        plt.ylabel("altitude (km)")
        plt.xlabel(r"depth ($g / cm^{-2})$")
        plt.legend(title=r"$\beta = {}\degree$".format(earth_emergence_ang))

    return out_alts
