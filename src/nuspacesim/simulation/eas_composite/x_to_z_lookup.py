import numpy as np
from nuspacesim.simulation.eas_optical.atmospheric_models import (
    cummings_atmospheric_density,
    us_std_atm_density,
    slant_depth,
)


def depth_to_alt_lookup(slant_depths, angle, starting_alt, direction="up", s=2000):
    max_alt = 150  # the nominal stopping point of the atm in km
    angle = np.radians(angle)

    x = []

    if direction == "down":
        # integrate from all altitude to the max altitude (in km)
        altitudes = np.linspace(0, starting_alt, s)
        for alt in altitudes:
            g_cm2 = slant_depth(alt, starting_alt, angle)
            x.append(g_cm2)
    elif direction == "up":
        # integrate from 0 to a given altitude
        altitudes = np.linspace(starting_alt, max_alt, s)
        for alt in altitudes:
            g_cm2 = slant_depth(starting_alt, alt, angle)
            x.append(g_cm2)
    else:
        print("not a valid trajectory")
        # return np.nan

    look_up_depths = np.array(x)[:, 0]

    residuals = np.abs(look_up_depths - slant_depths[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1)
    out_alts = altitudes[closest_match_idxs]

    # high_res_slant_depth = np.linspace(0, max_alt, 1500)
    # interpolated_altitudes = np.interp(high_res_slant_depth, xp=slant_depths, fp=out_alts)

    return out_alts


# #%% test for the function above
# import matplotlib.pyplot as plt

# slant_depths = np.linspace(0, 17500, 10000)
# altitudes = depth_to_alt_lookup(slant_depths, 95, starting_alt=6, direction="up")
# plt.figure(figsize=(8, 6), dpi=200)
# plt.scatter(slant_depths, altitudes, s=0.01)
# # plt.plot(interpolated_altitudes , high_res_slant_depth )
