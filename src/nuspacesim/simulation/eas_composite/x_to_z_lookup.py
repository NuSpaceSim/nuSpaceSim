import numpy as np
import matplotlib.pyplot as plt
from nuspacesim.simulation.eas_optical.atmospheric_models import (
    cummings_atmospheric_density,
    us_std_atm_density,
    slant_depth,
    slant_depth_integrand,
)


def depth_to_alt_lookup(slant_depths, angle, starting_alt, direction="up", s=2000):
    max_alt = 256  # the nominal stopping point of the atm in km
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
    # filter for when it breaks

    max_val = np.argmax(out_alts)
    out_alts[max_val + 1 :] = np.nan
    return out_alts


def depth_to_alt_lookup_v2(slant_depths, angle, starting_alt, direction="up", s=8000):
    """
    by fred
    """
    max_alt = 256  # the nominal stopping point of the atm in km
    angle = np.radians(angle)

    x = []

    if direction == "down":
        # integrate from all altitude to the max altitude (in km)
        altitudes = np.linspace(0, starting_alt, s)
        for alt in altitudes:

            g_cm2, _ = slant_depth(
                alt,
                starting_alt,
                angle,
                earth_radius=6371,
                func=lambda y, theta_tr, earth_radius: slant_depth_integrand(
                    y,
                    theta_tr=theta_tr,
                    rho=lambda z: us_std_atm_density(z, earth_radius),
                    earth_radius=earth_radius,
                ),
            )
            x.append(g_cm2)
    elif direction == "up":
        # integrate from 0 to a given altitude
        altitudes = np.linspace(starting_alt, max_alt, s)
        for alt in altitudes:

            g_cm2, err = slant_depth(
                starting_alt,
                alt,
                angle,
                earth_radius=6371,
                func=lambda y, theta_tr, earth_radius: slant_depth_integrand(
                    y,
                    theta_tr=theta_tr,
                    rho=lambda z: us_std_atm_density(z, earth_radius),
                    earth_radius=earth_radius,
                ),
            )
            x.append(g_cm2)
    else:
        print("not a valid trajectory")
        # return np.nan

    look_up_depths = np.array(x)

    residuals = np.abs(look_up_depths - slant_depths[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1)
    out_alts = altitudes[closest_match_idxs]

    # high_res_slant_depth = np.linspace(0, max_alt, 1500)
    # interpolated_altitudes = np.interp(high_res_slant_depth, xp=slant_depths, fp=out_alts)

    max_val = np.argmax(out_alts)
    out_alts[max_val + 1 :] = np.nan

    return out_alts


# #%% test for the function above


# #%%
# plt.figure(figsize=(4, 4), dpi=200)
# slant_depths = np.linspace(0, 15000, 10000)
# altitudes = depth_to_alt_lookup(slant_depths, 95, starting_alt=0, direction="up")
# new_altitudes = depth_to_alt_lookup_v2(slant_depths, 95, starting_alt=0, direction="up")

# plt.plot(slant_depths, altitudes, label="old version")


# plt.plot(slant_depths, new_altitudes, label="with Alex's input")

# plt.xlabel("slant depth (g / cm^2)")
# plt.ylabel("altitude (km)")
# plt.legend()
# # plt.plot(interpolated_altitudes , high_res_slant_depth )
# # plt.plot(interpolated_altitudes , high_res_slant_depth )

# #%%
# z_lo = np.array([0])
# z_hi = np.linspace(0, 100, 5)
# theta_tr = np.radians([95])

# gcm2, err = slant_depth(
#     z_lo,
#     z_hi,
#     theta_tr,
#     earth_radius=6371,
#     func=lambda y, theta_tr, earth_radius: slant_depth_integrand(
#         y,
#         theta_tr=theta_tr,
#         rho=lambda z: us_std_atm_density(z, earth_radius),
#         earth_radius=earth_radius,
#     ),
# )
# plt.scatter(gcm2, z_hi)
# # plt.xscale("log")
# # plt.yscale("log")
