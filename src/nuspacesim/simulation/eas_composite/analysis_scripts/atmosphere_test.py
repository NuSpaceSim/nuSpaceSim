from nuspacesim.simulation.eas_optical.atmospheric_models import (
    cummings_atmospheric_density,
    us_std_atm_density,
    slant_depth,
)
import numpy as np
import matplotlib.pyplot as plt

stopping_altitude = 30
altitudes = np.linspace(0, stopping_altitude, 1000)  # km
atm_dens = cummings_atmospheric_density(altitudes)

plt.figure(figsize=(8, 6), dpi=200)
plt.scatter(altitudes, atm_dens)
# plt.yscale("log")
plt.ylabel("Atm Density (g/cm^3)")
plt.xlabel("Altitutde (km)")
#%%

plt.figure(figsize=(8, 6), dpi=200)
angles = np.radians(np.array([95]))  # np.radians(np.linspace(0, 80, 5))

slant_depths_per_angle = []

for angle in angles:

    slant_depths = []

    for i, alt in enumerate(altitudes):
        g_cm2 = slant_depth(0, alt, angle)
        slant_depths.append(g_cm2)

    slt_dpths = np.array(slant_depths)[:, 0]

    slant_depths_per_angle.append(slt_dpths)

slant_depths_per_angle = np.array(slant_depths_per_angle)


for i, dpth in enumerate(slant_depths_per_angle):

    plt.scatter(
        altitudes,
        dpth,
        label=r"{:.2f}$\degree$".format(np.degrees(angles[i])),
    )

plt.title("track and earth zenith")
plt.ylabel("Integrated Slant Depth From Altitude to 20 km")
plt.xlabel("Altitutde (km)")
# plt.yscale('log')
plt.grid(visible=True)
plt.legend()
