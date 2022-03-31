from nuspacesim.simulation.eas_optical.atmospheric_models import (
    cummings_atmospheric_density,
    us_std_atm_density,
    slant_depth,
)
import numpy as np
import matplotlib.pyplot as plt

altitudes = np.linspace(0, 25, 50)  # km
atm_dens = cummings_atmospheric_density(altitudes)

plt.figure(figsize=(12, 6), dpi=200)
plt.scatter(altitudes, atm_dens)
plt.ylabel("Atm Density (g/cm^3)")
plt.xlabel("Altitutde (km)")

plt.figure(figsize=(12, 6), dpi=200)
angles = np.radians(np.linspace(0, 80, 5))


for angle in angles:
    slant_depths = []
    for i, alt in enumerate(altitudes[1:]):

        g_cm2 = slant_depth(alt, 100, angle)
        slant_depths.append(g_cm2)

    slt_dpths = np.array(slant_depths)
    plt.plot(
        altitudes[1:],
        slt_dpths[:, :1],
        label="{:.2f} degrees".format(np.degrees(angle)),
    )

plt.title("track and earth zenith")
plt.ylabel("Slant Depth from .5 km below altitude to given Altitude(g/cm^2)")
plt.xlabel("Altitutde (km)")
# plt.yscale('log')
plt.grid(visible=True)
plt.legend()
