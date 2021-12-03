from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np

from nuspacesim.simulation.eas_optical.atmospheric_models import polyrho, rho


def us_std_atm_density(z, earth_radius=6371):
    H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852])
    Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
    T_b = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    P_b = (
        np.array(
            [
                1.0,
                2.233611e-1,
                5.403295e-2,
                8.5666784e-3,
                1.0945601e-3,
                6.6063531e-4,
                3.9046834e-5,
                3.68501e-6,
            ]
        )
        * 1.01325e5
    )

    Rstar = 8.31432e3
    M0 = 28.9644
    gmr = 34.163195

    z = np.asarray(z)

    h = z * earth_radius / (z + earth_radius)
    i = np.searchsorted(H_b, h, side="right") - 1

    deltah = h - H_b[i]

    temperature = T_b[i] + Lm_b[i] * deltah

    mask = Lm_b[i] == 0
    pressure = np.full(z.shape, P_b[i])
    pressure[mask] *= np.exp(-gmr * deltah[mask] / T_b[i][mask])
    pressure[~mask] *= (T_b[i][~mask] / temperature[~mask]) ** (gmr / Lm_b[i][~mask])

    density = (pressure * M0) / (Rstar * temperature)  # kg/m^3
    return 1e-3 * density  # g/cm^3


if __name__ == "__main__":

    zs = np.linspace(0, 65, int(1e6))
    print(timeit(lambda: us_std_atm_density(zs), number=1))
    print(timeit(lambda: rho(zs), number=1))
    print(timeit(lambda: polyrho(zs), number=1))

    for i in [0, 0.05, 1, 5, 10, 11, 12, 20, 30, 50, 65]:  # km
        print(f"{i}km \t76_atm: {us_std_atm_density(i)}\tg/cm^3 ~~ param: {rho(i)}")

    zs = np.linspace(0, 65, int(1e5))
    rs76 = us_std_atm_density(zs)
    rsPr = rho(zs)

    diff = rs76 - rsPr
    ratio = rs76 / rsPr

    fig, (ax1, ax2) = plt.subplots(2, 2)
    ax1[0].plot(zs, rsPr, label="Parameterization rho")
    ax1[0].plot(zs, rs76, label="US std 76 atmosphere rho")
    ax1[0].set_xlabel("geometric altitude")
    ax1[0].set_ylabel("density g/cm^3")

    ax2[0].plot(zs, rsPr, label="Parameterization rho")
    ax2[0].plot(zs, rs76, label="US std 76 atmosphere rho")
    ax2[0].set_xlabel("geometric altitude")
    ax2[0].set_ylabel("log(density g/cm^3)")
    ax2[0].set_yscale("log")

    ax1[1].plot(zs, diff, label="difference")
    ax1[1].plot(zs, ratio, label="ratio")
    ax1[1].set_xlabel("geometric altitude")
    ax1[1].set_ylabel("deviation")

    ax2[1].plot(zs, diff, label="difference")
    ax2[1].plot(zs, ratio, label="ratio")
    ax2[1].set_xlabel("geometric altitude")
    ax2[1].set_ylabel("log(deviation)")
    ax2[1].set_yscale("log")

    ax1[0].legend()
    ax1[1].legend()

    plt.show()
