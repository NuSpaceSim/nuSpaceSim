from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np

from nuspacesim.simulation.eas_optical.atmospheric_models import (
    cummings_atmospheric_density,
    polyrho,
    us_std_atm_density,
)

if __name__ == "__main__":
    zmax = 200
    zs = np.linspace(0, zmax, int(1e6))
    print(timeit(lambda: us_std_atm_density(zs), number=1))
    print(timeit(lambda: cummings_atmospheric_density(zs), number=1))
    print(timeit(lambda: polyrho(zs), number=1))

    for i in [0, 0.05, 1, 5, 10, 11, 12, 20, 30, 50, 65]:  # km
        print(
            f"{i}km \t76_atm: {us_std_atm_density(i)}\tg/cm^3 ~~ param: {cummings_atmospheric_density(i)}"
        )

    zs = np.linspace(0, 200, int(1e5))
    rs76 = us_std_atm_density(zs)
    rsPr = cummings_atmospheric_density(zs)

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
