import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial, polynomial
from scipy.interpolate import splev, splrep

from nuspacesim.simulation.eas_optical import atmospheric_models
from nuspacesim.simulation.eas_optical.quadeas import (
    aerosol_optical_depth,
    differential_ozone,
    ozone,
)


def atmospheric_density_fit():
    # slant-depth from gaussian quadriture
    matplotlib.rcParams.update({"font.size": 18})

    tabz = np.array([0, 0.5, 1, 1.5, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 65])
    # fmt: off
    tabr = np.array([
        1.225, 1.1673, 1.1116, 1.0581, 1.0065, 9.0925e-1, 7.3643e-1, 4.1351e-1,
        1.9476e-1, 8.891e-2, 1.8410e-2, 3.9957e-3, 1.0269e-3, 3.0968e-4, 1.6321e-4
    ]) * 1e-3
    # fmt: on

    N = int(1e3)
    alt_dec = np.linspace(0.0, 65.0, N)
    c_rhos = atmospheric_models.cummings_atmospheric_density(alt_dec)
    p_rhos = atmospheric_models.polyrho(alt_dec)
    us_rho = atmospheric_models.us_std_atm_density(alt_dec)

    spl = splrep(tabz, tabr)
    s_rho = splev(alt_dec, spl)

    fig, (ax1, ax2) = plt.subplots(2, 2, squeeze=True)
    ax1[0].plot(alt_dec, us_rho, "b-", label="76 Atmosphere implementation.")
    ax1[0].plot(alt_dec, c_rhos, "g:", label="Cummings")
    ax1[0].plot(alt_dec, p_rhos, "k--", label="Degree 10 Polynomial")
    ax1[0].plot(alt_dec, s_rho, "r-.", label="Spline")
    ax1[0].plot(tabz, tabr, "b+", markersize=10, label="76 Atmosphere Table values")
    ax1[0].set_ylabel("Density g/cm^3")
    ax1[0].set_xlabel("Altitude km")
    ax1[0].grid(True)
    ax1[0].legend()
    ax1[0].set_title("Atmospheric Density")

    ax2[0].plot(alt_dec, us_rho, "b-", label="76 Atmosphere implementation.")
    ax2[0].plot(alt_dec, c_rhos, "g:", label="Cummings")
    ax2[0].plot(alt_dec, p_rhos, "k--", label="Polynomial")
    ax2[0].plot(alt_dec, s_rho, "r-.", label="Spline")
    ax2[0].plot(tabz, tabr, "b+", markersize=10, label="76 Atmosphere Table values")
    ax2[0].set_yscale("log")
    ax2[0].set_ylabel("Log(Density g/cm^3)")
    ax2[0].set_xlabel("Altitude km")
    ax2[0].grid(True)

    resids = []

    splresid = np.sum((us_rho - s_rho) ** 2)

    for i in range(30):
        popt, rlst = Polynomial.fit(alt_dec, us_rho, i, full=True)
        resids.append(rlst[0])
        # print(rlst[0])

    ax1[1].plot(resids, ".")
    ax1[1].set_ylabel("Residual Error")
    ax1[1].set_xlabel("Degree of polynomial")
    ax1[1].set_title("Residual error of approximating polynomials.")
    ax1[1].grid(True)
    ax1[1].axhline(splresid, c="r", linestyle=":", label="Spline residual error")
    ax1[1].legend()

    ax2[1].plot(resids, ".")
    ax2[1].set_ylabel("log(Residual Error)")
    ax2[1].set_xlabel("Degree of polynomial")
    ax2[1].set_yscale("log")
    ax2[1].grid(True)
    ax2[1].axhline(splresid, c="r", linestyle=":", label="Spline residual error")

    fig.suptitle("Atmospheric Density Models")

    plt.show()


def atmospheric_ozone_fit(index):
    # slant-depth from gaussian quadriture
    matplotlib.rcParams.update({"font.size": 16})

    N = int(22)
    M = int(1e5)
    x = np.linspace(0.0, 100.0, N)
    z = np.linspace(0.0, 100.0, M)
    xoz = ozone(x)
    oz = ozone(z)
    doz = differential_ozone(z)

    fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True, squeeze=True)

    popt = Polynomial.fit(z, oz, index)
    spl = splrep(x, xoz)
    soz = splev(z, spl)

    ax1[0].plot(z, oz, "b-")
    ax1[0].plot(z, popt(z), "g:")
    ax1[0].plot(z, soz, "r--")
    ax1[0].set_xlabel("Altitude (KM)")
    ax1[0].set_ylabel("Ozone Depth")
    ax1[0].set_title("Ozone Depth")
    ax1[0].legend(["Nimbus", "Polynomial", "Spline"])
    ax1[0].grid(True)

    ax2[0].plot(z, oz, "b-")
    ax2[0].plot(z, popt(z), "g:")
    ax2[0].plot(z, soz, "r--")
    ax2[0].set_yscale("log")
    ax2[0].set_xlabel("Altitude (KM)")
    ax2[0].set_ylabel("Log(Ozone Depth)")
    ax2[0].grid(True)

    dopt = Polynomial.fit(z, doz, index)
    dsoz = -splev(z, spl, der=1)

    ax1[1].plot(z, doz, "b-")
    ax1[1].plot(z, dopt(z), "g:")
    ax1[1].plot(z, dsoz, "r--")
    ax1[1].set_xlabel("Altitude (KM)")
    ax1[1].set_ylabel("d(Ozone Depth)/d altitude")
    ax1[1].set_title("negative first derivative")
    ax1[1].legend(["Nimbus", "Polynomial", "Spline"])
    ax1[1].grid(True)

    ax2[1].plot(z, doz, "b")
    ax2[1].plot(z, dopt(z), "g:")
    ax2[1].plot(z, dsoz, "r--")
    ax2[1].set_yscale("log")
    ax2[1].set_xlabel("Altitude (KM)")
    ax2[1].set_ylabel("log(d(Ozone Depth)/d altitude)")
    ax2[1].grid(True)

    plt.show()
    return popt, spl


def atmospheric_aerosol_fit():
    matplotlib.rcParams.update({"font.size": 16})

    z = np.append(
        np.linspace(0.0, 33.0, 34),
        np.linspace(40, 100.0, 7),
    )
    # fmt: off
    aOD55 = np.array(
        [0.250, 0.136, 0.086, 0.065, 0.055, 0.049, 0.045, 0.042, 0.038, 0.035, 0.032,
         0.029, 0.026, 0.023, 0.020, 0.017, 0.015, 0.012, 0.010, 0.007, 0.006, 0.004,
         0.003, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
    """internal aOD55 array"""
    # fmt: on

    spl = splrep(z, aOD55)
    zz = np.linspace(0.0, 100.0, 1000)

    fig, (ax1) = plt.subplots(2, 1, sharex=True, squeeze=True)

    ax1[0].plot(zz, splev(zz, spl), "r-", label="Spline")
    ax1[0].plot(z, aOD55, "b+", markersize=14, label="aOD55")
    # ax1[0].plot(zz, aerosol_optical_depth(zz), "r:")
    ax1[0].set_xlabel("Altitude (KM)")
    ax1[0].set_ylabel("optical depth")
    ax1[0].set_title("optical depth")
    ax1[0].grid()
    ax1[0].legend()

    ax1[1].plot(zz, splev(zz, spl), "r-")
    ax1[1].plot(z, aOD55, "b+", markersize=14)
    ax1[1].set_xlabel("Altitude (KM)")
    ax1[1].set_ylabel("Log(optical depth)")
    ax1[1].set_yscale("log")
    ax1[1].grid()

    plt.show()

    # print(spl)


if __name__ == "__main__":
    atmospheric_density_fit()
    atmospheric_ozone_fit(26)
    atmospheric_aerosol_fit()
