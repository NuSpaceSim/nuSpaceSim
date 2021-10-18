from nuspacesim.simulation.eas_optical.atmospheric_models import (
    rho,
    polyrho,
    slant_depth_integrand,
)

import numpy as np
from numpy.polynomial import Polynomial, polynomial
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from timeit import timeit


def atmospheric_density_fit():
    # slant-depth from gaussian quadriture
    matplotlib.rcParams.update({"font.size": 16})

    N = int(1e5)
    alt_dec = np.linspace(0.0, 101.0, N)
    rs = rho(alt_dec)

    # polnos = []
    resids = []

    for i in range(30):
        popt, rlst = Polynomial.fit(alt_dec, rs, i, full=True)
        # polnos.append(popt)
        resids.append(rlst[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, squeeze=True)
    ax1.plot(resids, ".")
    ax2.plot(resids, ".")
    ax2.set_yscale("log")
    plt.show()

    print(Polynomial.fit(alt_dec, rs, 11, domain=[0, 100], full=True))


def theta_correction_fit():
    # slant-depth from gaussian quadriture
    matplotlib.rcParams.update({"font.size": 16})

    N = int(1e5)
    alt_dec = np.linspace(0.0, 101.0, N)
    rs = rho(alt_dec)

    # polnos = []
    resids = []

    for i in range(30):
        popt, rlst = Polynomial.fit(alt_dec, rs, i, full=True)
        # polnos.append(popt)
        resids.append(rlst[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, squeeze=True)
    ax1.plot(resids, ".")
    ax2.plot(resids, ".")
    ax2.set_yscale("log")
    plt.show()

    print(Polynomial.fit(alt_dec, rs, 11, domain=[0, 100], full=True))


def multi_poly_fit_slant():
    def length(z, t):
        t = np.asarray(t)

        i = 6371.0 ** 2 * np.cos(theta_tr) ** 2
        j = z ** 2
        k = 2 * z * 6371
        ijk = i + j + k

        return (z + 6371) / np.sqrt(ijk)

    N = int(1e4)

    alt_dec = np.random.uniform(0.0, 100.0, N)
    theta_tr = np.random.uniform(0.0, np.pi / 2, N)

    xy = np.vstack((alt_dec, theta_tr)).T
    y = length(alt_dec, theta_tr)

    x1 = np.random.uniform(0.0, 65, N)
    x2 = np.random.uniform(0.0, np.pi / 2, N)

    for i in range(1, 30):
        model = Pipeline(
            [("poly", PolynomialFeatures(degree=i)), ("linear", LinearRegression())]
        )
        model = model.fit(xy, y)
        p1 = model.predict(xy)

        z = length(x1, x2)
        x1x2 = np.vstack((x1, x2)).T
        p2 = model.predict(x1x2)

        print(i, mean_squared_error(y, p1), mean_squared_error(z, p2))

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=6)),
            ("linear", LinearRegression(fit_intercept=False)),
        ]
    )

    model = model.fit(xy, y)
    print(model.named_steps["linear"].coef_)

    lx = np.linspace(0, 65, 100)
    ly = np.linspace(0, np.pi / 2, 100)

    X, Y = np.meshgrid(lx, ly)
    print(X.shape, Y.shape)
    XY = np.stack((X.ravel(), Y.ravel()), axis=1)
    print(XY.shape)
    p1 = model.predict(XY)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(alt_dec, theta_tr, y, alpha=0.1)
    ax.plot_wireframe(
        X, Y, p1.reshape(100, 100), rcount=10, ccount=10, color="red", alpha=0.6
    )
    plt.show()


if __name__ == "__main__":
    pass
