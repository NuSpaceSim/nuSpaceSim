import cubepy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline, splev, splrep
from scipy.misc import derivative
from scipy.optimize import curve_fit
from scipy.special import kl_div

__all__ = ["differential_ozone", "ozone"]


def altitude_along_prop_axis(L, z_start, beta_tr, Re=6378.1):
    r1 = Re**2
    r2 = 2 * Re * z_start
    r3 = z_start**2
    return -Re + np.sqrt(
        L**2 + 2 * L * np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3) + r1 + r2 + r3
    )


def ozone(z):
    OzZeta = np.array(
        [5.35, 10.2, 14.75, 19.15, 23.55, 28.1, 32.8, 37.7, 42.85, 48.25, 100.0]
    )
    OzDepth = np.array(
        [15.0, 9.0, 10.0, 31.0, 71.0, 87.2, 57.0, 29.4, 10.9, 3.2, 1.3],
    )
    OzDsum = np.array(
        [310.0, 301.0, 291.0, 260.0, 189.0, 101.8, 44.8, 15.4, 4.5, 1.3, 0.1]
    )

    TotZon = np.where(z < 5.35, 310 + ((5.35 - z) / 5.35) * 15, 0.1)

    msk3 = np.logical_and(z >= 5.35, z < 100)
    i = np.searchsorted(OzZeta, z[msk3])

    TotZon[msk3] = (
        OzDsum[i] + ((OzZeta[i] - z[msk3]) / (OzZeta[i] - OzZeta[i - 1])) * OzDepth[i]
    )
    return TotZon


def differential_ozone(z):
    return -derivative(ozone, z)


def plot_hist_errors(
    Z, T, OD, OD_fit, title="Ozone Depth Approximaiton Error Histograms"
):

    rad_ticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
    rad_tick_labels = [
        "0",
        r"$\frac{\pi}{8}$",
        r"$\frac{\pi}{4}$",
        r"$\frac{3\pi}{8}$",
        r"$\frac{\pi}{2}$",
    ]

    print("Absolute Error", np.sum(np.abs(OD - OD_fit)))
    print("Relative Error", np.sum(np.abs(OD[OD > 0] - OD_fit[OD > 0]) / OD[OD > 0]))

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    fig.suptitle(title)

    ax = fig.add_subplot(231)
    ax.hist(np.abs(OD - OD_fit).ravel(), bins=100, log=True)
    ax.set_xlabel("Absolute Error |quadrature - approx|")

    ax = fig.add_subplot(232)
    ax.hist(
        (np.abs(OD[OD > 0] - OD_fit[OD > 0]) / OD[OD > 0]).ravel(), bins=100, log=True
    )
    ax.set_xlabel("Relative Error |quadrature - approx|/slant_depth")

    ax = fig.add_subplot(233)
    ax.hist(kl_div(OD, OD_fit).ravel(), bins=100, log=True)
    ax.set_xlabel("KL divergence")

    ax = fig.add_subplot(234, projection="3d")
    ax.plot_surface(Z, T, OD, color="b", alpha=0.6)
    ax.plot_wireframe(Z, T, OD_fit.reshape(Z.shape), rstride=5, cstride=5, color="r")
    ax.set_xlabel("Altitude")
    ax.set_ylabel("Theta")
    ax.set_yticks(rad_ticks)
    ax.set_yticklabels(rad_tick_labels)
    ax.set_zlabel("Ozone Depth")

    ax = fig.add_subplot(235, projection="3d")
    ax.plot_surface(Z, T, np.abs(OD - OD_fit), cmap="coolwarm")
    ax.set_xlabel("Altitude")
    ax.set_ylabel("Theta")
    ax.set_yticks(rad_ticks)
    ax.set_yticklabels(rad_tick_labels)
    ax.set_title("Absolute Error |quadrature - approx|")

    ax = fig.add_subplot(236)
    ax.contourf(Z, T, np.abs(OD - OD_fit), cmap="coolwarm")
    ax.set_xlabel("Altitude")
    ax.set_ylabel("Theta")
    ax.set_yticks(rad_ticks)
    ax.set_yticklabels(rad_tick_labels)
    ax.set_title("Absolute Error |quadrature - approx|")

    plt.show()


def spline_ozone(z):
    return splev(
        z,
        (
            # fmt: off
            np.array(
                [3.47776272e-26, 3.47776272e-26, 3.47776272e-26, 3.47776272e-26,
                 1.67769146e01, 2.42250765e01, 3.29917411e01, 4.08815284e01,
                 4.88486345e01, 5.10758252e01, 5.69937723e01, 6.34112652e01,
                 6.58453130e01, 9.02965953e01, 9.02965953e01, 9.02965953e01,
                 9.02965953e01]),
            np.array(
                [2.80373832e00, 3.41049110e00, -4.22593929e00, 2.57297359e01,
                 6.33106297e00, 1.72934400e00, 6.66859235e-02, 8.66981139e-03,
                 3.18321395e-02, 2.45436302e-02, 2.63104170e-02, 2.51036483e-02,
                 2.51207729e-02, 0.00000000e00, 0.00000000e00, 0.00000000e00,
                 0.00000000e00]),
            # fmt: on
            3,
        ),
    )


_spline_ozone = BSpline(
    # fmt: off
    np.array(
        [3.47776272e-26, 3.47776272e-26, 3.47776272e-26, 3.47776272e-26, 1.67769146e1,
         2.42250765e1, 3.29917411e1, 4.08815284e1, 4.88486345e1, 5.10758252e1,
         5.69937723e1, 6.34112652e1, 6.58453130e1, 9.02965953e1, 9.02965953e1,
         9.02965953e1, 9.02965953e1]),
    np.array(
        [2.80373832, 3.41049110, -4.22593929, 2.57297359e1, 6.33106297, 1.72934400,
         6.66859235e-2, 8.66981139e-3, 3.18321395e-2, 2.45436302e-2, 2.63104170e-2,
         2.51036483e-2, 2.51207729e-2, 0.0, 0.0, 0.0, 0.0]),
    # fmt: on
    3,
    # extrapolate=True,
)


def optimal_ozone_altitude_spline(N, zk, doz, plot=False):
    def f(x, *knots):
        xs = np.sort([*knots])
        ys = differential_ozone(xs)
        tck = splrep(xs, ys)
        val = splev(x, tck)
        return val

    zn = np.linspace(0.0, 100.0, N)

    popt, *_ = curve_fit(f, zk, doz, p0=zn, bounds=(0.0, 100.0))

    doz_fit = f(zk, *popt)

    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        fig.suptitle(f"Spline fit {len(popt)}: {popt}")
        ax = fig.add_subplot(111)
        ax.plot(zk, doz, "b")
        ax.plot(zk, doz_fit, "r--")
        ax.set_xlabel("Altitude")
        ax.set_ylabel("Ozone")

        plt.show()

    return doz_fit, popt


def do_spline_fit():

    M = int(1e5)
    zk = np.linspace(0.0, 100.0, M)
    doz = differential_ozone(zk)

    # for n in range(5, 22):
    #     a, b = optimal_ozone_altitude_spline(n, zk, doz, True)

    _, popt = optimal_ozone_altitude_spline(13, zk, doz, False)

    xs = np.sort([*popt])
    ys = differential_ozone(xs)
    tck = splrep(xs, ys)

    print(repr(tck))


def dL(z, t):
    r = 6378.1
    i = r**2 * np.cos(t) ** 2
    j = z**2
    k = 2.0 * r * z
    ijk = i + j + k
    return (r + z) / np.sqrt(ijk)


def quad_ozone_depth(x_lo, x_hi, y_lo, y_hi, points=20, plot=True):
    def help_dOD(z, idx, t):
        return _spline_ozone(z) * dL(z, t[idx])

    N = points
    z_hi = np.linspace(x_lo, x_hi, N)
    theta = np.linspace(y_lo, y_hi, N)

    Z, T = np.meshgrid(z_hi, theta)

    OD, err = cp.integrate(
        help_dOD,
        np.full_like(Z, x_lo).ravel(),
        Z.ravel(),
        is_1d=True,
        range_dim=1,
        evt_idx_arg=True,
        args=(T.ravel(),),
        parallel=True,
        tile_byte_limit=2**25,
        abstol=1e1,
        reltol=1e-1,
    )
    OD = OD.reshape(Z.shape)

    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)

        ax = fig.add_subplot(121, projection="3d")
        ax.plot_surface(Z, T, _spline_ozone(Z) * dL(Z, T), cmap="viridis")
        ax.set_title("Ozone Content Along Path")
        ax.set_xlabel("Altitude")
        ax.set_ylabel("Theta")
        ax.set_zlabel("Ozone Content")

        ax = fig.add_subplot(122, projection="3d")
        ax.plot_surface(Z, T, OD, cmap="viridis")
        ax.set_title("Ozone Depth Integral")
        ax.set_xlabel("Altitude")
        ax.set_ylabel("Theta")
        ax.set_zlabel("Ozone Depth")

        plt.show()

    return Z, T, OD, err


def ozone_depth_temp(x, p1, p2, p3, p4):
    z, t = x
    s = p4 * _spline_ozone.antiderivative()(z)
    v = (p1 + p2 * 1 / np.sqrt(np.cos(t))) * (p3 + s)
    return v.ravel()


def curve_fit_integral(Z, T, OD, plot=True):

    p0 = np.full(4, 0.9)

    popt, *_ = curve_fit(ozone_depth_temp, (Z.ravel(), T.ravel()), OD.ravel(), p0=p0)

    OD_fit = ozone_depth_temp((Z, T), *popt).reshape(OD.shape)

    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(Z, T, OD_fit, color="r", rstride=5, cstride=5, alpha=0.6)
        ax.plot_surface(Z, T, OD, rstride=1, cstride=1, alpha=0.6)
        ax.set_xlabel("Altitude")
        ax.set_ylabel("Theta")
        ax.set_zlabel("ozone depth")

        plt.show()

    return popt, OD_fit


def do_curve_fit():
    b_low, b_high = 5e-2, np.pi * 0.5
    t_low, t_high = 0.5 * np.pi - b_low, 0.5 * np.pi - b_high

    Z, T, OD, err = quad_ozone_depth(0.0, 100.0, t_low, t_high, points=100, plot=False)
    popt, OD_fit = curve_fit_integral(Z, T, OD, plot=False)

    print(repr(popt))
    print(_spline_ozone.antiderivative().tck)

    plot_hist_errors(Z, T, OD, OD_fit)


if __name__ == "__main__":
    np.set_printoptions(linewidth=256, precision=17)
    do_curve_fit()
