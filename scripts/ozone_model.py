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
    r1 = Re ** 2
    r2 = 2 * Re * z_start
    r3 = z_start ** 2
    return -Re + np.sqrt(
        L ** 2 + 2 * L * np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3) + r1 + r2 + r3
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


def poly_differential_ozone(z):
    # fmt: off
    _polydifoz = Polynomial(
        [1.58730302e-1, -2.28377732e00, 3.88514872e00, -1.00201924e02, 1.25384358e03,
         -6.07444561e01, -2.55521209e04, 9.95782609e03, 2.89535348e05, -1.68713471e05,
         -1.88322268e06, 1.28137760e06, 7.56064015e06, -5.30174814e06, -1.99411195e07,
         1.33214081e07, 3.58159190e07, -2.13788677e07, -4.43222183e07, 2.21324708e07,
         3.73327083e07, -1.43479612e07, -2.04932327e07, 5.30961778e06, 6.62152119e06,
         -8.57379855e05, -9.56235193e05, ],
        domain=[0.0, 100.0],
    )
    # fmt: on
    return _polydifoz(z)


ozone_spline = BSpline(
    # fmt: off
    np.array(
        [0.0, 0.0, 0.0, 0.0, 6.89655172, 10.34482759, 13.79310345, 17.24137931,
         20.68965517, 24.13793103, 27.5862069, 31.03448276, 34.48275862, 37.93103448,
         41.37931034, 44.82758621, 48.27586207, 51.72413793, 55.17241379, 58.62068966,
         62.06896552, 65.51724138, 68.96551724, 72.4137931, 75.86206897, 79.31034483,
         82.75862069, 86.20689655, 89.65517241, 93.10344828, 100.0, 100.0, 100.0,
         100.0, ]
    ),
    np.array(
        [3.25000000e02, 3.18177024e02, 3.08908644e02, 3.00483582e02, 2.95353477e02,
         2.76720925e02, 2.38445425e02, 1.80430762e02, 1.06225920e02, 6.45461187e01,
         3.28591575e01, 1.22379408e01, 7.65516571e00, 2.81772715e00, 1.04250806e00,
         1.40834257e00, 1.20048349e00, 1.14634515e00, 1.05101746e00, 9.66726427e-01,
         8.79478129e-01, 7.93022227e-01, 7.06354003e-01, 6.19742671e-01,
         5.33116094e-01, 4.46493602e-01, 3.59870016e-01, 2.44372291e-01,
         1.57748887e-01, 1.00000000e-01, 0.00000000e00, 0.00000000e00, 0.00000000e00,
         0.00000000e00, ]
    ),
    3,
    # fmt: on
)


def spline_differential_ozone(z):
    return -ozone_spline.derivative(1)(z)


def ozone_content(L_n, Lmax, alt_dec, beta_tr):
    def f(x):
        return spline_differential_ozone(altitude_along_prop_axis(x, alt_dec, beta_tr))

    return cp.integrate(
        f, L_n, Lmax, is_1d=True, parallel=True, tile_byte_limit=2 ** 25
    )


def ozone_losses(ZonZ, wavelength):
    """
    Calculate ozone losses from points along shower axis (l) in km.

    ############################
    Implementation needs review.
    ############################

    """

    Okappa = 10 ** (110.5 - 44.21 * np.log10(wavelength))
    return np.exp(-1e-3 * np.multiply(ZonZ, Okappa))


def atmospheric_ozone_fit(_):
    # slant-depth from gaussian quadriture
    matplotlib.rcParams.update({"font.size": 16})

    N = int(22)
    M = int(1e5)
    x = np.linspace(0.0, 100.0, N)
    z = np.linspace(0.0, 100.0, M)
    xoz = ozone(x)
    oz = ozone(z)
    doz = differential_ozone(z)

    spl = splrep(x, xoz)
    soz = splev(z, spl)

    fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True, squeeze=True)

    ax1[0].plot(z, oz, "b-")
    ax1[0].plot(z, soz, "r--")
    ax1[0].set_xlabel("Altitude (KM)")
    ax1[0].set_ylabel("Ozone Depth")
    ax1[0].set_title("Ozone Depth")
    ax1[0].legend(["Nimbus", "Spline"])
    ax1[0].grid(True)

    ax2[0].plot(z, oz, "b-")
    ax2[0].plot(z, soz, "r--")
    ax2[0].set_yscale("log")
    ax2[0].set_xlabel("Altitude (KM)")
    ax2[0].set_ylabel("Log(Ozone Depth)")
    ax2[0].grid(True)

    dsoz = -splev(z, spl, der=1)

    ax1[1].plot(z, doz, "b-")
    ax1[1].plot(z, dsoz, "r--")
    ax1[1].set_xlabel("Altitude (KM)")
    ax1[1].set_ylabel("d(Ozone Depth)/d altitude")
    ax1[1].set_title("negative first derivative")
    ax1[1].legend(["Nimbus", "Spline"])
    ax1[1].grid(True)

    ax2[1].plot(z, doz, "b")
    ax2[1].plot(z, dsoz, "r--")
    ax2[1].set_yscale("log")
    ax2[1].set_xlabel("Altitude (KM)")
    ax2[1].set_ylabel("log(d(Ozone Depth)/d altitude)")
    ax2[1].grid(True)

    plt.show()


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

    # ax = fig.add_subplot(324)
    # ax.scatter((np.abs(OD[OD > 0] - OD_fit[OD > 0]) / OD[OD > 0]).ravel())
    # ax.set_xlabel("Relative Error |quadrature - approx|/slant_depth")

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
    i = r ** 2 * np.cos(t) ** 2
    j = z ** 2
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
        tile_byte_limit=2 ** 25,
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
