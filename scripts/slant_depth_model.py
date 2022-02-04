from inspect import signature

import cubepy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import bisplev, bisplrep, splev, splrep

# from scipy.misc import derivative
from scipy.optimize import curve_fit
from scipy.special import kl_div


def us_std_atm_density(z, earth_radius=6378.1):
    H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852])
    Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
    T_b = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
    # fmt: off
    P_b = 1.01325e5 * np.array(
        [1.0, 2.233611e-1, 5.403295e-2, 8.5666784e-3, 1.0945601e-3, 6.6063531e-4,
         3.9046834e-5, 3.68501e-6, ])
    # fmt: on

    Rstar = 8.31432e3
    M0 = 28.9644
    gmr = 34.163195

    z = np.asarray(z)

    h = z * earth_radius / (z + earth_radius)
    i = np.searchsorted(H_b, h, side="right") - 1  # <--!!

    deltah = h - H_b[i]

    temperature = T_b[i] + Lm_b[i] * deltah

    mask = Lm_b[i] == 0
    pressure = np.full(z.shape, P_b[i])
    pressure[mask] *= np.exp(-gmr * deltah[mask] / T_b[i][mask])  # <--!!
    pressure[~mask] *= (T_b[i][~mask] / temperature[~mask]) ** (gmr / Lm_b[i][~mask])

    density = (pressure * M0) / (Rstar * temperature)  # kg/m^3
    return 1e-3 * density  # g/cm^3


rho_spline = (
    np.array(
        [
            # fmt:off
            1.2722928345804028, 1.2722928345804028, 1.2722928345804028, 1.2722928345804028,
            17.551584821402038, 26.693912708416367, 42.07417928917581, 94.8876226581273,
            94.8876226581273, 94.8876226581273, 94.8876226581273
            # fmt:on
        ]
    ),
    np.array(
        [
            # fmt:off
            0.00146630263741946, 0.00725777601552678, 0.00967697934455952, 0.01029578198358256,
            0.01037915940543929, 0.01034895283423567, 0.01035607695460405, 0.0, 0.0, 0.0, 0.0,
            # fmt:on
        ]
    ),
    3,
)


def slant_depth_integrand(z, theta_tr, earth_radius, rho=us_std_atm_density):

    i = earth_radius ** 2 * np.cos(theta_tr) ** 2
    j = z ** 2
    k = 2 * z * earth_radius

    ijk = i + j + k

    rval = 1e5 * rho(z) * ((z + earth_radius) / np.sqrt(ijk))
    return rval


def slant_depth(z_lo, z_hi, theta_tr, earth_radius=6378.1, abstol=1e-6, reltol=1e-6):
    """
    Slant-depth in g/cm^2

    Parameters
    ----------
    z_lo : float
        Starting altitude for slant depth track.
    z_hi : float
        Stopping altitude for slant depth track.
    theta_tr: float, array_like
        Trajectory angle in radians between the track and earth zenith.

    """

    theta_tr = np.asarray(theta_tr)

    def helper(z, evt_idx, theta_tr, earth_radius):
        return slant_depth_integrand(z, theta_tr[evt_idx], earth_radius)

    return cp.integrate(
        helper,
        z_lo,
        z_hi,
        args=(theta_tr, earth_radius),
        is_1d=True,
        evt_idx_arg=True,
        abstol=abstol,
        reltol=reltol,
        tile_byte_limit=2 ** 25,
        parallel=False,
    )


def plot_hist_errors(
    Z, T, OD, OD_fit, title="Slant Depth Approximaiton Error Histograms"
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

    # ax = fig.add_subplot(233)
    # ax.hist(kl_div(OD, OD_fit).ravel(), bins=100, log=True)
    # ax.set_xlabel("KL divergence")

    ax = fig.add_subplot(234, projection="3d")
    ax.plot_surface(Z, T, OD, color="b", alpha=0.6)
    ax.plot_wireframe(Z, T, OD_fit.reshape(Z.shape), rstride=5, cstride=5, color="r")
    ax.set_xlabel("Altitude")
    ax.set_ylabel("Theta")
    ax.set_yticks(rad_ticks)
    ax.set_yticklabels(rad_tick_labels)
    ax.set_zlabel("Slant Depth")

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


def quad_slant_depth(x_lo, x_hi, y_lo, y_hi, points=20, plot=False):

    N = points
    z_hi = np.linspace(x_lo, x_hi, N)
    theta = np.linspace(y_lo, y_hi, N)

    Z, T = np.meshgrid(z_hi, theta)

    SD, err = slant_depth(
        np.full_like(Z, x_lo).ravel(), Z.ravel(), T.ravel(), abstol=1e-8, reltol=1e-8
    )
    SD = SD.reshape(Z.shape)

    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)

        ax = fig.add_subplot(121, projection="3d")
        ax.plot_surface(Z, T, slant_depth_integrand(Z, T, 6378.1), cmap="viridis")
        ax.set_title("Grammage Along Path")
        ax.set_xlabel("Altitude")
        ax.set_ylabel("Theta")
        ax.set_zlabel("Grammage")

        ax = fig.add_subplot(122, projection="3d")
        ax.plot_surface(Z, T, SD, cmap="viridis")
        ax.set_title("Slant Depth Integral")
        ax.set_xlabel("Altitude")
        ax.set_ylabel("Theta")
        ax.set_zlabel("Slant Depth")

        plt.show()

    return Z, T, SD, err


def slant_depth_temp(x, p1, p2, p4, p5, p6):
    z, t = x
    eps = 1e-5
    # rho0 = us_std_atm_density(0)
    rhoz = p4 * us_std_atm_density(z)
    # intrhoz = p4 * 1e5 * splev(z, rho_spline)

    st = np.sin(0.5 * t)
    ct = np.cos(0.5 * t)
    lv = np.log(st + ct + eps) - np.log(ct - st + eps)
    v = p1 + p2 * lv ** p5
    # v *= p3 + intrhoz
    v *= p6 + rhoz

    return v.ravel()


def slant_depth_curve_fit_integral(Z, T, OD, plot=True):

    sig = signature(slant_depth_temp)
    p0 = np.full(len(sig.parameters) - 1, 1.1)

    popt, *_ = curve_fit(slant_depth_temp, (Z.ravel(), T.ravel()), OD.ravel(), p0=p0)

    OD_fit = slant_depth_temp((Z, T), *popt).reshape(OD.shape)

    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(Z, T, OD_fit, color="r", rstride=5, cstride=5, alpha=0.6)
        ax.plot_surface(Z, T, OD, rstride=1, cstride=1, alpha=0.6)
        ax.set_xlabel("Altitude")
        ax.set_ylabel("Theta")
        ax.set_zlabel("Slant Depth")

        plt.show()

    return popt, OD_fit


def do_slant_depth_curve_fit():
    b_low, b_high = 5e-3, np.pi * 0.5
    t_low, t_high = 0.5 * np.pi - b_high, 0.5 * np.pi - b_low

    Z, T, SD, err = quad_slant_depth(0.0, 100.0, t_low, t_high, points=100, plot=False)
    # print(err)
    popt, SD_fit = slant_depth_curve_fit_integral(Z, T, SD, plot=True)
    # print(repr(popt))
    # plot_hist_errors(Z, T, OD, OD_fit)


def dL(z, t):
    r = 6378.1
    i = r ** 2 * np.cos(t) ** 2
    j = z ** 2
    k = 2.0 * r * z
    ijk = i + j + k
    return (r + z) / np.sqrt(ijk)


def rcos(t):
    return 1 / np.sqrt(np.cos(t) ** 2)


def quad_rcos(x_lo, x_hi, points=20, plot=True):
    N = points
    theta = np.linspace(x_lo, x_hi, N)
    val, err, *_ = cp.integrate(
        rcos, np.zeros(N), theta, is_1d=True, parallel=True, tile_byte_limit=2 ** 25
    )
    val = val + 1
    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(theta, val)
        ax.set_title("integral 1/cos ")
        ax.set_xlabel("Theta")
        ax.set_ylabel("integral")
        plt.show()

    return theta, val, err


def rcos_curve_fit(T, val, plot=True):

    # coefs = [np.polynomial.polynomial.polyfit(T, val, d) for d in range(3, 19)]
    # val_fits = [np.polynomial.polynomial.Polynomial(c)(T) for c in coefs]

    # if plot:
    #     fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    #     for i, val_fit in enumerate(val_fits):
    #         print(val_fit.shape)
    #         ax = fig.add_subplot(4, 4, i + 1)
    #         ax.plot(T, np.abs(val - val_fit), "r", alpha=0.5)
    #         ax.plot(T, val)
    #         ax.plot(T, val_fit, "--", alpha=0.8)
    #         ax.set_ylabel("Theta")
    #         ax.set_xlabel("integral")

    #     plt.show()

    coefs = np.polynomial.polynomial.polyfit(T, val, 11)
    val_fit = np.polynomial.polynomial.Polynomial(coefs)(T)

    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111)
        # ax.plot(T, np.abs(val - val_fit), "r", alpha=0.5)
        ax.plot(T, val)
        ax.plot(T, val_fit)
        # ax.plot(T, 1 / np.cos(T))
        # ax.plot(T, 1 / np.cosh(T))
        ax.set_ylabel("Theta")
        ax.set_xlabel("integral")

        plt.show()


def do_rcos_fit():
    b_low, b_high = 0, np.pi * 0.5
    t_low, t_high = 0.5 * np.pi - b_high, 0.5 * np.pi - b_low
    T, val, err = quad_rcos(t_low, t_high, points=int(1e6), plot=True)
    rcos_curve_fit(T, val, plot=True)
    # print(repr(popt))
    # plot_hist_errors(Z, T, OD, OD_fit)


def quad_rho(z_lo, z_hi, points=None, plot=False):

    N = points if np.isscalar(z_hi) else z_hi.size
    Zhi = np.linspace(z_lo, z_hi, N) if np.isscalar(z_hi) else z_hi.ravel()
    Zlo = np.zeros(N) if np.isscalar(z_lo) else z_lo.ravel()

    val, err, *_ = cp.integrate(
        us_std_atm_density,
        Zlo,
        Zhi,
        is_1d=True,
        parallel=True,
        tile_byte_limit=2 ** 25,
        abstol=1e-8,
        reltol=1e-8,
    )

    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(Zhi, val)
        ax.set_title("integral rho")
        ax.set_xlabel("Altitude")
        ax.set_ylabel("integral")
        plt.show()

    return Zhi, val, err


def optimal_spline(f, N, x, y):
    def g(x, *knots):
        xs = np.sort([*knots]).ravel()
        tck = splrep(xs, f(xs))
        val = splev(x, tck)
        return val

    xn = np.linspace(np.amin(x), np.amax(x), N)

    popt, *_ = curve_fit(g, x, y, p0=xn, bounds=(0.0, 100.0))
    popt = np.sort(popt)

    # y_fit = g(x, *popt)

    tck = splrep(popt, f(popt))

    return tck, popt, splev(x, tck)


def rho_curve_fit(T, val, plot=True):
    def f(z):
        _, val, *_ = quad_rho(np.zeros_like(z.ravel()), z.ravel())
        return val

    tf = [optimal_spline(f, k, T, val) for k in range(4, 10)]

    if plot:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        for i, (_, pts, fit) in enumerate(tf):
            ax = fig.add_subplot(2, 3, 1 + i)
            ax.plot(T, val)
            ax.plot(T, fit, "g--", alpha=0.8)
            ax.plot(T, np.abs(val - fit), "r", alpha=0.5)
            ax.vlines(pts, 0, 1.1e-2, linestyles="dotted")
            ax.set_ylabel("Altitude")
            ax.set_xlabel("integral")
            ax.text(50, 5e-3, f"{np.linalg.norm(val-fit):.4E}")

        plt.show()

    tck, pts, fit = optimal_spline(f, 7, T, val)

    return tck, pts, fit


def do_rho_fit():
    T, val, _ = quad_rho(0, 100, points=int(1e5), plot=True)
    tck, pts, _ = rho_curve_fit(T, val, plot=True)
    print("tck", repr(tck))
    print("pts", repr(pts))
    # plot_hist_errors(Z, T, OD, OD_fit)


def optimal_bspline(f, N, x, y, z):
    def g(v, *knots):

        xys = np.array([*knots])
        length = xys.size // 2

        xs = np.sort(xys[:length]).ravel()
        ys = np.sort(xys[length:]).ravel()
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.ravel()
        ys = ys.ravel()

        val = f(xs, ys)

        tck = bisplrep(xs, ys, val)
        val = bisplev(v[0], v[1], tck)

        return val.ravel()

    xn = np.linspace(np.amin(x), np.amax(x), N)
    yn = np.linspace(np.amin(y), np.amax(y), N)
    p0 = np.concatenate((xn.ravel(), yn.ravel()))

    popt, *_ = curve_fit(g, (x[0], y[:, 0]), z.ravel(), p0=p0, bounds=(0.0, 100.0))
    length = popt.size // 2
    xs = np.sort(popt[:length]).ravel()
    ys = np.sort(popt[length:]).ravel()
    xs, ys = np.meshgrid(xs, ys)

    tck = bisplrep(xs.ravel(), ys.ravel(), f(xs, ys).ravel())
    val = bisplev(x[0], y[:, 0], tck)
    return tck, popt, val


def do_slant_depth_spline_fit():
    N = int(20)

    b_low, b_high = 5e-3, np.pi * 0.5
    t_low, t_high = 0.5 * np.pi - b_high, 0.5 * np.pi - b_low

    z_hi = np.linspace(0.0, 100.0, N)
    theta = np.linspace(t_low, t_high, N)

    Z, T = np.meshgrid(z_hi, theta)

    def f_sd(z, t):

        SD, *_ = slant_depth(
            np.zeros_like(z).ravel(),
            z.ravel(),
            t.ravel(),
            abstol=1e-8,
            reltol=1e-8,
        )
        return SD

    SD = f_sd(Z, T)
    tck, pts, z_fit = optimal_bspline(f_sd, 8, Z, T, SD)
    SD = SD.reshape(Z.shape)
    z_fit = z_fit.reshape(Z.shape)
    # Z, T, SD, err = quad_slant_depth(0.0, 100.0, t_low, t_high, points=100, plot=False)
    # popt, SD_fit = slant_depth_curve_fit_integral(Z, T, SD, plot=True)
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Z, T, SD, color="b", alpha=0.6)
    ax.plot_wireframe(Z, T, z_fit, rstride=5, cstride=5, color="r")
    ax.plot_surface(Z, T, np.abs(SD - z_fit), alpha=0.2)
    ax.set_xlabel("Altitude")
    ax.set_ylabel("Theta")
    ax.set_zlabel("Slant Depth")
    # ax.text(50, 5e-3, f"{np.linalg.norm(SD-z_fit):.4E}")

    plt.show()


if __name__ == "__main__":
    np.set_printoptions(linewidth=256, precision=17)
    do_slant_depth_curve_fit()
    # do_rho_fit()
    # do_slant_depth_spline_fit()
