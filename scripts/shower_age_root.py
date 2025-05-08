from timeit import timeit

import numpy as np
from scipy.optimize import newton

from nuspacesim.simulation.eas_optical.atmospheric_models import *
from nuspacesim.simulation.eas_optical.quadeas import *


def shower_age_newton():
    earth_radius = 6371.036063815867
    alt_dec = 1.0
    # z_obs = 525.0
    z_max = 65.0
    beta_tr = np.radians(1)
    theta_tr = (np.pi / 2) - beta_tr
    param_beta = np.log(10**8 / (0.710 / 8.36))

    np.set_printoptions(linewidth=256)
    Lmax = length_along_prop_axis(alt_dec, z_max, beta_tr, earth_radius)
    print("Lmax", Lmax)
    ll = np.linspace(0, 8, 5)
    print("ll", ll)
    z = altitude_along_prop_axis(ll, alt_dec, beta_tr, earth_radius)
    print("z", z)

    X = slant_depth(alt_dec, z, theta_tr, epsrel=1e-3)[0]
    print("X", X)

    T = rad_len_atm_depth(X)
    s = shower_age(T, param_beta)
    print("s", s)

    def X_s(s):
        """Slant depth as a function of shower age."""
        return -1.222e19 * param_beta * s / ((10 / 6) * 1e17 * s - 5e17)

    print("X_s", X_s(0.2), X_s(0.4), X_s(1.0))

    def ff(_x, alt_dec, theta_tr, s):
        return slant_depth(alt_dec, _x, theta_tr, epsrel=1e-3)[0] - X_s(s)

    def df(_x, alt_dec, theta_tr, s):
        return slant_depth_integrand(_x, theta_tr)

    print("ff", ff(z, alt_dec, theta_tr, 0.2))

    z_s02 = newton(
        ff, alt_dec, fprime=df, args=(alt_dec, theta_tr, 0.2), full_output=True
    )
    z_s04 = newton(
        ff, alt_dec, fprime=df, args=(alt_dec, theta_tr, 0.4), full_output=True
    )
    z_s1 = newton(
        ff, alt_dec, fprime=df, args=(alt_dec, theta_tr, 1.0), full_output=True
    )

    print(z_s02)
    print(z_s04)
    print(z_s1)
    xs = slant_depth(alt_dec, np.array([z_s02[0], z_s04[0], z_s1[0]]), theta_tr)[0]
    print(shower_age(rad_len_atm_depth(xs), param_beta))

    def altitude_at_shower_age(s, alt_dec, beta_tr, param_beta=param_beta, **kwargs):
        """Altitude as a function of shower age, decay altitude and emergence angle."""

        theta_tr = (np.pi / 2) - beta_tr

        def X_s(s):
            """Slant depth as a function of shower age."""
            return -1.222e19 * param_beta * s / ((10 / 6) * 1e17 * s - 5e17)

        def ff(z):
            return slant_depth(alt_dec, z, theta_tr, epsrel=1e-3)[0] - X_s(s)

        def df(z):
            return slant_depth_integrand(z, theta_tr)

        return newton(ff, alt_dec, fprime=df, **kwargs)

    N = int(1e6)
    ads = np.linspace(0.5, 1.5, N)
    bts = np.linspace(1, 89, N)

    altitude_at_shower_age(0.2, ads, bts)
    altitude_at_shower_age(0.4, ads, bts)
    altitude_at_shower_age(1.0, ads, bts)


def shower_age_of_greisen_particle_count(target_count, x0=2):
    param_beta = np.log(10**8 / (0.710 / 8.36))

    def rns(s):
        return (
            0.31
            * np.exp((2 * param_beta * s * (1.5 * np.log(s) - 1)) / (s - 3))
            / np.sqrt(param_beta)
            - target_count
        )

    return newton(rns, x0)


if __name__ == "__main__":
    print(shower_age_newton())
