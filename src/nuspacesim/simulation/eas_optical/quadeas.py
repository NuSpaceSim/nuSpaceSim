import numpy as np

import quadpy as qp


def index_of_refraction_air(X_v):
    r"""Index of refraction in air (Nair)

    Index of refraction as a function of vertical atmospheric depth x_v (g/cm^2)

    Hillas 1475 eqn (2)
    """
    temperature = 204.0 + 0.091 * X_v
    n = 1.0 + 0.000296 * (X_v / 1032.9414) * (273.2 / temperature)
    return n


def rad_len_atm_depth(x, L0=36.66):
    """
    T is X / L0, units of radiation length scaled g/cm^2
    """
    T = x / L0
    return T


def shower_age(T, param_beta):
    r"""Shower age (s) as a function of atmospheric depth in mass units (g/cm^2)


    Hillas 1475 eqn (1)

    s = 3 * T / (T + 2 * beta)
    """
    return 3.0 * T / (T + 2.0 * param_beta)


def greisen_particle_count(T, s, y):
    r"""Particle count as a function of radiation length from atmospheric depth

    Hillas 1461 eqn (6)

    N_e(T)
    y is beta in EASCherGen
    """
    N_e = (0.31 / np.sqrt(y)) * np.exp(T * (1.0 - 1.5 * np.log(s)))
    return N_e


def e0(s):
    r"""Kinetic Energy charged primary of shower particles (MeV)"""
    E0 = np.where(s >= 0.4, 44.0 - 17.0 * (s - 1.46), 26.0)
    return E0


def fractional_track_length(E, s, e0=e0):
    r"""Fractional Track Length in radiation lengths.

    Hillas 1461 eqn (8) variable T(E) =
    (Total track length of all charged particles with kinetic energy > E)
    /
    (Total vertical component of tracks of all charged particles).

    total track length of particles if energy > E in vertical thickness interval dx of
    the shower is N_e*T(E)*dx.

    """
    E0 = e0(s)
    return ((0.89 * E0 - 1.2) / (E0 + E)) ** s * (1.0 + 1.0e-4 * s * E) ** -2


def shibata_grammage(z):
    z = np.asarray(z)
    # conditional cutoffs
    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25

    # 1 / 0.19
    r1 = np.reciprocal(0.19)

    # X_v vertical atmospheric depth g / cm^2
    X_v = np.empty_like(z)
    X_v[mask1] = np.power(((z[mask1] - 44.34) / -11.861), r1)
    X_v[mask2] = np.exp((z[mask2] - 45.5) / -6.34)
    X_v[mask3] = np.exp(13.841 - np.sqrt(28.920 + 3.344 * z[mask3]))

    return X_v


def shibata_density(z):
    z = np.asarray(z)
    # conditional cutoffs
    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25

    # 1 / 0.19
    r1 = np.reciprocal(0.19)

    # X_v = shibata_grammage(z)

    rho = np.empty_like(z)
    rho[mask1] = -1.0e-5 * r1 / -11.861 * ((z[mask1] - 44.34) / -11.861) ** (r1 - 1.0)
    # rho[mask2] = -1e-5 * np.reciprocal(-6.34) * X_v[mask2]
    rho[mask2] = (1e-5 / 6.34) * np.exp((z[mask2] - 45.5) / -6.34)
    # rho[mask3] = ((0.5e-5 * 3.344) / np.sqrt(28.920 + 3.344 * z[mask3])) * X_v[mask3]
    rho[mask3] = ((0.5e-5 * 3.344) / np.sqrt(28.920 + 3.344 * z[mask3])) * np.exp(
        13.841 - np.sqrt(28.920 + 3.344 * z[mask3])
    )

    return rho


def slant_depth_numeric(z, z_max=np.inf, beta=0, atm_density_model=shibata_density):

    earth_radius = 6378.14
    return qp.quad(
        lambda h: 1e5
        * atm_density_model(h)
        * (h + earth_radius)
        / np.sqrt(
            (earth_radius ** 2 * np.cos(beta) ** 2) + (h ** 2) + (2 * h * earth_radius)
        ),
        z,
        z_max,
    )


def slant_depth_analytic(z, z_max, beta):

    z = np.asarray(z)
    z_max = np.asarray(z_max)

    assert np.all(z_max >= z)

    # conditional cutoffs
    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25

    def sd_11(z):
        return (
            122.610146005993
            * z
            * z
            * (1 - 0.02255299954894 * z) ** 4.26315789473684
            * np.cos(beta)
        )

    def sd_25(z):
        return (
            206.392022662692
            * (-6.34 * z - 40.1956)
            * np.exp(-0.157728706624606 * z)
            * np.cos(beta)
        )

    def sd_oo(z):
        x = np.sqrt(0.115629322268326 * z + 1)
        return (
            -318938.577439959
            * (3.21634680855233 * z + 10.3448867928848 * x + 1.92365239745953)
            * np.exp(-5.3777318638995 * x)
            * np.cos(beta)
        )

    sd = np.zeros_like(z_max)
    sd += sd_11(np.minimum(z_max, 11.0)) - sd_11(np.minimum(z, 11.0))
    sd += sd_25(
        np.where(
            (z_max >= 11.0) & (z_max < 25.0), z_max, np.where(z_max < 11.0, 11.0, 25.0)
        )
    ) - sd_25(np.where(mask2, z, np.where(mask1, 11.0, 25.0)))
    sd += sd_oo(np.maximum(z_max, 25.0)) - sd_oo(np.where(mask3, z, 25.0))

    return sd


def theta_view(beta, z_max, earth_radius):
    return np.arcsin((earth_radius / (earth_radius + z_max)) * np.cos(beta))


def sin_theta_view(beta, z_max, earth_radius):
    return np.sin(theta_view(beta, z_max, earth_radius))


def theta_prop(z, sinThetView, z_max, earth_radius):
    return np.arccos(sinThetView * ((earth_radius + z_max) / (earth_radius + z)))


def photon_yeild(thetaC, wavelength):
    return (2e9 * np.pi * np.sin(thetaC) ** 2) / (137.04 * wavelength)


def sokolsky_rayleigh_scatter(dX, wavelength):
    return np.exp(-dX / 2974.0 * (400.0 * wavelength) ** 4)


def elterman_mie_aerosol_scatter(
    z, z_max, beta, wavelength, aOD55, dfaOD55, inv_beta_poly, earth_radius
):

    aBeta = np.reciprocal(inv_beta_poly(wavelength)) / 0.158

    def aODepth(z):
        return aOD55[np.int32(z)] - (z - np.int32(z)) * dfaOD55[np.int32(z)] * aBeta

    def tprp(z):
        return theta_prop(
            z, sin_theta_view(beta, z_max, earth_radius), z_max, earth_radius
        )

    aTrans = np.where(z < 30, np.exp(-aODepth(z) / np.pi / (2 - tprp(z))), 1.0)
    return aTrans


def ozone_losses(z, wavelength):
    """
    Calculate ozone losses from altitudes (z) in km.

    ############################
    Implementation needs review.
    ############################

    """
    # OzZeta = np.array(
    #     [5.35, 10.2, 14.75, 19.15, 23.55, 28.1, 32.8, 37.7, 42.85, 48.25, 100.0]
    # )
    # OzDepth = np.array(
    #     [15.0, 9.0, 10.0, 31.0, 71.0, 87.2, 57.0, 29.4, 10.9, 3.2, 1.3],
    # )
    # OzDsum = np.array(
    #     [310.0, 301.0, 291.0, 260.0, 189.0, 101.8, 44.8, 15.4, 4.5, 1.3, 0.1]
    # )

    # TotZon = np.where(z < 5.35, 310 + ((5.35 - z) / 5.35) * 15, 0.1)

    # msk3 = np.logical_and(z >= 35, z < 100)
    # i = np.searchsorted(OzZeta, z[msk3])
    # Okappa = -1e03 * 10 ** (110.5 - 44.21 * np.log10(wavelength))

    # TotZon[msk3] = (
    #     OzDsum[i] + ((OzZeta[i] - z[msk3]) / (OzZeta[i] - OzZeta[i - 1])) * OzDepth[i]
    # )
    # np.exp(TotZon * Okappa)
    return 1.0


def scaled_photon_yeild(z, wavelength):
    pass

    # X = slant_depth_analytic(z, 65)
    # T = rad_len_atm_depth(X)
    # s = shower_age(T, 1)

    # pyield = photon_yeild(1, wavelength)
    # rayleigh_trans = 1.0  # sokolsky_rayleigh_scatter(dx, wavelength)
    # ozone_trans = 1.0  # ozone_losses(z, wavelength)
    # mie_aerosol_trans = (
    #     1.0  # elterman_mie_aerosol_scatter(z, 1, 1, wavelength, 1, 1, 1, 1)
    # )
    # RN = greisen_particle_count(T, s, 1)
    # return pyield * rayleigh_trans * ozone_trans * mie_aerosol_trans * RN


def dndu(theta, logenergy, s):

    energy = 10 ** logenergy

    whill = 2.0 * (1.0 - np.cos(theta)) * (energy / 21.0) ** 2
    # print("whill", whill)
    e2hill = 1150.0 + 454 * np.log(s)

    vhill = np.where(e2hill > 0.0, energy / e2hill, 0.0)
    # print("vhill", vhill)
    w_ave = 0.0054 * energy * (1.0 + vhill) / (1.0 + 13.0 * vhill + 8.3 * vhill ** 2)
    # print("w_ave", w_ave)
    uhill = whill / w_ave
    # print("uhill", uhill)

    zmz0hill = -np.sqrt(uhill) - 0.59
    # print("zmz0hill", zmz0hill)
    lambdai = np.where(zmz0hill < 0.0, 0.478, 0.380)
    # print("lambdai", lambdai)
    sv2 = 0.777 * np.exp(zmz0hill / lambdai)
    # print("sv2", sv2)
    return sv2


# from nuspacesim.eas_optical.atmospheric_models import slant_depth


def intf(x_):
    z = x_[0]
    # w = x_[1]
    o = x_[1]
    e = x_[2]

    print("z", z)
    X = np.empty_like(z)
    for i in range(z.size):
        X[i] = slant_depth_numeric(1.0, z[i], beta=np.radians(42.0))[0]
    print("X", X)

    T = rad_len_atm_depth(X)
    s = shower_age(T, np.log(10 ** 8 / (0.710 / 8.36)))
    # print("s", s)
    # print("o", o)
    # print("e", e)
    # v1= scaled_photon_yeild(z, w)
    v1 = 1
    v2 = dndu(o, e, s)
    return v1 * v2


if __name__ == "__main__":
    import nuspacesim as nss

    nss.eas_optical.atmospheric_models.slant_depth(1.0, 65.0, np.radians(10))

    import timeit

    print(slant_depth_numeric(1.0, 5.0, np.radians(10)))
    earth_radius = 6378.14

    def f(h):
        return (
            1e5
            * shibata_density(h)
            * (h + earth_radius)
            / np.sqrt(
                (earth_radius ** 2 * np.cos(np.radians(10)) ** 2)
                + (h ** 2)
                + (2 * h * earth_radius)
            )
        )

    print(
        timeit.timeit(
            lambda: qp.quad(f, 1.0, 5.0, epsabs=1e-2, epsrel=1e-2), number=10000
        )
    )
    print(qp.quad(f, 1.0, 5.0, epsabs=1e-2, epsrel=1e-2))
    # slant_depth_analytic(1., 65., np.radians(10))

    # print(qp.quad(intf, [1,2.857e-4,1], [65,1.657e-2,10], epsabs=1e-7))

# return qp.quad( lambda x: , )
