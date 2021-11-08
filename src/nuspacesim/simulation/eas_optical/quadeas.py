from functools import cache

import numpy as np
from numpy.polynomial import Polynomial

import quadpy as qp

from .atmospheric_models import slant_depth, slant_depth_integrand
from scipy.misc import derivative
from scipy.interpolate import BSpline
from scipy.optimize import newton


def viewing_angle(beta_tr, Zdet, Re):
    return np.arcsin((Re / (Re + Zdet)) * np.cos(beta_tr))


def propagation_angle(beta_tr, z, Re):
    return np.arccos((Re / (Re + z)) * np.cos(beta_tr))


def propagation_theta(beta_tr, z, Re):
    return propagation_angle(beta_tr, z, Re)


def length_along_prop_axis(z_start, z_stop, beta_tr, Re):
    L1 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_stop + z_stop ** 2
    L2 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_start + z_start ** 2
    L = np.sqrt(L1) - np.sqrt(L2)
    return L


def deriv_length_along_prop_axis(z_stop, beta_tr, Re):
    L1 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_stop + z_stop ** 2
    L = (Re + z_stop) / np.sqrt(L1)
    return L


def altitude_along_prop_axis(L, z_start, beta_tr, Re):
    r1 = Re ** 2
    r2 = 2 * Re * z_start
    r3 = z_start ** 2
    return -Re + np.sqrt(
        L ** 2 + 2 * L * np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3) + r1 + r2 + r3
    )


def deriv_altitude_along_prop_axis(L, z_start, beta_tr, Re):
    r1 = Re ** 2
    r2 = 2 * Re * z_start
    r3 = z_start ** 2
    r4 = np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3)
    denom = np.sqrt(L ** 2 + 2 * L * r4 + r1 + r2 + r3)
    numer = (Re + z_start) * ((L) / r4 + 1)
    return numer / denom


def gain_in_altitude_along_prop_axis(L, z_start, beta_tr, Re):
    return altitude_along_prop_axis(L, z_start, beta_tr, Re) - z_start


def distance_to_detector(beta_tr, z, z_det, earth_radius):
    theta_view = viewing_angle(beta_tr, z_det, earth_radius)
    theta_prop = propagation_angle(beta_tr, z, earth_radius)
    ang_e = np.pi / 2 - theta_view - theta_prop
    return np.sin(ang_e) / np.sin(theta_view) * (z + earth_radius)


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


def shower_age(T, param_beta=np.log(10 ** 8 / (0.710 / 8.36))):
    r"""Shower age (s) as a function of atmospheric depth in mass units (g/cm^2)


    Hillas 1475 eqn (1)

    s = 3 * T / (T + 2 * beta)
    """
    return 3.0 * T / (T + 2.0 * param_beta)


def greisen_particle_count(T, s, param_beta):
    r"""Particle count as a function of radiation length from atmospheric depth

    Hillas 1461 eqn (6)

    N_e(T) where y is beta in EASCherGen
    """
    N_e = (0.31 / np.sqrt(param_beta)) * np.exp(T * (1.0 - 1.5 * np.log(s)))
    return N_e


def shower_age_of_greisen_particle_count(target_count, x0=2):

    # for target_count = 2, shower_age = 1.899901462640018

    param_beta = np.log(10 ** 8 / (0.710 / 8.36))

    def rns(s):
        return (
            0.31
            * np.exp((2 * param_beta * s * (1.5 * np.log(s) - 1)) / (s - 3))
            / np.sqrt(param_beta)
            - target_count
        )

    return newton(rns, x0)


def altitude_at_shower_age(s, alt_dec, beta_tr, z_max=65.0, **kwargs):
    """Altitude as a function of shower age, decay altitude and emergence angle."""

    alt_dec = np.asarray(alt_dec)
    beta_tr = np.asarray(beta_tr)

    theta_tr = (np.pi / 2) - beta_tr
    param_beta = np.log(10 ** 8 / (0.710 / 8.36))

    # Check that shower age is within bounds
    X = slant_depth(alt_dec, z_max, theta_tr, epsrel=1e-2)[0]
    ss = shower_age(rad_len_atm_depth(X))

    mask = ss < s

    X_s = -1.222e19 * param_beta * s / ((10 / 6) * 1e17 * s - 5e17)

    def ff(z):
        return slant_depth(alt_dec, z, theta_tr, epsrel=1e-2)[0] - X_s

    def df(z):
        return slant_depth_integrand(z, theta_tr)

    altitude = np.full_like(alt_dec, z_max)
    altitude[~mask] = newton(ff, alt_dec, fprime=df, **kwargs)

    return altitude


# def e0(s):
#     r"""Kinetic Energy charged primary of shower particles (MeV)"""
#     E0 = np.where(s >= 0.4, 44.0 - 17.0 * (s - 1.46), 26.0)
#     return E0


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


def cherenkov_angle(AirN):
    np.asarray(AirN)
    thetaC = np.zeros_like(AirN)
    mask = AirN != 0
    thetaC[mask] = np.arccos(np.reciprocal(AirN[mask]))
    return thetaC


def cherenkov_threshold(AirN):
    np.asarray(AirN)
    eCthres = np.full_like(AirN, 1e6)
    mask = (AirN != 0) & (AirN != 1)
    eCthres[mask] = 0.511 / np.sqrt(1 - np.reciprocal(AirN[mask] ** 2))
    return eCthres


def cherenkov_photon_yeild(thetaC, wavelength):
    return np.multiply.outer(
        1e9 * (2 * np.pi * (1.0 / 137.04) * np.sin(thetaC) ** 2),
        np.reciprocal(wavelength ** 2),
    )


def sokolsky_rayleigh_scatter(X, wavelength):
    return np.exp(np.multiply.outer(-X / 2974.0, (400.0 / wavelength) ** 4))


_spline_aod = BSpline(
    # fmt: off
    np.array([0., 0., 0., 0., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
              15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
              30., 31., 32., 33., 40., 50., 60., 70., 80., 100., 100., 100., 100.]),
    np.array([2.5e-01, 1.44032618e-01, 8.79347648e-02, 6.34311861e-02, 5.44663855e-02,
              4.87032719e-02, 4.47205268e-02, 4.24146208e-02, 3.76209899e-02,
              3.51014195e-02, 3.19733321e-02, 2.90052521e-02, 2.60056594e-02,
              2.29721102e-02, 2.01058999e-02, 1.66042901e-02, 1.54769396e-02,
              1.14879516e-02, 1.05712540e-02, 6.22703239e-03, 6.52061645e-03,
              3.69050182e-03, 2.71737625e-03, 3.43999316e-03, 1.52265111e-03,
              2.46940239e-03, 5.99739313e-04, 1.13164035e-03, 8.73699276e-04,
              1.37356254e-03, -3.67949454e-4, 9.82352716e-05, -2.49916324e-5,
              5.51770388e-05, -3.37867722e-5, 1.03674479e-05, -2.77699498e-6,
              7.40531994e-07, -4.93687996e-7, 2.46843998e-7, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # fmt: on
    3,
    extrapolate=False,
)
""" BSpline approximation. Aerosol Depth at 55 microns."""


def aerosol_optical_depth(z):
    """Aerosol Optical Depth at 55 microns."""
    z = np.asarray(z)
    # mask = z < 30.0
    # aod = np.zeros_like(z)
    # aod[mask] = _spline_aod(z[mask])
    # return aod
    return _spline_aod(z)


aP = Polynomial(
    np.array(
        [-1.2971, 0.22046e-01, -0.19505e-04, 0.94394e-08, -0.21938e-11, 0.19390e-15]
    )
)
"""Degree 5 polynomial fit of 1/beta_aerosol vs wavelength."""


def aBetaF(w):
    return np.reciprocal(aP(w)) / 0.158


def elterman_mie_aerosol_scatter(z, wavelength, beta_tr, earth_radius):
    """
    z values above 30 (km altitude) return OD filled to 0, this should then
    return aTrans = 1, but in the future masking may be used to further
    optimze for performance by avoiding this computation.
    """

    theta = np.sin(propagation_angle(beta_tr, z, earth_radius))
    aTrans = np.exp(
        -np.multiply.outer(aerosol_optical_depth(z) / theta, aBetaF(wavelength))
    )
    return aTrans


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


def ozone_content(L_n, Lmax, alt_dec, beta_tr, earth_radius, full=False):
    def f(x):
        y = np.multiply.outer(Lmax - L_n, x).T + L_n
        return (
            spline_differential_ozone(
                altitude_along_prop_axis(y, alt_dec, beta_tr, earth_radius)
            )
            * (Lmax - L_n)
        ).T

    zonz, err = qp.quad(f, 0.0, 1.0, epsabs=1e-2, epsrel=1e-2, limit=100)
    return (zonz, err) if full else (zonz)


def ozone_losses(ZonZ, wavelength):
    """
    Calculate ozone losses from points along shower axis (l) in km.

    ############################
    Implementation needs review.
    ############################

    """

    Okappa = 10 ** (110.5 - 44.21 * np.log10(wavelength))
    return np.exp(-1e-3 * np.multiply.outer(ZonZ, Okappa))


def scaled_differential_photon_yield(
    w, ll, z, thetaC, X_ahead, RN, Lmax, alt_dec, beta_tr, earth_radius
):

    pyield = cherenkov_photon_yeild(thetaC, w)
    ZonZ = ozone_content(ll, Lmax, alt_dec, beta_tr, earth_radius)
    TrOz = ozone_losses(ZonZ, w)
    TrRayl = sokolsky_rayleigh_scatter(X_ahead, w)
    TrAero = elterman_mie_aerosol_scatter(z, w, beta_tr, earth_radius)
    SPyield = (pyield * TrOz * TrRayl * TrAero) * (1000 * RN[:, None])
    return SPyield


# def fractional_track_length(s, E, e0=e0):
#     r"""Fractional Track Length in radiation lengths.

#     Hillas 1461 eqn (8) variable T(E) =
#     (Total track length of all charged particles with kinetic energy > E)
#     /
#     (Total vertical component of tracks of all charged particles).

#     total track length of particles if energy > E in vertical thickness interval dx of
#     the shower is N_e*T(E)*dx.

#     """
#     E0 = e0(s)
#     return ((0.89 * E0 - 1.2) / (E0 + E)) ** s * (1.0 + 1.0e-4 * s * E) ** -2


# def outer_fractional_track_length(s, E, e0=e0):
#     E0 = e0(s)
#     pc = ((0.89 * E0 - 1.2)[:, None] / np.add.outer(E0, E)) ** s[:, None]
#     pg = (1.0 + 1.0e-4 * np.multiply.outer(s, E)) ** -2

#     return pc * pg

# def broad_fractional_track_length(s, E, e0=e0):
#     E0 = e0(s)
#     pc = ((0.89 * E0 - 1.2)[:, None] / (E0[:, None]+E)) ** s[:, None]
#     pg = (1.0 + 1.0e-4 * (s[:, None]* E)) ** -2
#     return pc * pg


# def dndu(logenergy, o, s, AirN, thetaC):

#     # z, thenergy

#     theta = np.multiply.outer(thetaC, o)
#     eCthres = cherenkov_threshold(AirN)
#     e2hill = 1150.0 + 454 * np.log(s)

#     energy = 10 ** logenergy

#     vhill = np.multiply.outer(np.reciprocal(e2hill), energy)

#     whill = 2.0 * (1.0 - np.cos(theta)) * ((energy / 21.0) ** 2)
#     w_ave = 0.0054 * energy * (1.0 + vhill) / (1.0 + 13.0 * vhill + 8.3 * vhill ** 2)
#     uhill = whill / w_ave

#     zhill = np.sqrt(uhill)
#     a2hill = np.where(zhill < 0.59, 0.478, 0.380)
#     sv2 = 0.777 * np.exp(-(zhill - 0.59) / a2hill) * uhill

#     # # track = fractional_track_length(s, np.where(energy >= eCthres, energy, eCthres))
#     E = np.multiply.outer(eCthres, energy)
#     mask = energy[None, :] >= eCthres[:, None]
#     E[mask] = np.multiply.outer(np.ones_like(eCthres), energy)[mask]
#     E[~mask] = np.multiply.outer(eCthres, np.ones_like(energy))[~mask]
#     etrack = broad_fractional_track_length(s, E)
#     return sv2 * thetaC[:, None] * energy[None, :] * np.log(10)


def tracklen(E0, eCthres, s):
    """Return tracklength and Tfrac."""
    t1 = np.subtract(np.multiply(0.89, E0), 1.2)
    t2 = np.divide(t1, (E0 + eCthres))
    t3 = np.power(t2, s)
    t4 = np.power(1 + (1e-4 * s * eCthres), 2)
    return np.divide(t3, t4)


def photon_sum(SPYield, DistStep, thetaC, e2hill, eCthres, Tfrac, E0, s):
    """
    Sum photons (Gaisser-Hillas)
    """
    # c     Calc ang spread ala Hillas
    Ieang = 8
    eang = np.arange(1.0, Ieang + (2))
    ehill = np.power(10.0, eang)
    # print(ehill)

    sigval = SPYield

    # c   set limits by distance to det
    # c     and Cherenkov Angle
    CradLim = DistStep * np.tan(thetaC)
    jlim = np.floor(CradLim) + 1
    max_jlim = np.amax(jlim)
    jstep = np.arange(max_jlim)
    jjstep = np.broadcast_to(jstep, (*CradLim.shape, *jstep.shape))
    jmask = jjstep < jlim[..., None]

    athetaj = jjstep[:, 1:] - 0.5
    athetaj = np.arctan2(athetaj, DistStep[:, None])
    athetaj = 2.0 * (1.0 - np.cos(athetaj))

    sthetaj = np.arctan2(jjstep, DistStep[:, None])
    sthetaj = 2.0 * (1.0 - np.cos(sthetaj))
    print(athetaj)

    # c     Calc ang spread ala Hillas
    ehillave = np.where(
        eCthres[..., None] >= ehill[:-1][None, :],
        (eCthres[..., None] + ehill[1:][None, :]) / (2),
        (5) * ehill[:-1][None, :],
    )

    tlen = np.where(
        eCthres[..., None] >= ehill[None, :],
        Tfrac[..., None],
        tracklen(E0[..., None], ehill[None, :], s[:, None]),
    )

    deltrack = tlen[..., :-1] - tlen[..., 1:]
    deltrack[deltrack < 0] = 0.0
    print(eang.shape, deltrack.shape)

    vhill = ehillave / e2hill[..., None]

    wave = 0.0054 * ehillave * (1 + vhill) / (1 + 13 * vhill + 8.3 * vhill ** 2)

    poweha = np.power(ehillave / 21.0, 2)

    uhill = np.einsum("zj,ze->zje", athetaj, poweha)
    uhill /= wave[..., None, :]
    ubin = np.einsum("zj,ze->zje", sthetaj, poweha)
    ubin /= wave[..., None, :]
    ubin = ubin[..., 1:, :] - ubin[..., :-1, :]
    ubin[ubin < 0] = 0

    xhill = np.sqrt(uhill) - (0.59)

    svtrm = np.where(xhill < 0, (0.478), (0.380))
    svtrm = np.exp(-xhill / svtrm)
    svtrm *= ubin * deltrack[..., None, :] * (0.777)
    svtrm[~jmask[..., 1:]] = 0

    photsum = np.einsum("zje,z->z", svtrm, sigval)

    return photsum


def e0(shape, s):
    """not sure what E0 is?"""
    E0 = np.full(shape, 26.0)
    E0[s >= 0.4] = 44.0 - 17.0 * (s[(s >= 0.4)] - 1.46) ** 2
    return E0


def intf(ll, alt_dec, beta_tr, z_max, z_det, earth_radius):

    theta_tr = (np.pi / 2) - beta_tr
    param_beta = np.log(10 ** 8 / (0.710 / 8.36))

    z = altitude_along_prop_axis(ll, alt_dec, beta_tr, earth_radius)
    Lmax = length_along_prop_axis(alt_dec, z_max, beta_tr, earth_radius)

    X_behind = slant_depth(alt_dec, z, theta_tr, epsrel=1e-4)[0]
    X_ahead = slant_depth(z, z_max, theta_tr, epsrel=1e-4)[0]

    T = rad_len_atm_depth(X_behind)
    s = shower_age(T, param_beta)
    RN = greisen_particle_count(T, s, param_beta)

    X_v = shibata_grammage(z)
    AirN = index_of_refraction_air(X_v)
    thetaC = cherenkov_angle(AirN)

    SPyield = qp.quad(
        lambda w: scaled_differential_photon_yield(
            w, ll, z, thetaC, X_ahead, RN, Lmax, alt_dec, beta_tr, earth_radius
        ),
        200,
        900,
        epsabs=1e-4,
        epsrel=1e-4,
    )[0]

    # rect = qp.c2.rectangle_points([1.0, 10.0], [0.0, 1.0])
    # scheme = qp.c2.get_good_scheme(9)
    # val = scheme.integrate(lambda x: dndu(x[0], x[1], s, AirN, thetaC), rect)
    # print("z", z)
    # print("photsum", val)
    # return SPyield * val

    ThetView = viewing_angle(beta_tr, z_det, earth_radius)
    thetPrpa = propagation_angle(beta_tr, z, earth_radius)
    AngE = np.pi / 2 - ThetView - thetPrpa
    DistStep = np.sin(AngE) / np.sin(ThetView) * (z + earth_radius)
    e2hill = 1150.0 + 454 * np.log(s)
    eCthres = cherenkov_threshold(AirN)
    E0 = e0(z.shape, s)
    Tfrac = tracklen(E0, eCthres, s)
    return photon_sum(SPyield, DistStep, thetaC, e2hill, eCthres, Tfrac, E0, s)


def photon_density(alt_dec, beta_tr, z_max, z_det, earth_radius):

    s0 = 0.079417252568371  # s0 = np.exp(-575/227) <-- e2hill == 0: Shower too young.
    s1 = 1.899901462640018  # shower age when greisen_particle_count(s) == 1.0.

    z_lo = altitude_at_shower_age(s0, alt_dec, beta_tr)
    z_hi = altitude_at_shower_age(s1, alt_dec, beta_tr)
    l_lo = length_along_prop_axis(alt_dec, z_lo, beta_tr, earth_radius)
    l_hi = length_along_prop_axis(alt_dec, z_hi, beta_tr, earth_radius)

    print("llow", l_lo, "lhi", l_hi)

    return qp.quad(
        intf,
        l_lo,
        l_hi,
        args=(alt_dec, beta_tr, z_max, z_det, earth_radius),
        epsrel=1e-6,
    )
