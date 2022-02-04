import cubepy as cp
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline
from scipy.optimize import newton


def viewing_angle(beta_tr, Zdet, Re=6378.1):
    return np.arcsin((Re / (Re + Zdet)) * np.cos(beta_tr))


def propagation_angle(beta_tr, z, Re=6378.1):
    return np.arccos((Re / (Re + z)) * np.cos(beta_tr))


def propagation_theta(beta_tr, z, Re=6378.1):
    return propagation_angle(beta_tr, z, Re)


def length_along_prop_axis(z_start, z_stop, beta_tr, Re=6378.1):
    L1 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_stop + z_stop ** 2
    L2 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_start + z_start ** 2
    L = np.sqrt(L1) - np.sqrt(L2)
    return L


def deriv_length_along_prop_axis(z_stop, beta_tr, Re=6378.1):
    L1 = Re ** 2 * np.sin(beta_tr) ** 2 + 2 * Re * z_stop + z_stop ** 2
    L = (Re + z_stop) / np.sqrt(L1)
    return L


def altitude_along_prop_axis(L, z_start, beta_tr, Re=6378.1):
    r1 = Re ** 2
    r2 = 2 * Re * z_start
    r3 = z_start ** 2
    return -Re + np.sqrt(
        L ** 2 + 2 * L * np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3) + r1 + r2 + r3
    )


def deriv_altitude_along_prop_axis(L, z_start, beta_tr, Re=6378.1):
    r1 = Re ** 2
    r2 = 2 * Re * z_start
    r3 = z_start ** 2
    r4 = np.sqrt(r1 * np.sin(beta_tr) ** 2 + r2 + r3)
    denom = np.sqrt(L ** 2 + 2 * L * r4 + r1 + r2 + r3)
    numer = (Re + z_start) * ((L) / r4 + 1)
    return numer / denom


def gain_in_altitude_along_prop_axis(L, z_start, beta_tr, Re=6378.1):
    return altitude_along_prop_axis(L, z_start, beta_tr, Re) - z_start


def distance_to_detector(beta_tr, z, z_det, earth_radius=6378.1):
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


def greisen_particle_count(T, s, param_beta=np.log(10 ** 8 / (0.710 / 8.36))):
    r"""Particle count as a function of radiation length from atmospheric depth

    Hillas 1461 eqn (6)

    N_e(T) where y is beta in EASCherGen
    """
    N_e = (0.31 / np.sqrt(param_beta)) * np.exp(T * (1.0 - 1.5 * np.log(s)))
    N_e[N_e < 0] = 0.0
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


def us_std_atm_density(z, earth_radius=6371):
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


# fmt: off
_polyrho = Polynomial(
    [-1.00867666e-07, 2.39812768e-06, 9.91786255e-05, -3.14065045e-04, -6.30927456e-04,
     1.70053229e-03, 2.61087236e-03, -5.69630760e-03, -2.12098836e-03, 5.68074214e-03,
     6.54893281e-04, -1.98622752e-03, ],
    domain=[0.0, 100.0],
)
# fmt: on


def polyrho(z):
    """
    Density (g/cm^3) parameterized from altitude (z) values

    Computation is an (11) degree polynomial fit to equation (2)
    in https://arxiv.org/pdf/2011.09869.pdf
    Fit performed using numpy.Polynomial.fit
    """

    p = np.where(z < 100, _polyrho(z), 1e-9)
    return p


def slant_depth_integrand(z, theta_tr, earth_radius, rho=polyrho):

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


# def slant_depth_temp(x, p1, p2, p3, p4):
#     z, t = x
#     v = (p1 * np.tan(t) + p2) * (
#         p3 + us_std_atm_density(0) - p4 * 6378.1 * us_std_atm_density(z)
#     )
#     return v.ravel()


def slant_depth_trig_approx(z_lo, z_hi, theta_tr, z_max=100.0):

    rho = us_std_atm_density
    r0 = rho(0)
    ptan = 92.64363150999402 * np.tan(theta_tr) + 101.4463720303218

    def approx_slant_depth(z):
        return ptan * (8.398443922535177 + r0 - 6340.6095008383245 * rho(z))

    fmax = approx_slant_depth(z_max)
    sd_hi = np.where(z_hi >= z_max, fmax, approx_slant_depth(z_hi))
    sd_lo = np.where(z_lo >= z_max, fmax, approx_slant_depth(z_lo))

    return sd_hi - sd_lo


def slant_depth_trig_behind_ahead(z_lo, z, z_hi, theta_tr, z_max=100.0):

    rho = us_std_atm_density
    r0 = rho(0)
    ptan = 92.64363150999402 * np.tan(theta_tr) + 101.4463720303218

    def approx_slant_depth(z):
        return ptan * (8.398443922535177 + r0 - 6340.6095008383245 * rho(z))

    fmax = approx_slant_depth(z_max)
    sd_hi = np.where(z_hi >= z_max, fmax, approx_slant_depth(z_hi))
    sd_mid = np.where(z >= z_max, fmax, approx_slant_depth(z))
    sd_lo = np.where(z_lo >= z_max, fmax, approx_slant_depth(z_lo))

    return sd_mid - sd_lo, sd_hi - sd_mid


def altitude_at_shower_age(s, alt_dec, beta_tr, z_max=65.0, **kwargs):
    """Altitude as a function of shower age, decay altitude and emergence angle."""

    alt_dec = np.asarray(alt_dec)
    beta_tr = np.asarray(beta_tr)

    theta_tr = 0.5 * np.pi - beta_tr
    param_beta = np.log(10 ** 8 / (0.710 / 8.36))

    # Check that shower age is within bounds
    ss = shower_age(
        rad_len_atm_depth(slant_depth_trig_approx(alt_dec, z_max, theta_tr))
    )
    mask = ss < s

    X_s = -1.222e19 * param_beta * s / ((10.0 / 6.0) * 1e17 * s - 5e17)

    def ff(z):
        X = slant_depth_trig_approx(alt_dec[~mask], z, theta_tr[~mask])
        return X - X_s

    altitude = np.full_like(alt_dec, z_max)
    altitude[~mask] = newton(ff, alt_dec[~mask], **kwargs)

    return altitude


def shibata_grammage(z):
    z = np.asarray(z)
    # conditional cutoffs
    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25

    # 1 / 0.19
    r1 = 1 / 0.19

    # X_v vertical atmospheric depth g / cm^2
    X_v = np.empty_like(z)
    X_v[mask1] = np.power(((z[mask1] - 44.34) / -11.861), r1)
    X_v[mask2] = np.exp((z[mask2] - 45.5) / -6.34)
    X_v[mask3] = np.exp(13.841 - np.sqrt(28.920 + 3.344 * z[mask3]))

    return X_v


def cherenkov_angle(AirN):
    np.asarray(AirN)
    thetaC = np.zeros_like(AirN)
    mask = AirN > 0.0
    thetaC[mask] = np.arccos(1 / AirN[mask])
    return thetaC


def cherenkov_threshold(AirN):
    np.asarray(AirN)
    eCthres = np.full_like(AirN, 1e6)
    mask = (AirN != 0) & (AirN != 1)
    eCthres[mask] = 0.511 / np.sqrt(1.0 - (1.0 / AirN[mask] ** 2))
    return eCthres


def cherenkov_photon_yeild(thetaC, wavelength):
    return (
        1e9 * 2 * np.pi * (1.0 / 137.04) * np.sin(thetaC) ** 2 * (1.0 / wavelength ** 2)
    )


def sokolsky_rayleigh_scatter(X, wavelength):
    return np.exp(np.multiply(-X / 2974.0, (400.0 / wavelength) ** 4))


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
    extrapolate=False,
)

_spline_ozone_antideriv = BSpline(
    # fmt: off
    np.array(
        [3.47776272e-26, 3.47776272e-26, 3.47776272e-26, 3.47776272e-26,
         3.47776272e-26, 1.67769146e01, 2.42250765e01, 3.29917411e01, 4.08815284e01,
         4.88486345e01, 5.10758252e01, 5.69937723e01, 6.34112652e01, 6.58453130e01,
         9.02965953e01, 9.02965953e01, 9.02965953e01, 9.02965953e01, 9.02965953e01]
    ),
    np.array(
        [0.0, 11.759519588846869, 32.41437153886416, -2.4409022011352945,
         260.5268300289521, 311.2888495897276, 322.8973948796908, 323.2975442838028,
         323.34637642598335, 323.4816365862455, 323.72229160561886, 323.94134439572065,
         324.1100743635356, 324.26363314097864, 324.26363314097864, 324.26363314097864,
         324.26363314097864, 324.26363314097864, 324.26363314097864]),
    # fmt: on
    4,
)


def ozone_losses(z, detector_altitude, theta_tr, wavelength):

    term1 = -2.3457647210565447 + 3.386767160062623 * 1 / np.sqrt(np.cos(theta_tr))

    def ozone_depth_approx(x):
        s = 0.8509489496334921 * _spline_ozone_antideriv(x)
        return term1 * (5.090655199334124 + s)

    d1 = ozone_depth_approx(detector_altitude)
    d0 = ozone_depth_approx(z)
    depth = d1 - d0

    ozone_kappa = 10 ** (110.5 - 44.21 * np.log10(wavelength))

    return np.exp(-1e-3 * depth * ozone_kappa)


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
    return _spline_aod(np.asarray(z))


aP = Polynomial(
    np.array(
        [-1.2971, 0.22046e-01, -0.19505e-04, 0.94394e-08, -0.21938e-11, 0.19390e-15]
    )
)
"""Degree 5 polynomial fit of 1/beta_aerosol vs wavelength."""


def aBetaF(w):
    return (1.0 / aP(w)) / 0.158


def elterman_mie_aerosol_scatter(z, wavelength, beta_tr, earth_radius=6378.1):
    """
    z values above 30 (km altitude) return OD filled to 0, this should then
    return aTrans = 1, but in the future masking may be used to further
    optimze for performance by avoiding this computation.
    """

    theta = np.sin(propagation_angle(beta_tr, z, earth_radius))
    aTrans = np.exp(-np.multiply(aerosol_optical_depth(z) / theta, aBetaF(wavelength)))
    return aTrans


def track_length(s, E):
    r"""Differential Track Length in radiation lengths.

    Hillas 1461 eqn (8) variable T(E) =
    (Total track length of all charged particles with kinetic energy > E)
    /
    (Total vertical component of tracks of all charged particles).

    The total track length of particles of energy > E in vertical thickness interval dx
    of the shower is N_e*T(E)*dx.

    """

    E0 = np.where(s >= 0.4, 44.0 - 17.0 * (s - 1.46), 26.0)
    "Kinetic Energy charged primary of shower particles (MeV)"

    return ((0.89 * E0 - 1.2) / (E0 + E)) ** s * (1.0 + 1.0e-4 * s * E) ** -2


def dndu(energy, theta, s):

    e2hill = 1150.0 + 454.0 * np.log(s)
    mask = e2hill > 0
    vhill = energy[mask] / e2hill[mask]
    whill = 2.0 * (1.0 - np.cos(theta[mask])) * ((energy[mask] / 21.0) ** 2)
    w_ave = (
        0.0054 * energy[mask] * (1.0 + vhill) / (1.0 + 13.0 * vhill + 8.3 * vhill ** 2)
    )
    uhill = whill / w_ave

    zhill = np.sqrt(uhill)
    a2hill = np.where(zhill < 0.59, 0.478, 0.380)
    sv2 = 0.777 * np.exp(-(zhill - 0.59) / a2hill)
    rval = np.zeros_like(e2hill)
    rval[mask] = sv2
    return rval


def photon_count():

    detector_altitude = 100.0

    def df(x, alt_dec, beta_tr):

        z = x[0]
        w = x[1]
        o = x[2]
        logenergy = x[3]
        energy = 10 ** logenergy
        theta_tr = (np.pi / 2) - beta_tr

        X_v = shibata_grammage(z)
        AirN = index_of_refraction_air(X_v)
        thetaC = cherenkov_angle(AirN)
        theta = thetaC * o

        X_behind, X_ahead = slant_depth_trig_behind_ahead(
            alt_dec, z, detector_altitude, theta_tr
        )

        T = rad_len_atm_depth(X_behind)
        s = shower_age(T)
        RN = greisen_particle_count(T, s)

        rval = (
            1e3
            * RN
            * track_length(s, energy)
            * cherenkov_photon_yeild(thetaC, w)
            * sokolsky_rayleigh_scatter(X_ahead, w)
            * elterman_mie_aerosol_scatter(z, w, beta_tr)
            * ozone_losses(z, detector_altitude, theta_tr, w)
            * dndu(energy, theta, s)
            * thetaC
        )

        return rval

    def eas_wrapper(x, evtidx, alt_dec, beta_tr):
        return df(x, alt_dec[evtidx], beta_tr[evtidx])

    n = int(1e3)
    N = n * n

    # alt_dec = np.random.uniform(0.1, 2.2, N)
    # beta_tr = np.random.uniform(np.radians(1), 0.5 * np.pi, N)
    alt_dec = np.linspace(0.2, 2.2, n)
    beta_tr = np.linspace(np.radians(20), 0.5 * np.pi, n)
    alt_dec, beta_tr = np.meshgrid(alt_dec, beta_tr)
    alt_dec = alt_dec.ravel()
    beta_tr = beta_tr.ravel()

    s0 = 0.079417252568371  # s0 = np.exp(-575/227) <-- e2hill == 0: Shower too young.
    s1 = 1.899901462640018  # shower age when greisen_particle_count(s) == 1.0.

    z_lo = altitude_at_shower_age(s0, alt_dec, beta_tr)
    z_hi = altitude_at_shower_age(s1, alt_dec, beta_tr)

    lo = np.stack((z_lo, np.full(N, 200.0), np.zeros(N), np.ones(N)), 0)
    hi = np.stack((z_hi, np.full(N, 900.0), np.ones(N), np.full(N, 17)), 0)

    return cp.integrate(
        eas_wrapper,
        lo,
        hi,
        args=(alt_dec, beta_tr),
        evt_idx_arg=True,
        range_dim=1,
        abstol=1e8,
        reltol=1e-1,
        parallel=True,
        tile_byte_limit=2 ** 25,
    )


if __name__ == "__main__":
    np.set_printoptions(linewidth=256, precision=17)

    val, err = photon_count()

    minerr_i = np.argmin(err)
    maxerr_i = np.argmax(err)

    print(f"{val[minerr_i]:E}", f"{err[minerr_i]:E}")
    print(f"{val[maxerr_i]:E}", f"{err[maxerr_i]:E}")
