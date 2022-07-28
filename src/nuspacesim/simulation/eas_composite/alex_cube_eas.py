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


def altitude_at_shower_age(s, alt_dec, beta_tr, z_max=500, **kwargs):
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


if __name__ == "__main__":
    alt = altitude_at_shower_age(s=1, alt_dec=1, beta_tr=np.radians(5), z_max=100)

    # np.set_printoptions(linewidth=256, precision=17)

    # val, err = photon_count()

    # minerr_i = np.argmin(err)
    # maxerr_i = np.argmax(err)

    # print(f"{val[minerr_i]:E}", f"{err[minerr_i]:E}")
    # print(f"{val[maxerr_i]:E}", f"{err[maxerr_i]:E}")
