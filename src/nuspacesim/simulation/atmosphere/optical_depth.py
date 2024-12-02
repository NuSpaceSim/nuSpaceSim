import numpy as np
from scipy.integrate import quad


class US76Atmosphere:
    """Class to hold look-up arrays for US 1976 standard atmosphere."""

    def __init__(self):
        self.H_b = np.array([0, 11, 20, 32, 47, 51, 71, 84.852])
        self.Lm_b = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])
        self.T_b = np.array(
            [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
        )
        # fmt: off
        self.P_b = np.array([1.01325000e+05, 2.26320635e+04, 5.47488866e+03, 8.68018689e+02,
                             1.10906302e+02, 6.69388728e+01, 3.95642046e+00, 3.73383638e-01])
        # fmt: on

        self.Rstar = 8.31432e-3
        self.M0 = 28.9644
        self.gmr = 34.163195

        # Bonus
        self.R = 6371.0
        self.earth_radius = self.R
        self.Z_b = self.H_b * self.R / (self.R - self.H_b)

    def __call__(self, h):
        i = (h > self.H_b[4]) << 2
        i |= (h > self.H_b[i + 2]) << 1
        i |= h > self.H_b[i + 1]
        return i


def us_std_atm_density(z, R=6371.0, a=US76Atmosphere()):
    z = np.asarray(z)
    h = z * R / (z + R)
    i = a(h)

    deltah = h - a.H_b[i]
    temperature = a.T_b[i] + a.Lm_b[i] * deltah

    mask = a.Lm_b[i] == 0
    pressure = np.full(z.shape, a.P_b[i])
    pressure[mask] *= np.exp(-a.gmr * deltah[mask] / a.T_b[i][mask])
    pressure[~mask] *= (a.T_b[i][~mask] / temperature[~mask]) ** (
        a.gmr / a.Lm_b[i][~mask]
    )

    # density = (pressure * M0) / (Rstar * 1e6 * temperature)  # kg/m^3
    # return 1e-3 * density  # g/cm^3
    density = pressure / temperature
    rho = 1e-9 * a.M0 / a.Rstar * density  # g/cm^3
    return rho


def differendial_pathlength(z, beta, R=6371.0):
    z_ = z
    b_ = beta
    return (z_ + R) / np.sqrt(R**2 * np.sin(b_) ** 2 + z_**2 + 2 * R * z_)


def pathlength(z, beta, R=6371.0):
    return -R * np.sin(beta) + np.sqrt((R * np.sin(beta)) ** 2 + z**2 + 2 * R * z)


def differential_optical_depth(z, beta, *args, **kwargs):
    return 1e5 * us_std_atm_density(z) * differendial_pathlength(z, beta)


def optical_depth(low, high, beta):
    max_len = np.amax(
        [
            1 if np.isscalar(low) else len(low),
            1 if np.isscalar(high) else len(high),
            1 if np.isscalar(beta) else len(beta),
        ]
    )

    high = np.full(max_len, high) if np.isscalar(high) else np.asarray(high)
    low = np.full(max_len, low) if np.isscalar(low) else np.asarray(low)
    beta = np.full(max_len, beta) if np.isscalar(beta) else np.asarray(beta)

    def dX_b(beta):
        def dX(z):
            return differential_optical_depth(z, beta)

        return dX

    return np.asarray(
        [quad(dX_b(b), lo, h, epsabs=1e-3)[0] for lo, h, b in zip(low, high, beta)]
    )


def hessian_optical_depth(z, beta, R=6371.0, a=US76Atmosphere(), zeps=1e-4, beps=1e-4):
    """
    Much of the aparatus is shared with dX. The two might be combined.
    """
    z = z + zeps
    b = beta + beps
    dln = z + R
    dld2 = R**2 * np.sin(b) ** 2 + z**2 + 2 * R * z
    dld = np.sqrt(dld2)
    dl = dln / dld

    Rstar = 8.31432e-3
    M0 = 28.9644
    gmr = 34.163195
    coeff = 1e-4 * M0 / Rstar

    z = np.asarray(z)

    h = z * R / (z + R)
    i = a(h)
    deltah = h - a.H_b[i]
    temperature = a.T_b[i] + a.Lm_b[i] * deltah

    mask = a.Lm_b[i] == 0
    pressure = np.full(z.shape, a.P_b[i])
    pressure[mask] *= np.exp(-gmr * deltah[mask] / a.T_b[i][mask])
    pressure[~mask] *= (a.T_b[i][~mask] / temperature[~mask]) ** (
        gmr / a.Lm_b[i][~mask]
    )
    density = pressure / (temperature)
    dX = coeff * density * dl

    dh = (R / dln) ** 2
    # d2l = 1.0/dld - dl*dln/dld2
    d2X = dX * ((1.0 / dln) - (dln / dld2) - dh * (a.Lm_b[i] + gmr) / temperature)
    # d2X = dX * (d2l/dl - dh*(a.Lm_b[i] + gmr)/temperature)
    # d2X = coeff * density * d2l - dX*dh*(a.Lm_b[i] + gmr)/temperature
    # d2X = coeff*density/dld -dX*dln/dld2 -dX*dh*(a.Lm_b[i] + gmr)/np.where(mask, a.T_b[i], temperature)

    return d2X


def shower_propagation_length(
    decay_altitude,
    beta,
    dfunc=differential_optical_depth,
    hess=hessian_optical_depth,
    Xtarg=2344.6,
    zMax=125.0,  # R=6371.0,
    Nqmax=8,
    Nqroot=3,
    Niter=1,
    xeps=1e-4,
):
    """
    Computes the shower propagation length based on decay altitude and beta angle.

    Parameters
    ----------
    decay_altitude : scalar or array-like
        Altitude at which the decay occurs (in km). Can be a scalar or an array.
    beta : scalar or array-like
        Angle of entry or beta angle (in radians). Can be a scalar or an array.
    dfunc : callable, optional
        First derivative function for numerical integration, default is `dX`.
    hess : callable, optional
        Second derivative function for numerical integration, default is `d2X`.
    Xtarg : float, optional
        Target path length threshold (default is 2344.6).
    zMax : float, optional
        Maximum altitude for the computation (default is 125.0 km).
    R : float, optional
        Radius of the Earth (default is 6371.0 km).
    Nqmax : int, optional
        Number of quadrature points for numerical integration of `XMax` (default is 8).
    Nqroot : int, optional
        Number of quadrature points for root-finding (default is 3).
    Niter : int, optional
        Number of iterations for Halley's method during root-finding (default is 1).

    Returns
    -------
    scalar or ndarray
        The computed shower propagation length(s). Returns a scalar if both `decay_altitude`
        and `beta` are scalars, otherwise returns an array with the same shape as the input.

    Raises
    ------
    RuntimeError
        If `decay_altitude` and `beta` do not have the same shape.

    Notes
    -----
    - The function uses numerical integration with Gauss-Legendre quadrature and Halley's method for root finding.
    - Root finding determines the altitude where the path length threshold `Xtarg` is met, considering physical constraints.
    - If `decay_altitude` and `beta` are scalars, the function simplifies to return a scalar result.

    Example
    -------
    Compute the propagation length for a single altitude and beta:

    >>> decay_altitude = 10.0  # km
    >>> beta = np.radians(15)  # radians
    >>> propagation_len = shower_propagation_length(decay_altitude, beta)
    >>> print(propagation_len)

    Compute the propagation length for arrays of altitudes and betas:

    >>> decay_altitude = np.linspace(5, 15, 100)
    >>> beta = np.radians(np.linspace(10, 30, 100))
    >>> propagation_len = shower_propagation_length(decay_altitude, beta)
    >>> print(propagation_len.shape)  # Output: (100,)
    """

    return_as_scalar = np.isscalar(decay_altitude) and np.isscalar(beta)
    decay_altitude = np.atleast_1d(decay_altitude)
    beta = np.atleast_1d(beta)

    if decay_altitude.shape != beta.shape:
        raise RuntimeError("Decay Altitude and Beta parameters must have same shape!")

    shape = beta.shape

    # Xmax = legendre_slant_depth(beta, decay_altitude, dfunc, Nqmax)
    x, w = np.polynomial.legendre.leggauss(Nqmax)
    x_new = (
        0.5 * (zMax - decay_altitude)[..., None] * x[None, :]
        + 0.5 * (zMax + decay_altitude)[..., None]
    )
    w_new = 0.5 * (zMax - decay_altitude)[..., None] * w[None, :]
    XMax = np.sum(w_new * dfunc(x_new, beta[..., None]), axis=-1)

    no_root_mask = XMax > Xtarg

    # z0 = approximate_halleys_legendre(decay_altitude, -Xtarg, dX_b(beta), d2X_b(beta), Nqroot, Niter)
    x0 = decay_altitude[no_root_mask]

    y0 = -Xtarg
    mbeta = beta[no_root_mask]
    dfval = dfunc(x0, mbeta)
    d2fxval = hess(x0, mbeta)
    # Halley's approximation for initial step
    xn = x0 - 2.0 * y0 * dfval / (2.0 * dfval**2 - y0 * d2fxval)
    # Nuemrical fix for the rare cases where Helly's approx results in a poor initial step.
    bad_step_mask = xn < x0
    # Newton's step instead
    xn[bad_step_mask] = x0[bad_step_mask] - y0 / dfval[bad_step_mask]
    # final fix for poor initial step. bump by arbitrary perturbation
    bad_step_mask = xn < x0
    xn[bad_step_mask] = x0[bad_step_mask] + xeps

    qx, qw = np.polynomial.legendre.leggauss(Nqroot)
    for _ in range(Niter):
        qix = 0.5 * (xn - x0)[..., None] * qx[None, :] + 0.5 * (xn + x0)[..., None]
        assert np.all(qix > 0.0)
        qiw = 0.5 * (xn - x0)[..., None] * qw[None, :]
        yn = y0 + np.sum(qiw * dfunc(qix, mbeta[..., None]), axis=-1)

        # Step 2: Halley's hop
        dfval = dfunc(xn, mbeta)
        d2fxval = hess(xn, mbeta)
        xnp1 = xn - 2.0 * yn * dfval / (2.0 * dfval**2 - yn * d2fxval)
        # Nuemrical fix for the rare cases where Helly's approx results in a poor initial step.
        bad_step_mask = xnp1 < x0
        # Newton's step instead
        xnp1[bad_step_mask] = (
            xn[bad_step_mask] - yn[bad_step_mask] / dfval[bad_step_mask]
        )
        bad_step_mask = xnp1 < x0
        xnp1[bad_step_mask] = xn[bad_step_mask] + xeps
        xn = xnp1

    max_altitude = np.full(shape, zMax)
    max_altitude[no_root_mask] = xn

    propagation_len = pathlength(max_altitude, beta) - pathlength(decay_altitude, beta)
    return propagation_len[0] if return_as_scalar else propagation_len
