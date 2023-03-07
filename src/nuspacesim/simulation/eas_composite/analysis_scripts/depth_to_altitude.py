import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def altittude_to_depth(z):
    """
    # c     Calculate Grammage
    """
    z = np.array(z)
    X = np.empty_like(z)
    mask1 = z < 11
    mask2 = np.logical_and(z >= 11, z < 25)
    mask3 = z >= 25
    X[mask1] = np.power(((z[mask1] - 44.34) / -11.861), (1 / 0.19))
    X[mask2] = np.exp(np.divide(z[mask2] - 45.5, -6.34))
    X[mask3] = np.exp(np.subtract(13.841, np.sqrt(28.920 + 3.344 * z[mask3])))

    rho = np.empty_like(z)
    rho[mask1] = (
        -1.0e-5
        * (1 / 0.19)
        / (-11.861)
        * ((z[mask1] - 44.34) / -11.861) ** ((1.0 / 0.19) - 1.0)
    )
    rho[mask2] = np.multiply(-1e-5 * np.reciprocal(-6.34), X[mask2], dtype=np.float32)
    rho[mask3] = np.multiply(
        np.divide(0.5e-5 * 3.344, np.sqrt(28.920 + 3.344 * z[mask3])), X[mask3]
    )
    return X  # , rho


def depth_to_altitude(x):
    x = np.array(x)
    altitude_out = np.empty_like(x)

    # for altitudes z < 11
    altitude = (-11.861 * x ** 0.19) + 44.34
    mask1 = altitude < 11
    altitude_out[mask1] = altitude[mask1]

    # for altitudes z >= 11, z < 25
    altitude = -6.34 * np.log(x) + 45.5
    mask2 = (altitude >= 11) & (altitude < 25)
    altitude_out[mask2] = altitude[mask2]

    # for altitudes  z >= 25
    altitude = ((13.841 - np.log(x)) ** 2 - 28.920) / 3.344
    mask3 = altitude >= 25
    altitude_out[mask3] = altitude[mask3]

    return altitude_out


def shabita_depth_to_altitude(x):
    x = np.array(x)
    altitude_out = np.empty_like(x)
    mask1 = x < 25
    mask2 = (x >= 25) & (x < 230)
    mask3 = x >= 230
    altitude_out[mask1] = (
        47.05 - 6.9 * np.log(x[mask1]) + 0.299 * np.log(x[mask1] / 10) ** 2
    )
    altitude_out[mask2] = 45.5 - 6.34 * np.log(x[mask2])
    altitude_out[mask3] = 44.34 - 11.861 * (x[mask3]) ** 0.19
    return altitude_out


# x = np.linspace(1, 1030, 100)
# z = depth_to_altitude(x)
# plt.plot(z, x)
# z_1 = shabita_depth_to_altitude(x)
# x_1 = altittude_to_depth(z)


def slant_depth_to_depth(slant_depth, corrected_angles):
    depth = slant_depth * np.cos(np.radians(corrected_angles))
    return depth


def calc_alpha(obs_height, earth_emergence_angle):
    """
    calculate the angle of the detector’s line of sight (respect to the local zenith),
    alpha
    """

    def f(xy, r_earth=6371):

        x, y = xy
        z = np.array(
            [
                y
                - (np.cos(np.radians(earth_emergence_angle)) / (r_earth + obs_height)),
                y - (np.cos(x) / r_earth),
            ]
        )
        return z

    alpha_rads = fsolve(f, [0, 1])[0]
    print(fsolve(f, [0, 1])[1])
    alpha_degs = np.degrees(alpha_rads) % 360
    print(alpha_degs)
    if alpha_degs >= 180:
        alpha_degs = 360 - alpha_degs

    return alpha_degs


altitude_array = np.linspace(0, 100, 1000)  # determines the step in altitude
depths = altittude_to_depth(altitude_array)
lower_vertical_depths = depths[:-1]
upper_vertical_depths = depths[1:]
delta_vertical_depth = lower_vertical_depths - upper_vertical_depths

obs_height = 33
beta = 5

r_earth = 6371
# calculate alpha given earth emergance angle and beta by setting a equal to 0
# alpha_deg = calc_alpha(obs_height, earth_emergence_angle=beta)

alpha_deg = np.degrees(
    np.arcsin((np.cos(np.radians(beta)) * r_earth) / (r_earth + obs_height))
)
beta_prime = np.degrees(
    np.arccos(
        (np.sin(np.radians(alpha_deg)) * (r_earth + obs_height))
        / (r_earth + altitude_array[1:])
    )
)
corrected_path_length = delta_vertical_depth / np.sin(np.radians(beta_prime))

upper_slant_depth = np.cumsum(corrected_path_length)

plt.figure(dpi=300)
plt.plot(altitude_array[1:], delta_vertical_depth, label="path length")
plt.plot(altitude_array[1:], corrected_path_length, label="corrected path length")
plt.xlabel("altitude (km)")
plt.ylabel("$\Delta X$")
plt.ylim(-5, 1000)
plt.legend(title=r"$\beta = {}\degree$".format(beta))

plt.figure(dpi=300)
plt.plot(altitude_array[1:], beta_prime)
plt.xlabel("altitude (km)")
plt.ylabel(r"$\beta'$")
plt.legend(title=r"$\beta = {}\degree$".format(beta))

plt.figure(dpi=300)
plt.plot(altitude_array[1:], upper_vertical_depths, label="vertical depth")
plt.plot(
    altitude_array[1:],
    upper_slant_depth,
    label="cumulative sum of corrected path length",
)
# plt.axvline(0.5)
# plt.axhline(800)
plt.xlabel("altitude (km)")
plt.ylabel(r"$g / cm^{-2}$")
plt.legend(title=r"$\beta = {}\degree$".format(beta))
# depth = slant_depth_to_depth(np.linspace(1, 2000, 10), 85)
#%%

# from scipy.integrate import quad


# class Atmosphere:
#     """
#     Class containing constants and methods for using the US Standard Atmosphere of 1976

#     The temperature is assumed to change linearly with height above sea level. From this
#     and the assumption of hybrdostatic equilibrium the pressure and density are calculated.
#     While instantiated, the default parameters give the US Standard Atmosphere. Other
#     atmospheres can be provided
#     """

#     # Class constants for 1976 US Standard Atmosphere
#     temperature_sea_level = 288.15  # K
#     pressure_sea_level = 101325  # Pa
#     density_sea_level = 1.225  # kg/m3
#     air_mol_weight = 28.9644  # amu
#     gravity = 9.80665  # m/s2
#     gas_constant = 8.31432  # J/MolK
#     gMR = gravity * air_mol_weight / gas_constant

#     def __init__(
#         self, altitudes=None, rel_pressure=None, temperatures=None, temp_gradient=None
#     ):
#         """
#         Create and instance of an atmospheric model.
#         If no parameters are provided, the US Standard Atmosphere is
#         used. It uses these values
#             altitudes    = np.array([0.,11000.,20000.,32000.,47000.,51000.,71000.,84852.])
#             rel_pressure = np.array([1.,
#                                      2.23361105092158e-1,5.40329501078488e-2,8.56667835929167e-3,
#                                      1.09456013377711e-3,6.60635313285837e-4,3.90468337334393e-5,
#                                      3.68500952357479e-6])
#             temperatures = np.array([288.15,216.65,216.65,228.65,270.65,270.65,214.65,186.946])
#             temp_gradient = np.array([-0.0065,0.,0.001,0.0028,0.,-0.0028,-0.002,0.])
#         If another atmosphere is to be created, each of the parameters should be an identical length
#         ndarray. All the parameters must be specified or none of them. The altitudes array must
#         be ordered.
#         """
#         if (
#             altitudes is None
#             and rel_pressure is None
#             and temperatures is None
#             and temp_gradient is None
#         ):
#             self.altitudes = np.array(
#                 [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0]
#             )  # m above SL
#             self.rel_pressure = np.array(
#                 [
#                     1.0,
#                     2.23361105092158e-1,
#                     5.40329501078488e-2,
#                     8.56667835929167e-3,
#                     1.09456013377711e-3,
#                     6.60635313285837e-4,
#                     3.90468337334393e-5,
#                     3.68500952357479e-6,
#                 ]
#             )
#             self.temperatures = np.array(
#                 [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
#             )  # K
#             self.temp_gradient = np.array(
#                 [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]
#             )  # K/m
#         else:
#             self.altitudes = altitudes
#             self.rel_pressure = rel_pressure
#             self.temperatures = temperatures
#             self.temp_gradient = temp_gradient
#         self.maximum_height = self.altitudes[-1]
#         self.minimum_height = self.altitudes[0]

#     def atmosphere(self, h):
#         """
#         This function returns atmospheric temperature, pressure, and density as a function of height.

#         Parameters:
#             h - height in atmosphere. This can be an ndarray or a single value. [m]
#         Returns:
#             T   - temperature [K]
#             P   - pressure [Pa]
#             rho - density [kg/m3]
#         """
#         if type(h) is not np.ndarray:
#             h = np.array([h], dtype=float)
#             nin = 0
#         else:
#             nin = len(h)

#         # Find the entry in the tables for each height
#         too_low = h < self.minimum_height
#         too_high = h > self.maximum_height
#         indx = np.searchsorted(self.altitudes, h, side="right")
#         idx = indx - 1

#         # Find the temperature at height
#         altitude = self.altitudes[idx]
#         base_temp = self.temperatures[idx]
#         temp_gradient = self.temp_gradient[idx]
#         delta_altitude = h - altitude
#         temperature = base_temp + temp_gradient * delta_altitude

#         # Find the relative pressure at height
#         base_rel_pressure = self.rel_pressure[idx]
#         flat = np.abs(temp_gradient) < 1.0e-10
#         rel_pressure = np.empty_like(h)
#         rel_pressure[flat] = base_rel_pressure[flat] * np.exp(
#             -self.gMR / 1000 * delta_altitude[flat] / base_temp[flat]
#         )
#         rel_pressure[~flat] = base_rel_pressure[~flat] * (
#             base_temp[~flat] / temperature[~flat]
#         ) ** (self.gMR / 1000 / temp_gradient[~flat])
#         pressure = rel_pressure * self.pressure_sea_level
#         density = (
#             rel_pressure
#             * self.density_sea_level
#             * self.temperature_sea_level
#             / temperature
#         )

#         temperature[too_low] = self.temperature_sea_level
#         pressure[too_low] = self.pressure_sea_level
#         density[too_low] = self.density_sea_level
#         temperature[too_high] = 0.0
#         pressure[too_high] = 0.0
#         density[too_high] = 0.0

#         T = temperature
#         P = pressure
#         rho = density

#         if nin == 0:
#             return T[0], P[0], rho[0]
#         else:
#             return T, P, rho

#     def temperature(self, h):
#         """
#         This function returns temperature as a function of height.

#         Parameters:
#             h - height in atmosphere. This can be an ndarray or a single value. [m]
#         Returns:
#             T - temperature [K]
#         """
#         T, _, _ = self.atmosphere(h)
#         return T

#     def pressure(self, h):
#         """
#         This function returns pressure as a function of height.

#         Parameters:
#             h - height in atmosphere. This can be an ndarray or a single value. [m]
#         Returns:
#             P - pressure [Pa]
#         """
#         _, P, _ = self.atmosphere(h)
#         return P

#     def density(self, h):
#         """
#         This function returns density as a function of height.

#         Parameters:
#             h - height in atmosphere. This can be an ndarray or a single value. [m]
#         Returns:
#             rho - density [kg/m3]
#         """
#         _, _, rho = self.atmosphere(h)
#         return rho

#     def delta(self, h):
#         """
#         This function returns the difference of the index-of-refraction from unity.

#         Parameters:
#             h - height in atmosphere. This can be an ndarray or a single value. [m]

#         Returns:
#             delta - equal to n - 1.
#         """
#         T, P, _ = self.atmosphere(h)
#         P /= 1000.0  # Pa -> kPa
#         return 7.86e-4 * P / T

#     def depth(self, h1, h2=None):
#         """
#         This function returns atmospheric depth. It is the integral of atmospheric density between two heights.

#         Parameters:
#         These parameters can be ndarrays or single values.

#         h1 - height 1 in atmosphere. This can be an ndarray or a single value. [m]
#         h2 - height 2; Default is hMaxAtm. This can be an ndarray or a single value [m]

#         If both h1 and h2 are ndarrays, they must be the same size (the length
#         of the shorter array is used).

#         If h1 or h2 is greater than hMaxAtm, hMaxAtm is used.

#         Returns:
#         The integral of rho from h1 to h2. The result is converted into g/cm2.

#         """
#         if h2 is None:
#             h2 = self.maximum_height * np.ones_like(h1)

#         if type(h1) is not np.ndarray and type(h2) is not np.ndarray:
#             h1 = np.array([h1], dtype=float)
#             h2 = np.array([h2], dtype=float)
#             nin = 0
#         elif type(h2) is not np.ndarray:
#             h2 = h2 * np.ones_like(h1)
#             nin = len(h1)
#         elif type(h1) is not np.ndarray:
#             h1 = h1 * np.ones_like(h2)
#             nin = len(h2)
#         else:
#             nin = min(len(h1), len(h2))

#         A = h1.copy()
#         B = h2.copy()
#         A[A < self.minimum_height] = self.minimum_height
#         B[B < self.minimum_height] = self.minimum_height
#         A[A > self.maximum_height] = self.maximum_height
#         B[B > self.maximum_height] = self.maximum_height

#         depth = np.array([quad(self.density, a, b)[0] for a, b in zip(A, B)])
#         depth /= 10.0  # 1 km/m2 == 1000/10,000 g/cm2

#         if nin == 0:
#             return depth[0]
#         else:
#             return depth

#     def slant_depth(self, theta, d1, d2=None):
#         """
#         This function returns atmospheric depth as a function of the slant angle with respect to the vertical.

#         Parameters:
#             theta - slant angle with respect to the vertical.This can be an ndarray or a single value. [rad]
#             d1 - Distance along slant trajectory. This can be an ndarray or a single value. [m]
#             d2 - Distance along slant trajectory. This can be an ndarray or a single value. [m]

#         If both theta, d1, and d2 are all ndarrays, they must be the same size (the length
#         of the shortest array is used).

#         If d1 or d2 is are beyond the limits of the atmosphere, the limit of the atmosphere is used
#         If d2 is not specified, the limit of the atmosphere is used.
#         A flat-Earth model is assumed, so theta=pi/2 will give infinite results

#         Returns:
#             The slant depth from d2 to d1 at angle theta. [g/cm2]
#         """
#         if d2 is None:
#             d2 = self.maximum_height / np.cos(theta)

#         if (
#             type(theta) is not np.ndarray
#             and type(d1) is not np.ndarray
#             and type(d2) is not np.ndarray
#         ):
#             theta = np.array([theta], dtype=float)
#             d1 = np.array([d1], dtype=float)
#             d2 = np.array([d2], dtype=float)
#             nin = 0
#         elif type(d1) is not np.ndarray and type(d2) is not np.ndarray:
#             d1 = d1 * np.ones_like(theta)
#             d2 = d2 * np.ones_like(theta)
#             nin = len(theta)
#         elif type(theta) is not np.ndarray and type(d2) is not np.ndarray:
#             theta = theta * np.ones_like(d1)
#             d2 = d2 * np.ones_like(d1)
#             nin = len(d1)
#         elif type(theta) is not np.ndarray and type(d1) is not np.ndarray:
#             theta = theta * np.ones_like(d2)
#             d1 = d1 * np.ones_like(d2)
#             nin = len(d2)
#         elif type(theta) is not np.ndarray:
#             theta = theta * np.ones_like(d1)
#             nin = min(len(d1), len(d2))
#         elif type(d1) is not np.ndarray:
#             d1 = d1 * np.ones_like(theta)
#             nin = min(len(theta), len(d2))
#         elif type(d2) is not np.ndarray:
#             d2 = d2 * np.ones_like(theta)
#             nin = min(len(theta), len(d1))
#         else:
#             nin = min(len(theta), len(d1), len(d2))

#         costheta = np.cos(theta)
#         A = d1.copy()
#         B = d2.copy()
#         A[A < self.minimum_height] = self.minimum_height
#         B[B < self.minimum_height] = self.minimum_height
#         bigA = A > self.maximum_height / costheta
#         A[bigA] = self.maximum_height / costheta[bigA]
#         bigB = B > self.maximum_height / costheta
#         B[bigB] = self.maximum_height / costheta[bigB]

#         h1 = A * costheta
#         h2 = B * costheta

#         if nin == 0:
#             return self.depth(h1, h2) / costheta[0]
#         else:
#             return self.depth(h1, h2) / costheta


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     plt.ion()
#     us77 = Atmosphere()
#     h = np.linspace(0, 100000, 101)
#     T, P, rho = us77.atmosphere(h)
#     X = us77.depth(h)
#     plt.plot(h, T, label="Temperature")
#     plt.plot(h, P, label="Pressure")
#     plt.plot(h, rho, label="Density")
#     plt.plot(h, X, label="Depth")
#     X30 = us77.slant_depth(30 * np.pi / 180, h)
#     X60 = us77.slant_depth(60 * np.pi / 180, h)
#     X75 = us77.slant_depth(85 * np.pi / 180, h)
#     # plt.plot(h, X30, label="Slant Depth, 30deg")
#     # plt.plot(h, X60, label="Slant Depth, 60deg")
#     plt.plot(h, X75, label="Slant Depth, 85deg")
#     plt.yscale("log")
#     plt.xlim(h[0], h[-1])
#     plt.grid()
#     plt.legend()
