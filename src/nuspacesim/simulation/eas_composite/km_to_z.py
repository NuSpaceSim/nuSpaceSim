import numpy as np


def altittude_to_depth(z):
    """
    # c     Calculate Grammage
    """
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


#%%


def depth_to_altitude(x):
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


x = np.linspace(1, 1030, 10)
z = depth_to_altitude(x)
x_1 = altittude_to_depth(z)


def slant_depth_to_depth(slant_depth, view_zenith_ang):
    depth = slant_depth * np.cos(np.radians(view_zenith_ang))
    return depth
