import numpy as np
import scipy.stats as stats


def voltage_from_field(
    Efield: np.ndarray,
    freqs: np.ndarray,
    gain: float,
) -> np.ndarray:
    """
    given a peak electric field in V/m and the frequency range of the antenna, calculate voltage seen at the antenna

    Parameters
    Efield: np.ndarray
        peak electric field in V/m
    freqs: np.ndarray
        frequency band in MHz
    gain: float
        gain of the antennas in dBi

    Returns:
        voltage seen by the antenna at each freq
    """

    G = gain  # TODO: get some real gain or directivity loadable with config
    Z_load = 50.0  # 50 ohm load
    Z_antenna = 50.0  # 50 ohm antenna
    c = 299792.458  # speed of light in km/s
    Z_0 = 376.730  # the impedance of free-space in Ohms

    # from https://arxiv.org/pdf/2004.12718.pdf, TODO: not certain about factor of 2 up front
    V = (
        2
        * Efield
        * (1e3 * c / (1e6 * freqs))
        * np.sqrt((Z_antenna / Z_0) * G / (4 * np.pi))
    )
    V_out = (Z_load / (Z_antenna + Z_load)) * V

    return V_out


def noise_voltage(freqs: np.ndarray, h_obs: float) -> np.ndarray:
    """
    returns noise voltage in V

    Parameters
    freqs: np.ndarray
        frequency band in MHz
    h_obs: float
        height in km above the earth surface of your observer

    Returns:
        noise voltage seen by the antenna at each freq
    """

    Re = 6371.0
    theta = np.arctan(Re / (Re + h_obs))
    solid_angle = 2 * np.pi * (1 - np.cos(theta))
    skyFrac = 1.0 - (solid_angle / (4 * np.pi))
    T_earth = 287 * np.ones_like(freqs)  # average earth temp in kelvin
    T_sys = 100.0 * np.ones_like(freqs)  # reasonable number, could be whatever
    T_sky = sky_noise(freqs)
    T_comb = T_sys + (T_earth * (1.0 - skyFrac) + T_sky * skyFrac)

    bw = 1e6 * (freqs[1] - freqs[0])  # bandwidth in Hz
    Z_load = 50  # 50 ohm load
    k_b = 1.38064852e-23  # boltzmann's constant Watts / Hz / K

    V_noise = np.sqrt(T_comb * bw * k_b * Z_load)
    return V_noise


def noise_efield_from_range(freqRange: tuple, h_obs: float) -> np.ndarray:
    df = 10.0
    freqs = np.arange(freqRange[0], freqRange[1], df) + df / 2.0
    return noise_efield(freqs, h_obs)


def noise_efield(freqs: np.ndarray, h_obs: float) -> float:
    """
    returns noise efield in V/m

    Parameters
    freqs: np.ndarray
        frequency band in MHz
    h_obs: float
        height in km above the earth surface of your observer

    Returns:
        sum of the noise efield seen by the antenna at each freq
    """

    Re = 6371.0
    theta = np.arctan(Re / (Re + h_obs))
    solid_angle = 2 * np.pi * (1 - np.cos(theta))
    skyFrac = 1.0 - (solid_angle / (4 * np.pi))
    T_earth = 287 * np.ones_like(freqs)  # average earth temp in kelvin
    T_sys = 100.0 * np.ones_like(freqs)  # reasonable number, could be whatever
    T_sky = sky_noise(freqs)
    T_comb = T_sys + (T_earth * (1.0 - skyFrac) + T_sky * skyFrac)

    bw = 1e6 * (freqs[1] - freqs[0])  # bandwidth in Hz
    Z_0 = 376.730  # the impedance of free-space in Ohms
    k_b = 1.38064852e-23  # boltzmann's constant Watts / Hz / K
    c = 299792458.0

    E_noise = np.sqrt(
        np.sum((2 * freqs * 1e6 / c * np.sqrt(T_comb * bw * k_b * Z_0 * np.pi)) ** 2)
    )
    return E_noise


def sky_noise(freqs: np.ndarray) -> np.ndarray:
    """
    calculate sky noise (galactic + extragalactic) as a function of frequency.
    Parametrization taken from Dulk 2001 (https://www.aanda.org/articles/aa/full/2001/02/aads1858/aads1858.right.html)

    Parameters
    freqs: np.ndarray
        frequencies to evaluate this at (in MHz)

    Returns
        noise temperature in Kelvin
    """
    fhz = freqs * 1e6  # freq in Hz
    c = 1e3 * 299792.458  # speed of light in m/s
    k_b = 1.38064852e-23  # boltzmann's constant Watts / Hz / K

    Ig = 2.48e-20  # galactic scaling
    Ieg = 1.06e-20  # extragalactic scaling
    tau = 5 * np.power(freqs, -2.1)  # tau factor
    P_g = Ig * np.power(freqs, -0.52) * ((1 - np.exp(-tau)) / tau)
    P_eg = Ieg * np.power(freqs, -0.8) * np.exp(-tau)
    P_tot = P_g + P_eg

    return (P_tot / k_b) * (c * c / (2 * fhz * fhz))


def calculate_snr(
    Efield: np.ndarray,
    freqRange: tuple,
    h_obs: float = 525.0,
    Nants: int = 1,
    gain: float = 10.0,
) -> np.ndarray:
    """
    given a peak electric field in V/m and a frequency range, calculate snr

    Parameters
    Efield: np.ndarray
        peak electric field in V/m
    freqRange: float
        tuple with low and high end of frequency band in MHz
    h_obs: float
        height in km above the earth surface of your observer (default = 525km)
    Nants: int
        number of antennas phased together (default = 1)
    gain: float
        gain of the antenna(s) in dBi

    Returns
        SNR for each trial
    """

    df = (
        10.0  # efields made with 10 MHz bins, would need to redo for different bin size
    )
    freqs = np.arange(freqRange[0], freqRange[1], df) + df / 2.0

    V_sig = Nants * voltage_from_field(Efield, freqs, gain)
    V_noise = np.sqrt(Nants * np.sum(noise_voltage(freqs, h_obs) ** 2.0))
    V_sigsum = np.sum(V_sig, axis=1)
    # print(V_sigsum.mean())
    # print(V_noise)

    return V_sigsum / V_noise
