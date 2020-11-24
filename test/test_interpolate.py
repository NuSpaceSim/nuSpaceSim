import numpy as np
from scipy import interpolate
import timeit

tecdfarr = np.array([
    [
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
        0.00e00,
    ],
    [
        0.00e00,
        0.00e00,
        0.00e00,
        3.99e-04,
        2.70e-03,
        3.77e-03,
        1.56e-03,
        2.00e-03,
        2.75e-03,
        0.00e00,
    ],
    [
        1.35e-04,
        3.61e-04,
        0.00e00,
        7.99e-04,
        4.05e-03,
        3.77e-03,
        1.56e-03,
        6.00e-03,
        5.51e-03,
        6.29e-03,
    ],
    [
        1.35e-04,
        7.21e-04,
        5.10e-04,
        1.20e-03,
        5.41e-03,
        7.53e-03,
        3.12e-03,
        1.40e-02,
        1.10e-02,
        1.89e-02,
    ],
    [
        1.35e-04,
        1.08e-03,
        7.65e-04,
        3.99e-03,
        1.22e-02,
        1.32e-02,
        7.79e-03,
        2.20e-02,
        2.48e-02,
        5.03e-02,
    ],
    [
        1.35e-04,
        1.80e-03,
        3.32e-03,
        9.19e-03,
        1.69e-02,
        2.17e-02,
        1.40e-02,
        3.20e-02,
        4.13e-02,
        6.92e-02,
    ],
    [
        2.70e-04,
        2.88e-03,
        5.61e-03,
        1.24e-02,
        2.36e-02,
        2.82e-02,
        2.49e-02,
        4.60e-02,
        6.89e-02,
        1.01e-01,
    ],
    [
        8.10e-04,
        4.33e-03,
        9.69e-03,
        1.84e-02,
        3.31e-02,
        4.52e-02,
        5.45e-02,
        5.80e-02,
        1.10e-01,
        1.57e-01,
    ],
    [
        2.16e-03,
        7.75e-03,
        1.53e-02,
        2.72e-02,
        4.39e-02,
        7.06e-02,
        7.94e-02,
        8.60e-02,
        1.43e-01,
        2.39e-01,
    ],
    [
        3.64e-03,
        1.17e-02,
        2.35e-02,
        4.03e-02,
        6.55e-02,
        9.79e-02,
        1.29e-01,
        1.46e-01,
        1.76e-01,
        3.40e-01,
    ],
    [
        6.07e-03,
        1.97e-02,
        3.42e-02,
        6.03e-02,
        9.66e-02,
        1.30e-01,
        1.81e-01,
        1.94e-01,
        2.45e-01,
        4.28e-01,
    ],
    [
        9.58e-03,
        2.90e-02,
        5.28e-02,
        8.39e-02,
        1.26e-01,
        1.76e-01,
        2.34e-01,
        2.68e-01,
        3.09e-01,
        5.16e-01,
    ],
    [
        1.55e-02,
        4.35e-02,
        8.21e-02,
        1.21e-01,
        1.70e-01,
        2.23e-01,
        2.98e-01,
        3.18e-01,
        3.80e-01,
        5.91e-01,
    ],
    [
        3.01e-02,
        6.29e-02,
        1.11e-01,
        1.72e-01,
        2.27e-01,
        2.80e-01,
        3.72e-01,
        3.80e-01,
        4.74e-01,
        6.48e-01,
    ],
    [
        4.98e-02,
        9.32e-02,
        1.53e-01,
        2.23e-01,
        2.98e-01,
        3.51e-01,
        4.55e-01,
        4.90e-01,
        5.51e-01,
        7.23e-01,
    ],
    [
        9.11e-02,
        1.44e-01,
        2.18e-01,
        2.95e-01,
        3.72e-01,
        4.47e-01,
        5.48e-01,
        6.02e-01,
        6.36e-01,
        8.11e-01,
    ],
    [
        1.71e-01,
        2.36e-01,
        3.14e-01,
        3.88e-01,
        4.72e-01,
        5.42e-01,
        6.26e-01,
        6.98e-01,
        7.11e-01,
        8.93e-01,
    ],
    [
        3.22e-01,
        3.85e-01,
        4.58e-01,
        5.19e-01,
        5.93e-01,
        6.61e-01,
        7.41e-01,
        7.94e-01,
        8.13e-01,
        9.43e-01,
    ],
    [
        5.86e-01,
        6.34e-01,
        6.78e-01,
        7.22e-01,
        7.80e-01,
        8.15e-01,
        8.68e-01,
        8.86e-01,
        8.90e-01,
        9.75e-01,
    ],
    [
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
    ],
    [
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
        1.00e00,
    ],
])

tetauefracarr = np.array([
    0.0112,
    0.0141,
    0.0178,
    0.0224,
    0.0282,
    0.0355,
    0.0447,
    0.0562,
    0.0708,
    0.0891,
    0.112,
    0.141,
    0.178,
    0.224,
    0.282,
    0.355,
    0.447,
    0.562,
    0.708,
    0.891,
    1.0,
])

tauEFracInterps = [
    interpolate.interp1d(tecdfarr[:, betaind], tetauefracarr)
    for betaind in range(10)
]


def interp_direct(betainds, us):
    rs = np.empty_like(us)
    for i, u in enumerate(np.nditer(us)):
        rs[i] = interpolate.interp1d(tecdfarr[:, betainds[i]],
                                     tetauefracarr)(u)
    return rs


def interp_loop_betas(betainds, u):
    r = np.empty_like(u)
    for i in range(r.size):
        r[i] = tauEFracInterps[betainds[i]](u[i])
    return r


def interp_loop_betas_mask(betainds, u):
    r = np.empty_like(u)
    for beta in range(10):
        idxs = betainds[betainds == beta]
        r[idxs] = tauEFracInterps[beta](u[idxs])
    return r


N = 1_000
repeat = 10

u5 = np.random.rand(N, )
betaIdxs = np.random.randint(10, size=N)

print("v3 execution time in seconds: {}".format(
    timeit.timeit(
        lambda: interp_loop_betas_mask(betaIdxs, u5),
        number=repeat,
        globals=globals(),
    )))

print("v2 execution time in seconds: {}".format(
    timeit.timeit(lambda: interp_loop_betas(betaIdxs, u5), number=repeat,
                  globals=globals())))

print("v1 execution time in seconds: {}".format(
    timeit.timeit(lambda: interp_direct(betaIdxs, u5), number=repeat,
                  globals=globals())))
