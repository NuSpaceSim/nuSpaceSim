from nuspacesim.simulation.eas_optical.atmospheric_models import (
    rho,
    slant_depth,
    slant_depth_integrand,
)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import quadpy as qp

from timeit import timeit

if __name__ == "__main__":

    # print(slant_depth_integrand(0, np.pi/2))
    # slant-depth from gaussian quadriture
    matplotlib.rcParams.update({"font.size": 18})

    N = int(1e5)
    theta_tr = np.random.uniform(0.0, np.pi / 2, N)
    alt_dec = np.random.uniform(0.0, 10.0, N)

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, squeeze=True)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(projection="3d")
    # # ax2 = fig.add_subplot(projection='3d')
    # Xs, ers = slant_depth(0, 65, 0, epsabs=1e-2, epsrel=1e-2, func=sdi)
    # print(Xs, Xs.shape)
    # Xs, ers = slant_depth(0, 65, 0, epsabs=1e-2, epsrel=1e-2)
    # print(Xs, Xs.shape)
    print(
        timeit(
            lambda: slant_depth(
                alt_dec, 65, theta_tr, rho=rho, epsabs=1e-7, epsrel=1e-7
            ),
            number=1,
        )
    )

    # # ax1.scatter(alt_dec, theta_tr, Xs, label="quad")
    # # ax1.scatter(alt_dec, theta_tr, ers, label="quad")

    # # for f in [qp.c1.gauss_patterson(5)]:
    # #     # f.show()
    # #     Xs, ers = f.integrate(lambda x: slant_depth_integrand(x, theta_tr, 6371.0), 0, 65)
    # #     ax1.plot(theta_tr, Xs, label=f.__name__)
    # #     ax2.plot(theta_tr, ers, label=f.__name__)
    # ax1.set_xlabel("Decay Altitude")
    # ax1.set_ylabel("Zenith Angle")
    # ax1.set_zlabel("Slant Depth")
    # # ax1.set_zlabel("Numerical Error")
    # # ax1.grid()
    # # ax1.legend()
    # # ax1.set_ylim([0, 4e4])
    # # for i, s, t, l in zip(coloridx, sds, tds, labs):
    # #     ax2.semilogy(theta_tr, s, alpha=0.5, label=l, color=plt.cm.jet(i))
    # #     ax2.semilogy(theta_tr, t, ":", label=l + "trap", color=plt.cm.jet(i))
    # # ax2.set_ylabel(r"slant depth (log $\frac{g}{cm^2}$)")
    # # ax2.grid()
    # # ax2.legend()
    # # # ax2.set_ylim([7e1, 4e4])
    # # ax2.set_xlabel(r"$\theta_{tr}$ (radians)")
    # # fig.suptitle(
    # #     r"Slant Depth over $\theta_{tr}\in\left(\frac{-\pi}{2}, \frac{\pi}{2}\right)$"
    # # )
    # plt.show()

    # plt.clf()
    # plt.hist(ers)
    # plt.show()
