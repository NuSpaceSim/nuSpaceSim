from nuspacesim.simulation.eas_optical.quadeas import *

import numpy as np
import quadpy as qp

if __name__ == "__main__":

    # # import timeit

    earth_radius = 6378.14
    zmax = 65.0

    # fmt: off
    wmean = np.array(
        [212.5, 237.5, 262.5, 287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 437.5, 462.5,
         487.5, 512.5, 537.5, 562.5, 587.5, 612.5, 637.5, 662.5, 687.5, 712.5, 737.5,
         762.5, 787.5, 812.5, 837.5, 862.5, 887.5, ])
    # fmt: on

    alt_dec = 1.0
    z_obs = 525.0
    z_max = 65.0
    beta_tr = np.radians(1)
    np.set_printoptions(linewidth=256)

    s0 = 0.079417252568371  # s0 = np.exp(-575/227) <-- e2hill == 0: Shower too young.
    s1 = 1.899901462640018  # shower age when greisen_particle_count(s) == 1.0.

    # print(
    #     slant_depth(
    #         1,
    #         np.linspace(1, 1.001, 100),
    #         (np.pi / 2) - np.radians(np.linspace(1, 80, 100)),
    #         epsrel=1e-2,
    #     )
    # )

    z_lo = altitude_at_shower_age(s0, alt_dec, beta_tr)
    z_hi = altitude_at_shower_age(s1, alt_dec, beta_tr)

    l_lo = length_along_prop_axis(alt_dec, z_lo, beta_tr, earth_radius)
    l_hi = length_along_prop_axis(alt_dec, z_hi, beta_tr, earth_radius)

    print(s0, s1)
    print(z_lo, z_hi)
    print(l_lo, l_hi)

    for i, b in enumerate(np.radians(np.linspace(1, 89, 10))):
        print(
            i,
            np.degrees(b),
            altitude_at_shower_age(s0, alt_dec, b),
            altitude_at_shower_age(s1, alt_dec, b),
        )

    # subl = np.linspace(l_lo, l_hi, 20)

    # for l in subl:
    #     for w in wmean:

    #         intf((l,w,1,1), alt_dec, beta_tr, z_max, z_obs, earth_radius)

    # psum = photon_density(alt_dec, beta_tr, z_max, z_obs, earth_radius)
    # print(f"{psum[0]:e}, {psum[1]:e}")

    # Lmax = length_along_prop_axis(alt_dec, z_max, beta_tr, earth_radius)
    # print("Lmax", Lmax)
    # l = np.linspace(0, Lmax, 10)
    # print("l", l)
    # z = altitude_along_prop_axis(l, alt_dec, beta_tr, earth_radius)
    # Zmax = altitude_along_prop_axis(Lmax, alt_dec, beta_tr, earth_radius)
    # print("z", z)
    # # intf(z, 1.0, wmean)
    # print(ozone(z))
    # print(differential_ozone(z))
    # print(poly_differential_ozone(z))

    # ZonZ = ozone_content(l, Lmax, alt_dec, beta_tr, earth_radius)
    # print(ZonZ)
    # TrOz = ozone_losses(ZonZ, wmean)
    # print(TrOz)

    # print("_z", subz)
    # X = slant_depth(subz, 65., (np.pi / 2) - beta_tr, epsrel=1e-3)[0]
    # print(X)
    # TrRayl = sokolsky_rayleigh_scatter(X, wmean)
    # print(TrRayl)

    # print(elterman_mie_aerosol_scatter(subz, wmean, beta_tr, earth_radius))
