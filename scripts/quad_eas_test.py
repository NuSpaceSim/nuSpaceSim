from nuspacesim.simulation.eas_optical.quadeas import *

import numpy as np

# import quadpy as qp

if __name__ == "__main__":

    earth_radius = 6378.14
    zmax = 65.0

    wave = np.arange(200.0, 901.0, 25)

    alt_dec = 1.0
    z_obs = 525.0
    z_max = 65.0
    beta_tr = np.radians(1)
    np.set_printoptions(linewidth=256)

    # s0 = 0.079417252568371  # s0 = np.exp(-575/227) <-- e2hill == 0: Shower too young.
    # s1 = 1.899901462640018  # shower age when greisen_particle_count(s) == 1.0.

    # z_lo = altitude_at_shower_age(s0, alt_dec, beta_tr)
    # z_hi = altitude_at_shower_age(s1, alt_dec, beta_tr)

    # l_lo = length_along_prop_axis(alt_dec, z_lo, beta_tr, earth_radius)
    # l_hi = length_along_prop_axis(alt_dec, z_hi, beta_tr, earth_radius)

    psum = photon_density(alt_dec, beta_tr, z_max, z_obs, earth_radius)
    # print(f"{psum:e}")
    print(f"{psum[0]:e}, {psum[1]:e}")
