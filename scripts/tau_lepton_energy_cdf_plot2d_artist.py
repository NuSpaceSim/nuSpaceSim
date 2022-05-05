import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from nuspacesim import NssConfig
from nuspacesim.simulation.taus import Taus
from nuspacesim.utils.grid import NssGrid


def gen_tau_e_frac():
    cfg = NssConfig()
    taus = Taus(cfg)

    N = int(1e5)
    log_e_nus = np.arange(6.0, 12.1, 0.1)
    beta_deg = np.arange(1.0, 32.0, 0.1)
    betas = np.radians(beta_deg)
    idxs = np.arange(0, N)
    B, L = np.meshgrid(betas, log_e_nus)
    B_N = np.full((N, *B.shape), B)
    L_N = np.full((N, *L.shape), L)
    tau_energy = taus.tau_energy(betas=B_N, log_e_nu=L_N)
    tau_efrac = np.log10(tau_energy / 10 ** L_N)
    grid = NssGrid(
        tau_efrac,
        [idxs, log_e_nus, beta_deg],
        ["index", "log_e_nu", "beta_deg"],
    )
    grid.write(
        "tau_lepton_sampled_log_efrac_grid.h5", path="/", overwrite=True, format="hdf5"
    )


gen_tau_e_frac()


# BINS = 100
# grid = NssGrid.read("tau_lepton_sampled_log_efrac_grid.h5")
# N = grid["index"].size
# B_N = np.full((N, *grid["beta_deg"].shape), grid["beta_deg"])
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ims = [
#     [
#         ax.hist2d(
#             np.ravel(grid[:, i, :]),
#             np.ravel(B_N),
#             bins=(BINS, *grid["beta_deg"].shape),
#             cmin=1,
#             cmap="jet",
#             animated=(i > 0),
#         )[3],
#     ]
#     for i in range(grid["log_e_nu"].size)
# ]
#
# anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
#
#
# plt.show()
