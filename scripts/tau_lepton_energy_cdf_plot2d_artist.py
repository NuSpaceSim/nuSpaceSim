import numpy as np
from matplotlib import animation, cm
from matplotlib import pyplot as plt

from nuspacesim import NssConfig
from nuspacesim.simulation.taus import Taus

cfg = NssConfig()
taus = Taus(cfg)

N = int(1e5)
BINS = 100
log_e_nus = np.arange(6.0, 12.1, 0.5)
betas = np.radians(np.arange(1.0, 22.1, 0.25))

fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
ax = fig.add_subplot(111)
# ax2 = fig.add_subplot(122)


def gen_tau_e(betas, logenu):

    barray = np.full((N, *betas.shape), betas)
    enuarray = np.full((N, *betas.shape), logenu)

    tau_energy = taus.tau_energy(betas=barray, log_e_nu=enuarray)

    return np.degrees(barray), np.log10(tau_energy / 10 ** logenu)


def tau_e_hist3d(ax, logenu, barray, tau_e_frac):

    hist, yedges, xedges = np.histogram2d(
        x=barray.ravel(),
        y=tau_e_frac.ravel(),
        bins=(barray.shape[-1], BINS),
    )

    xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:])

    xpos = xpos.ravel() / 2.0
    ypos = ypos.ravel() / 2.0
    zpos = np.zeros_like(xpos)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = np.arcsinh(hist.ravel())
    cmap = cm.get_cmap("jet")
    rgba = [cmap((k - np.min(dz)) / np.max(dz)) for k in dz]

    ax.set_xlabel("log_10(E_tau/E_nu)")
    ax.set_ylabel("Beta")
    ax.set_zlabel("arcsinh(Count)")
    ax.set_title(f"Tau Lepton Histogram for Log(E_nu)={logenu:3.2f} GeV")

    return ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort="average")


ims = []
for i, lenu in enumerate(log_e_nus):
    b, t = gen_tau_e(betas, lenu)
    im = ax.hist2d(
        t.ravel(),
        b.ravel(),
        bins=(BINS, *betas.shape),
        cmin=1,
        cmap="jet",
        animated=True,
    )[3]
    if i == 0:
        im = ax.hist2d(
            t.ravel(), b.ravel(), bins=(BINS, *betas.shape), cmin=1, cmap="jet"
        )[3]
    # print(im)
    ims.append([im])

# print(ims)

anim = animation.ArtistAnimation(fig, ims, interval=1000, blit=True)

# barray, tau_energy = gen_tau_e(betas, log_e_nu)
# pars = tau_e_histbar_pos(tau_energy, barray)
# tau_e_hist3d(ax, log_e_nu, *pars)


plt.show()
