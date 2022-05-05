import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from nuspacesim.utils.grid import NssGrid

grid = NssGrid.read("tau_lepton_sampled_log_efrac_grid.h5")


N = grid["index"].size
B_N = np.full((N, *grid["beta_deg"].shape), grid["beta_deg"])
EfracBins = 50
BINS = (np.linspace(-5.0, 0.0, EfracBins, endpoint=True), np.arange(0, 33, 1))

fig = plt.figure(dpi=150, constrained_layout=True)
ax = fig.add_subplot(111)
H, Xe, Ye, Img = ax.hist2d(
    np.ravel(grid[:, 0, :]),
    np.ravel(B_N),
    bins=BINS,
    cmin=1,
    cmap="jet",
    norm=LogNorm(),
)
ax.set_xlabel(r"$\log_{10}(\frac{E_\tau}{E_\nu})$")
ax.set_ylabel("Earth Emergence Angle (degrees)")
title_text = ax.set_title("")
author_text = ax.text(-2.0, -6.0, "Alexander Reustle (GSFC/NuSpaceSim)", fontsize=8)
cbar = fig.colorbar(Img, ax=ax, pad=0.0)
cbar.set_label("Counts")


def update_f(frame):
    i, lenu = frame
    h, *_ = np.histogram2d(np.ravel(grid[:, i, :]), np.ravel(B_N), bins=BINS)
    h[h < 1] = None

    Img.set_array(np.ravel(h.T))
    title_text.set_text(
        r"NuPyProp $\tau$ Energy fraction histogram: $\log_{10}E_\nu$ = "
        + f"{lenu:.1f}"
    )

    return (
        title_text,
        Img,
    )


anim = animation.FuncAnimation(
    fig, update_f, frames=enumerate(grid["log_e_nu"]), interval=200, repeat_delay=1000
)


def progress(i, n):
    if i % 10 == 0:
        print(f"Frame {i}/{n}")


# plt.show()

anim.save(
    "NuPyProp_NuSpaceSim_Tau_Efrac_sample_histogram.gif",
    dpi=300,
    fps=5,
    progress_callback=progress,
)
anim.save(
    "NuPyProp_NuSpaceSim_Tau_Efrac_sample_histogram.mp4",
    dpi=300,
    fps=5,
    progress_callback=progress,
)
