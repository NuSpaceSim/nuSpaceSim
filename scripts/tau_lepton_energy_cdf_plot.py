import time
from importlib.resources import as_file, files

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from nuspacesim import NssConfig
from nuspacesim.simulation.taus import Taus

cfg = NssConfig()
taus = Taus(cfg)

N = int(1e6)
BINS = 1000
log_e_nu = 10.0
betas = np.radians(np.arange(19.0, 21.1, 0.5))

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
fig.suptitle(f"Tau lepton CDF sample Histogram.\n Log_10 E_nu = {log_e_nu} GeV")

u = np.random.uniform(size=N)


def tau_e_hist(beta, logenu):
    barray = np.full(N, beta)
    enuarray = np.full(N, logenu)

    tau_energy = taus.tau_energy(betas=barray, log_e_nu=enuarray, u=u)
    efrac = tau_energy / (10 ** log_e_nu)

    ax.set_xlabel("Sampled (E_tau / E_nu)")
    ax.set_ylabel("Counts")
    ax.hist(
        efrac,
        bins=BINS,
        histtype="step",
        label=f"Beta Angle: {np.degrees(beta):3.1f} degrees.",
        log=True,
    )

    ax1.set_xlabel("Sampled Log_10(E_tau / E_nu)")
    ax1.set_ylabel("Counts")
    ax1.hist(
        np.log10(efrac),
        bins=BINS,
        histtype="step",
        label=f"Beta Angle: {np.degrees(beta):3.1f} degrees.",
        log=True,
    )


for b in betas:
    tau_e_hist(b, log_e_nu)


ax.legend()
plt.show()
