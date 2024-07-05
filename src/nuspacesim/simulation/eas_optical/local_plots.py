# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from matplotlib import pyplot as plt
#from shower_properties import greisen, x0, Eprime, x
import sys
import os

from .shower_properties import greisen, Eprime, x, x0, gaisser_hillas, n, xm, λ

sys.path.append(os.path.abspath('~/nuspacesim/simulation/eas_optical/shower_properties.py'))
from ...utils.plots import hist2d


def eas_optical_density(inputs, results, *args, **kwargs):
    r"""Plot some density plots"""

    eas_cls, betas, altDec, showerEnergy, *_ = inputs
    numPEs, costhetaChEff = results

    # Issue 94 == https://github.com/NuSpaceSim/nuSpaceSim/issues/94
    # Include only events with Npe >=photo_electron_threshold
    valid = numPEs >= eas_cls.config.detector.optical.photo_electron_threshold
    betas = betas[valid]
    altDec = altDec[valid]
    showerEnergy = showerEnergy[valid]
    numPEs = numPEs[valid]
    costhetaChEff = costhetaChEff[valid]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    hist2d(fig, ax[0, 0], np.degrees(betas), numPEs, "β", "numPEs", cmap="plasma")
    hist2d(
        fig,
        ax[1, 0],
        np.degrees(betas),
        costhetaChEff,
        "β",
        "cos(θ_chEff)",
        cmap="plasma",
    )

    hist2d(
        fig, ax[0, 1], altDec, numPEs, "decay altitude (km)", "numPEs", cmap="plasma"
    )

    hist2d(
        fig,
        ax[1, 1],
        altDec,
        costhetaChEff,
        "decay altitude (km)",
        "cos(θ_chEff)",
        cmap="plasma",
    )

    hist2d(
        fig,
        ax[0, 2],
        showerEnergy,
        numPEs,
        "showerEnergy (100 PeV)",
        "numPEs",
        cmap="plasma",
    )

    hist2d(
        fig,
        ax[1, 2],
        showerEnergy,
        costhetaChEff,
        "showerEnergy (100 PeV)",
        "cos(θ_chEff)",
        cmap="plasma",
    )

    fig.suptitle("EAS Optical Cherenkov properties.")
    plt.show()


def eas_optical_histogram(inputs, results, *args, **kwargs):
    r"""Plot some histograms"""

    eas_cls, *_ = inputs
    numPEs, costhetaChEff = results

    # Issue 94 == https://github.com/NuSpaceSim/nuSpaceSim/issues/94
    # Include only events with Npe >=photo_electron_threshold
    valid = numPEs >= eas_cls.config.detector.optical.photo_electron_threshold
    numPEs = numPEs[valid]
    costhetaChEff = costhetaChEff[valid]

    color = "salmon"
    alpha = 1

    fig, ax = plt.subplots(2, 1, constrained_layout=True)

    ax[0].hist(numPEs, 100, log=True, facecolor=color, alpha=alpha)
    ax[0].set_xlabel("log(numPEs)")

    ax[1].hist(costhetaChEff, 100, log=True, facecolor=color, alpha=alpha)
    ax[1].set_xlabel("log(cos(θ_chEff))")

    fig.suptitle("EAS Optical Cherenkov property Histograms")
    plt.show()
def greisen_plot(x,y):
    x=np.linspace(1,1500,1500)
y=greisen(x,Eprime)
plt.plot (x,y)
plt.title("Greisen")
plt.xlabel("Depth (g/cm^2)")
plt.ylabel("N(x)")
plt.show()

def gaisser_hillas_plot(x,n,x0,xm,λ):
    x = np.linspace(x0, 1500, 1500)
y=gaisser_hillas(x,n,x0,xm,λ)
plt.plot(x, y)
plt.title("Gaisser Hillas")
plt.xlabel("Depth (g/cm^2)")
plt.ylabel("N(x)")
plt.show()

def greisen_gaisser_hillas_overview():
    x = np.linspace(x0, 1500, 1500)

fig, ax = plt.subplots(3, 1,figsize=(7, 7), constrained_layout=True)

ax[0].plot(x, greisen(x,Eprime))
ax[0].set_title("Greisen")
ax[0].set_xlabel("Depth (g/cm^2)")
ax[0].set_ylabel("N(x)")

ax[1].plot(x, gaisser_hillas(x, n, x0, xm, λ))
ax[1].set_title("Gaisser Hillas")
ax[1].set_xlabel("Depth (g/cm^2)")
ax[1].set_ylabel("N(x)")

ax[2].plot(x, greisen(x,Eprime))
ax[2].plot(x, gaisser_hillas(x, n, x0, xm, λ))
ax[2].set_title("Greisen and Gaisser Hillas")
ax[2].set_xlabel("Depth (g/cm^2)")
ax[2].set_ylabel("N(x)")

plt.show()