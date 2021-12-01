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

from matplotlib import pyplot as plt
import numpy as np


def taus_scatter(inputs, results, *args, **kwargs):
    r"""Plot some scatterplots"""

    tau_self, betas, log_e_nu = inputs
    tauBeta, tauLorentz, showerEnergy, tauExitProb = results

    color = "c"
    alpha = 0.1 / np.log10(betas.size)

    fig, ax = plt.subplots(2, 2, sharex=True, constrained_layout=True)

    ax[0, 0].scatter(x=np.degrees(betas), y=np.log10(tauBeta), c=color, alpha=alpha)
    ax[0, 0].set_xlabel("β")
    ax[0, 0].set_ylabel("log(τ_β)")
    ax[0, 0].set_title("β vs log(τ_β)")

    ax[0, 1].scatter(x=np.degrees(betas), y=np.log10(tauLorentz), c=color, alpha=alpha)
    ax[0, 1].set_xlabel("β")
    ax[0, 1].set_ylabel("log(τ_Lorentz)")
    ax[0, 1].set_title("β vs log(τ_Lorentz)")

    ax[1, 0].scatter(
        x=np.degrees(betas), y=np.log10(showerEnergy), c=color, alpha=alpha
    )
    ax[1, 0].set_xlabel("β")
    ax[1, 0].set_ylabel("log(E_shower (100 PeV))")
    ax[1, 0].set_title("β vs log(E_shower)")

    ax[1, 1].scatter(x=np.degrees(betas), y=np.log10(tauExitProb), c=color, alpha=alpha)
    ax[1, 1].set_xlabel("β")
    ax[1, 1].set_ylabel("log(PExit(τ))")
    ax[1, 1].set_title("β vs log(Exit Probability)")

    fig.suptitle("Tau interaction properties vs. β_tr Angles")
    plt.show()


def taus_histogram(inputs, results, *args, **kwargs):
    r"""Plot some histograms"""

    tau_self, betas, log_e_nu = inputs
    tauBeta, tauLorentz, showerEnergy, tauExitProb = results

    color = "c"
    alpha = 1

    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    ax[0, 0].hist(tauBeta, 100, log=True, facecolor=color, alpha=alpha)
    ax[0, 0].set_xlabel("log(τ_β)")
    ax[0, 1].hist(tauLorentz, 100, log=True, facecolor=color, alpha=alpha)
    ax[0, 1].set_xlabel("log(τ_Lorentz)")
    ax[1, 0].hist(showerEnergy, 100, log=True, facecolor=color, alpha=alpha)
    ax[1, 0].set_xlabel("log(showerEnergy)")
    ax[1, 1].hist(tauExitProb, 100, log=True, facecolor=color, alpha=alpha)
    ax[1, 1].set_xlabel("log(PExit(τ))")

    fig.suptitle("Tau interaction property Histograms")
    plt.show()


def taus_pexit(inputs, results, *args, **kwargs):
    tau_self, betas, log_e_nu = inputs
    tauBeta, tauLorentz, showerEnergy, tauExitProb = results

    color = "c"
    alpha = 0.1 / np.log10(betas.size)

    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    ax[0].scatter(
        x=np.degrees(betas),
        y=np.log10(tauExitProb),
        s=1,
        c=color,
        marker=".",
        alpha=alpha,
    )
    ax[0].set_xlabel("β")
    ax[0].set_ylabel("log(PExit(τ))")
    ax[0].set_title("β vs Exit Probability.")

    ax[1].hist(tauExitProb, 100, log=True, facecolor=color)
    ax[1].set_ylabel("log(frequency)")
    ax[1].set_xlabel("PExit(τ)")

    fig.suptitle("Tau Pexit")
    plt.show()
