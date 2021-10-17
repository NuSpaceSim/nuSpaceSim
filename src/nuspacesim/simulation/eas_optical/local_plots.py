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


def eas_optical_scatter(inputs, results, *args, **kwargs):
    r"""Plot some scatterplots"""

    eas_self, betas, altDec, showerEnergy = inputs
    numPEs, costhetaChEff = results

    color = "salmon"
    alpha = 1 / np.log10(betas.size)

    fig, ax = plt.subplots(nrows=2, ncols=3)

    ax[0, 0].scatter(x=np.degrees(betas), y=numPEs, c=color, alpha=alpha)
    ax[0, 0].set_xlabel("β")
    ax[0, 0].set_ylabel("numPEs")
    ax[0, 0].set_title("β vs numPEs")
    ax[1, 0].scatter(x=np.degrees(betas), y=costhetaChEff, c=color, alpha=alpha)
    ax[1, 0].set_xlabel("β")
    ax[1, 0].set_ylabel("cos(θ_chEff)")
    ax[1, 0].set_title("β vs cos(θ_chEff)")

    ax[0, 1].scatter(x=altDec, y=numPEs, c=color, alpha=alpha)
    ax[0, 1].set_xlabel("decay altitude (KM)")
    ax[0, 1].set_ylabel("numPEs")
    ax[0, 1].set_title("altitude vs numPEs")
    ax[1, 1].scatter(x=altDec, y=costhetaChEff, c=color, alpha=alpha)
    ax[1, 1].set_xlabel("decay altitude (KM)")
    ax[1, 1].set_ylabel("cos(θ_chEff)")
    ax[1, 1].set_title("altitude vs cos(θ_chEff)")

    ax[0, 2].scatter(x=showerEnergy, y=numPEs, c=color, alpha=alpha)
    ax[0, 2].set_xlabel("showerEnergy (100 PeV)")
    ax[0, 2].set_ylabel("numPEs")
    ax[0, 2].set_title("showerEnergy vs numPEs")
    ax[1, 2].scatter(x=showerEnergy, y=costhetaChEff, c=color, alpha=alpha)
    ax[1, 2].set_xlabel("showerEnergy (100 PeV)")
    ax[1, 2].set_ylabel("cos(θ_chEff)")
    ax[1, 2].set_title("showerEnergy vs cos(θ_chEff)")

    fig.suptitle("EAS Optical Cherenkov properties.")
    plt.show()
