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


def spectra_histogram(inputs, results, *args, **kwargs):
    r"""Plot some histograms"""

    N, spectrum = inputs
    log_e_nu = results
    plotting_opts = kwargs.get("kwargs")

    color = "g"
    fig = plt.figure(figsize=plotting_opts.get("figsize"), constrained_layout=True)
    ax = fig.add_subplot(211)
    ax.hist(log_e_nu, 100, log=False, facecolor=color)
    ax.set_xlabel(f"log(E_nu) of {N} events")

    ax = fig.add_subplot(212)
    ax.hist(log_e_nu, 100, log=True, facecolor=color)
    ax.set_xlabel(f"log(E_nu) of {N} events")

    fig.suptitle(f"Energy Spectra Histogram, Log(E_nu)\n {spectrum}")
    if plotting_opts.get("pop_up") is True:
        plt.show()
    if plotting_opts.get("save_to_file") is True:
        fig.savefig(
            plotting_opts.get("filename")
            + "_spectra_histogram."
            + plotting_opts.get("save_as")
        )
