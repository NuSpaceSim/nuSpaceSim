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

from typing import Any, Callable, Union

import numpy as np
from numpy.typing import NDArray

from ...config import NssConfig, MonoSpectrum, PowerSpectrum, FileSpectrum
from ...utils import decorators
from .local_plots import spectra_histogram


@decorators.nss_result_plot(spectra_histogram)
@decorators.nss_result_store("log_e_nu")
def energy_spectra(
    N: int,
    spectra: Union[MonoSpectrum, PowerSpectrum, FileSpectrum, Callable],
    *args,
    **kwargs,
) -> NDArray[Any]:
    """Energy Spectra of thrown Neutrinos"""

    if isinstance(spectra, MonoSpectrum):
        return np.full(shape=(N), fill_value=spectra.log_nu_tau_energy)

    if isinstance(spectra, PowerSpectrum):
        delta = 10 ** spectra.upper_bound - 10 ** spectra.lower_bound
        return 10 ** spectra.lower_bound + delta * np.random.power(
            spectra.index, size=N
        )

    if isinstance(spectra, Callable):
        return spectra(*args, size=N, **kwargs)

    else:
        raise RuntimeError(f"Spectra type not recognized {type(Spectra)}")


class Spectra:
    """Energy Spectra of thrown Neutrinos"""

    def __init__(self, config: NssConfig):
        self.config = config

    def __call__(self, N, *args, **kwargs):
        es = energy_spectra(N, self.config.simulation.spectrum, *args, **kwargs)
        print(np.min(es), np.max(es))
        print(np.log10(np.min(es)), np.log10(np.max(es)))
        return np.log10(es)
