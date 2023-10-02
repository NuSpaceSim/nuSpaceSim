# The Clear BSD License
#
# Copyright (c) 2022 Alexander Reustle and the NuSpaceSim Team
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

from nuspacesim.utils.grid import NssGrid, grid_concatenate

pexit_v0 = NssGrid.read(
    "nupyprop_tables/nu2tau_pexit.0.h5", path="pexit_regen", format="hdf5"
)
pexit_v1 = NssGrid.read(
    "nupyprop_tables/nu2tau_pexit.1.h5", path="pexit_regen", format="hdf5"
)
new_pexit = grid_concatenate(pexit_v1[:, :9], pexit_v0, 1)
new_pexit.write(
    "nupyprop_tables/nu2tau_pexit.2.h5",
    path="/pexit_regen",
    overwrite=True,
    format="hdf5",
)

cdf_v0 = NssGrid.read("nupyprop_tables/nu2tau_cdf.0.h5", path="/", format="hdf5")
cdf_v1 = NssGrid.read("nupyprop_tables/nu2tau_cdf.1.h5", path="/", format="hdf5")
new_cdf = grid_concatenate(cdf_v1[:, :9, :], cdf_v0, 1)
new_cdf.write(
    "nupyprop_tables/nu2tau_cdf.2.h5",
    overwrite=True,
    format="hdf5",
)
