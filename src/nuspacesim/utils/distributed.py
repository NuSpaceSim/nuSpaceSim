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

"""Dask cluster lifecycle helpers shared across the simulation pipeline.

Pure infrastructure -- nothing here knows about EAS physics. Extracted from
``eas_optical.cphotang`` so consumers (``compute``, future streaming drivers)
don't import a physics module to obtain a cluster.
"""

import threading

from dask.distributed import Client, LocalCluster

__all__ = ["BackgroundCluster"]


class BackgroundCluster:
    """A process-based dask ``LocalCluster`` spun up in a background thread.

    A simulation runs exactly one :meth:`CphotAng.__call__`, so the cluster's
    whole life fits inside that single call -- but its ~2s process spawn would
    otherwise be paid serially right when the EAS optical stage starts. Creating
    it here, at the top of the pipeline, lets the worker spawn overlap the
    geometry/spectra/tau/decay stages (which release the GIL in their numpy/
    C work, so the spawning thread makes real progress). By the time
    :meth:`client` is called the workers are warm.

    Own the lifecycle from the caller: construct early, pass :meth:`client` into
    the EAS optical call, then :meth:`close`. Construction failures are deferred
    and re-raised from :meth:`client` so the caller's stack frame sees them.
    """

    def __init__(self):
        self._holder = {}
        self._error = None
        self._thread = threading.Thread(target=self._spawn, daemon=True)
        self._thread.start()

    def _spawn(self):
        try:
            cluster = LocalCluster(processes=True)
            self._holder["cluster"] = cluster
            self._holder["client"] = Client(cluster)
        except BaseException as exc:
            self._error = exc

    def client(self):
        """Block until the cluster is up and return its :class:`Client`."""
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._holder["client"]

    def close(self):
        """Tear the cluster down (idempotent; safe even if spawn failed)."""
        self._thread.join()
        client = self._holder.get("client")
        cluster = self._holder.get("cluster")
        if client is not None:
            client.close(timeout=2)
        if cluster is not None:
            cluster.close(timeout=2)
        self._holder.clear()
