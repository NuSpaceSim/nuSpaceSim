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

"""Decorator functions for nuspacesim simulation modules.

.. autosummary::
   :toctree:
   :recursive:

   nss_result_plot
   nss_result_store
   nss_result_store_scalar

"""


__all__ = ["nss_result_store", "nss_result_store_scalar", "nss_result_plot"]

from functools import wraps
from typing import Callable, Iterable, Union


def nss_result_store(*names):
    r"""Store result columns in nuspacesim.ResultsTable

    This decorator function allows easy decoration of simulation objects to store
    result arrays in named ResultsTable columns.

    Parameters
    ----------
    names: list
        list of names to give result columns.

    Examples
    --------
    In region_geometry.py

    >>> @nss_result_store("beta_rad")
    >>> def __call__(self, numtrajs):
    >>>     self.throw(numtrajs)
    >>>     return self.beta_rad()

    ... in calling code

    >>> sim = ResultsTable(config)
    >>> beta_tr = geom(config.simulation.N, store=sim)
    """

    def decorator_store(func):
        @wraps(func)
        def store_f(*args, store=None, **kwargs):
            values = func(*args, **kwargs)
            if store is not None:
                if isinstance(values, tuple):
                    assert len(names) == len(values)
                    store(names, [*values])
                else:
                    assert len(names) == 1
                    store(names, [values])
            return values

        return store_f

    return decorator_store


def nss_result_store_scalar(names, comments):
    r"""Store scalar results in nuspacesim.ResultsTable

    This decorator function allows easy decoration of simulation objects to store
    result scalars in the metadata section of nuspacesim.ResultsTable.

    Parameters
    ----------
    names: list[str]
        list of names to give result values.
    comments: list[str]
        list of comments to provide with corresponding result values.

    Examples
    --------
    In region_geometry.py

    >>> @decorators.nss_result_store_scalar(
    >>>    ["mcint", "mcintgeo", "nEvPass"],
    >>>    [
    >>>        "MonteCarlo Integral",
    >>>        "MonteCarlo Integral, GEO Only",
    >>>        "Number of Passing Events",
    >>>    ],
    >>> )
    >>> def mcintegral(self, numPEs, costhetaCh, tauexitprob):
    >>> ...

    ... in calling code

    >>> sim = ResultsTable(config)
    >>> beta_tr = geom.mcintegral(numPEs, costhetaCh, tauexitprob, store=sim)

    """

    assert len(names) == len(comments)

    def decorator_store_meta(func):
        @wraps(func)
        def store_f(*args, store=None, **kwargs):

            values = func(*args, **kwargs)

            if store is not None:
                if isinstance(values, tuple):
                    assert len(names) == len(values)
                    for name, value, comment in zip(names, values, comments):
                        store.add_meta(name, value, comment)
                else:
                    assert len(names) == 1
                    store(*names, *values, *comments)
            return values

        return store_f

    return decorator_store_meta


def nss_result_plot(*plot_fs):
    r"""Plot results of function.

    This decorator function allows easy decoration of simulation objects to store
    result scalars in the metadata section of nuspacesim.ResultsTable.

    Parameters
    ----------
    plot_f: Callable
        plotting function


    Examples
    --------
    In region_geometry.py

    >>> def betas_histogram(numtrajs, betas):
    >>>     plt.hist(betas, 50, alpha=0.75)
    >>>     plt.xlabel("beta_tr (radians)")
    >>>     plt.ylabel("frequency (counts)")
    >>>     plt.title(f"Histogram of {betas.size} Beta Angles")
    >>>     plt.show()
    >>>
    >>> @decorators.nss_result_plot(betas_histogram)
    >>> def __call__(self, numtrajs):
    >>>     self.throw(numtrajs)
    >>>     return self.beta_rad()

    ... in calling code

    >>> beta_tr = geom(plot=True)

    """

    def decorator_plot(func):
        from .plot_function_registry import registry

        for plotname in map(lambda p: p.__name__, plot_fs):
            registry.add(plotname)

        @wraps(func)
        def wrapper_f(
            *args, plot: Union[None, str, Iterable, Callable] = None, **kwargs
        ):
            values = func(*args, **kwargs)
            if isinstance(plot, str):
                for plotf in plot_fs:
                    if plotf.__name__ == plot:
                        plotf(args, values)
            elif callable(plot):
                plot(args, values)
            elif isinstance(plot, Iterable):
                if all(isinstance(p, str) for p in plot):
                    for plotf in plot_fs:
                        if plotf.__name__ in plot:
                            plotf(args, values)
                elif all(callable(p) for p in plot):
                    for plotf in plot:
                        plotf(args, values)
            return values

        return wrapper_f

    return decorator_plot


def nss_result_plot_from_file(sim, inputs, outputs, plotfs, plot):

    f_input = tuple() if inputs is None else tuple(sim[i] for i in inputs)
    results = tuple() if outputs is None else tuple(sim[o] for o in outputs)

    @nss_result_plot(*plotfs)
    def f(*args, **kwargs):
        return results

    f(None, *f_input, plot=plot)


def ensure_plot_registry(*plot_fs):
    def decorator_plot(func):
        from .plot_function_registry import registry

        for plotname in map(lambda p: p.__name__, plot_fs):
            registry.add(plotname)
        return func

    return decorator_plot
