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
"""``nuspacesim stream`` -- streaming, distributed optical Monte-Carlo.

.. currentmodule:: nuspacesim.apps.stream
"""

import click

from ..config import config_from_toml
from ..results_table import output_filename
from .utils import parse_cloud_options, parse_spectra_options


@click.command()
@click.option(
    "-o", "--output", type=click.Path(exists=False), default=None, help="Output file."
)
@click.option(
    "-n",
    "--no-result-file",
    is_flag=True,
    help="Do not save the results to an output file.",
)
@click.option(
    "--rel-unc",
    type=float,
    default=None,
    help="Target relative uncertainty (e.g. 0.01 for 1%). Overrides "
    "[simulation.streaming].rel_unc_target. 0 runs to the max-thrown backstop.",
)
@click.option(
    "--reservoir-size",
    type=int,
    default=None,
    help="Number of representative events (k) to keep and write to disk. "
    "Overrides [simulation.streaming].reservoir_size.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Trajectories thrown per batch (fixed, reproducible schedule). If "
    "omitted, the batch size adapts over the run to maximize throughput.",
)
@click.option(
    "--monospectrum",
    type=float,
    default=None,
    help="Mono Energetic Spectrum Log Energy.",
)
@click.option(
    "--powerspectrum",
    nargs=3,
    type=click.Tuple([float, float, float]),
    default=None,
    help="Power Spectrum index, lower_bound, upper_bound.",
)
@click.option("--nocloud", is_flag=True, default=None, help="No Cloud Model. [Default]")
@click.option(
    "--monocloud",
    type=float,
    default=None,
    help="Uniform (mono) Height Cloud Model (km).",
)
@click.option(
    "--pressuremapcloud",
    type=click.DateTime(["%m", "%B", "%b"]),
    default=None,
    help="Pressure Map Cloud Model month (name, abbreviation, or number).",
)
@click.argument(
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.argument("count", type=float, default=0.0)
def stream(
    config_file: str,
    count: float,
    output: str,
    no_result_file: bool,
    rel_unc: float,
    reservoir_size: int,
    batch_size: int,
    monospectrum: float,
    powerspectrum: click.Tuple,
    nocloud: bool,
    monocloud: float,
    pressuremapcloud: click.DateTime,
) -> None:
    """Streaming, distributed optical Monte-Carlo simulation.

    Throws neutrino events in batches across a process-based dask cluster,
    accumulates the optical Monte-Carlo integral as an exact converging sketch
    with live statistics, and stops when the integral's relative uncertainty
    reaches the configured target. Only the weighted reservoir of representative
    events is materialized to disk.

    \f

    Parameters
    ----------
    config_file : str
        TOML configuration file.
    count : float, optional
        Upper bound on thrown trajectories (the backstop). **Defaults to
        infinity** -- the run is driven by the relative-uncertainty target (or a
        spacebar interrupt), not a fixed count. Provide a value to cap it.
    output : str, optional
        Output FITS file for the reservoir + integral metadata.
    no_result_file : bool, optional
        Do not write the results file.
    rel_unc : float, optional
        Target relative uncertainty; overrides the config value.
    reservoir_size : int, optional
        Reservoir size k; overrides the config value.

    Examples
    --------
    Run until the integral reaches 1% relative uncertainty (no thrown cap)::

        nuspacesim stream --rel-unc 0.01 sample_input_file.toml

    On an interactive terminal, press the **spacebar** to end early and finalize
    the partial result.
    """
    config = config_from_toml(config_file)

    if rel_unc is not None:
        config.simulation.streaming.rel_unc_target = rel_unc
    if reservoir_size is not None:
        config.simulation.streaming.reservoir_size = reservoir_size

    overwrite_spectrum = parse_spectra_options(monospectrum, powerspectrum)
    if overwrite_spectrum:
        config.simulation.spectrum = overwrite_spectrum

    overwrite_cloud = parse_cloud_options(nocloud, monocloud, pressuremapcloud)
    if overwrite_cloud:
        config.simulation.cloud_model = overwrite_cloud

    # Default backstop is infinity: the precision target (or a spacebar
    # interrupt) drives the run, not a fixed thrown count.
    max_thrown = float("inf") if count == 0.0 else count

    # No --batch-size -> adaptive throughput-tuned sizing.
    batch = batch_size if batch_size is not None else "adaptive"

    # Imported here so `nuspacesim --help` stays light (defers dask/rich).
    from ..simulation.streaming import compute_streaming

    output = output_filename(output)
    simulation = compute_streaming(
        config,
        verbose=True,
        output_file=output,
        max_thrown=max_thrown,
        batch_size=batch,
    )

    if not no_result_file:
        simulation.write(output, format="fits", overwrite=True)
