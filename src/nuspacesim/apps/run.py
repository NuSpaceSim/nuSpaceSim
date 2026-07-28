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
"""Command line client source code.

.. _cli:

*****
 CLI
*****

.. currentmodule:: nuspacesim.apps.cli

.. click:: nuspacesim.apps.cli:cli
   :prog: nuspacesim
   :show-nested:

"""

import click
import numpy as np

from ..compute import compute
from ..config import config_from_toml
from ..results_table import output_filename
from ..utils.plot_function_registry import registry
from .utils import (
    parse_cloud_options,
    parse_diffuse_integ_method_options,
    parse_source_info,
    parse_spectra_options,
    parse_sun_moon_cuts,
    parse_target_integ_method_options,
    read_plot_config,
)


@click.command()
@click.option(
    "-o", "--output", type=click.Path(exists=False), default=None, help="Output file."
)
@click.option(
    "-p",
    "--plot",
    type=click.Choice(list(registry), case_sensitive=False),
    multiple=True,
    default=[],
    help="Available plotting functions. Select multiple plots with multiple uses of -p",
)
@click.option(
    "-P",
    "--plotconfig",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    help="Read selected plotting functions and options from the specified INI file",
)
@click.option("--plotall", is_flag=True, help="Show all result plots.")
@click.option(
    "-w",
    "--write-stages",
    is_flag=True,
    help="Write intermediate values after each simulation stage.",
)
@click.option(
    "-n",
    "--no-result-file",
    is_flag=True,
    help="Do not save the results to an output file.",
)
@click.option(
    "--diffuse",
    is_flag=True,
    default=None,
    help="Diffuse acceptance calculations (assuming isotropic neutrino fluxes).",
)
@click.option(
    "--target",
    is_flag=True,
    default=False,
    help="Point-source acceptance calculations (quasi-parallel neutrino trajectories).",
)
@click.option(
    "--sunmooncuts",
    is_flag=True,
    default=False,
    help="Apply cuts accounting for limited observing due to the Sun and the Moon",
)
@click.option(
    "--sunmooncutparameters",
    nargs=3,
    type=click.Tuple([float, float, float]),
    default=[np.radians(-40.5), np.radians(0.0), np.radians(150.0)],
    help="Sun/Moon cut parameters. Sun elevation angle. Moon elevation angle. Moon phase angle.",
)
@click.option(
    "--numtimebins",
    type=float,
    default=1,
    help="Number of time bins. Choose 1 for instantaneous acceptance or "
    "time-integrated exposure calculations. Input a number > 1 for "
    "time-differential or time-averaged calculations.",
)
@click.option(
    "--montecarlo",
    type=float,
    default=None,
    help="Monte Carlo integration. User should specify the number of events thrown per time bin. ",
)
@click.option(
    "--targetapprox",
    is_flag=True,
    default=False,
    help="Semi-analytical calculation of point-source acceptance using simplified "
    "approximations.",
)
@click.option(
    "--cubature",
    nargs=2,
    type=click.Tuple([int, float]),
    default=None,
    help="Cubature integration over the surface of the Earth. User should specify the degree "
    "(determines the number of nodes and the number of events per node.",
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
@click.option(
    "--nocloud",
    is_flag=True,
    default=None,
    help="No Cloud Model. [Default]",
)
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
    help="Pressure Map Cloud Model (built in and included with NuSpaceSim). This map is "
    "an instance of a global cloud top pressure map sampled from a model of all days in "
    "the given month over a 10-year time period from 2011 to 2020. Data provided by the "
    "MERRA-2 dataset. "
    "User should provide a month name, abbreviation, or number. ",
)
@click.option(
    "--sourceinfo",
    nargs=5,
    type=click.Tuple([float, float, str, str, float]),
    default=[np.radians(0.0), np.radians(0.0), "2022-06-02T01:00:00", "isot", 86400],
    help="Source information. RA, Dec., observation date, date format, amount of observation time (s).",
)
@click.argument(
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.argument("count", type=float, default=0.0)
def run(
    config_file: str,
    count: float,
    no_result_file: bool,
    output: str,
    plot: list,
    plotconfig: str,
    plotall: bool,
    write_stages: bool,
    sunmooncuts: bool,
    sunmooncutparameters: click.Tuple,
    diffuse: bool,
    target: bool,
    numtimebins: float,
    montecarlo: float,
    targetapprox: bool,
    cubature: click.Tuple,
    monospectrum: float,
    powerspectrum: click.Tuple,
    nocloud: bool,
    monocloud: float,
    pressuremapcloud: click.DateTime,
    sourceinfo: click.Tuple,
) -> None:
    """Perform the full nuspacesim simulation.

    Main Simulator for nuspacesim.  Given a TOML configuration file, and
    optionally a count of simulated nutrinos, runs nutrino simulation.

    \f

    Parameters
    ----------
    config_file: str
        TOML configuration file for particular simulation particular.
    count : int, optional
        Number of thrown trajectories. Optionally override value in config_file.
    spectrum_type : str, optional
        Type of
    output: str, optional
        Name of the output file holding the simulation results.
    plot: list, optional
        Plot the simulation results.
    plotconfig: str, optional
        INI file to select plots for each module, as well as to specifiy global plot settings.
    plotall: bool, optional
        Plot all the available the simulation results plots.
    no_result_file: bool, optional
        Disables writing results to output files.


    Examples
    --------
    Command line usage of the run command may work with the following invocation.

    `nuspacesim run sample_input_file.toml 1e5 8 -o my_sim_results.fits`
    """

    # User Inputs
    config = config_from_toml(config_file)

    # config.simulation.thrown_events = int(
    #    config.simulation.thrown_events if count == 0.0 else count
    # )

    if config.detector.sun_moon.id == "apply_sun_moon_cuts":
        overwrite_sunmoon = parse_sun_moon_cuts(sunmooncutparameters)
        if overwrite_sunmoon:
            config.detector.sun_moon = overwrite_sunmoon

    config.simulation.num_time_bins = int(
        config.simulation.num_time_bins if count == 0.0 else count
    )

    overwrite_integ_method = (
        parse_target_integ_method_options(montecarlo, targetapprox, cubature)
        if config.simulation.mode == "Target"
        else parse_diffuse_integ_method_options(montecarlo, targetapprox, cubature)
    )
    if overwrite_integ_method:
        config.simulation.integ_method = overwrite_integ_method

    overwrite_spectrum = parse_spectra_options(monospectrum, powerspectrum)
    if overwrite_spectrum:
        config.simulation.spectrum = overwrite_spectrum

    overwrite_cloud = parse_cloud_options(nocloud, monocloud, pressuremapcloud)
    if overwrite_cloud:
        config.simulation.cloud_model = overwrite_cloud

    if config.simulation.mode == "Target":
        overwrite_source_info = parse_source_info(sourceinfo)
        if overwrite_source_info:
            config.simulation.target = overwrite_source_info

    plot = read_plot_config(registry, plotall, plotconfig, plot)

    output = output_filename(output)
    simulation = compute(
        config,
        verbose=True,
        to_plot=plot,
        output_file=output,
        write_stages=write_stages,
    )

    if not no_result_file:
        simulation.write(output, format="fits", overwrite=True)
