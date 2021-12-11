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

""" Command line client source code.

.. _cli:

*****
 CLI
*****

.. currentmodule:: nuspacesim.apps.cli

.. click:: nuspacesim.apps.cli:cli
   :prog: nuspacesim
   :show-nested:

"""


import configparser

import click

from .. import NssConfig, SimulationParameters, simulation
from ..compute import compute
from ..config import FileSpectrum, MonoSpectrum, PowerSpectrum
from ..results_table import ResultsTable
from ..utils import plots
from ..utils.plot_function_registry import registry
from ..xml_config import config_from_xml, create_xml

__all__ = ["create_config", "run", "show_plot"]


@click.group()
# @click.option("--debug/--no-debug", default=False)
def cli():
    pass
    # def cli(ctx, debug):
    #     # ctx.ensure_object(dict)
    #     # ctx.obj["DEBUG"] = debug


@cli.command()
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
@click.argument(
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.argument("count", type=float, default=0.0)
def run(
    config_file: str,
    count: float,
    monospectrum,
    powerspectrum,
    no_result_file: bool,
    output: str,
    plot: list,
    plotconfig: str,
    plotall: bool,
    write_stages: bool,
) -> None:
    """Perform the full nuspacesim simulation.

    Main Simulator for nuspacesim.  Given a XML configuration file, and
    optionally a count of simulated nutrinos, runs nutrino simulation.

    \f

    Parameters
    ----------
    config_file: str
        XML configuration file for particular simulation particular.
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

    `nuspacesim run sample_input_file.xml 1e5 8 -o my_sim_results.fits`
    """

    # User Inputs
    config = config_from_xml(config_file)

    config.simulation.N = int(config.simulation.N if count == 0.0 else count)

    if monospectrum is not None and powerspectrum is not None:
        raise RuntimeError("Only one of --monospectrum or --powerspectrum may be used.")
    if monospectrum is not None:
        config.simulation.spectrum = MonoSpectrum(monospectrum)
    if powerspectrum is not None:
        config.simulation.spectrum = PowerSpectrum(*powerspectrum)

    plot = (
        list(registry)
        if plotall
        else read_plot_config(plotconfig)
        if plotconfig
        else plot
    )
    simulation = compute(
        config,
        verbose=True,
        to_plot=plot,
        output_file=output,
        write_stages=write_stages,
    )

    if not no_result_file:
        simulation.write(output, overwrite=True)


@cli.command()
@click.option(
    "-n", "--numtrajs", type=float, default=100, help="number of trajectories."
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
@click.argument("filename")
def create_config(filename: str, numtrajs: float, monospectrum, powerspectrum) -> None:
    """Generate a configuration file from the given parameters.

    \f

    Parameters
    ----------
    filename: str
        Name of output xml configuration file.
    numtrajs: float, optional
        Number of thrown trajectories. Optionally override value in config_file.

    Examples
    --------
    Command line usage of the create_config command may work with the following invocation.

    `nuspacesim create_config -n 1e5 sample_input_file.xml`
    """
    if monospectrum is not None and powerspectrum is not None:
        raise RuntimeError("Only one of --monospectrum or --powerspectrum may be used.")

    spec = MonoSpectrum()

    if monospectrum is not None:
        spec = MonoSpectrum(monospectrum)
    if powerspectrum is not None:
        spec = PowerSpectrum(*powerspectrum)
    # spec = FileSpectrum()

    simulation = SimulationParameters(N=int(numtrajs), spectrum=spec)

    create_xml(filename, NssConfig(simulation=simulation))


@cli.command()
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
@click.argument(
    "simulation_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
def show_plot(
    simulation_file: str,
    plot: list,
    plotconfig: str,
    plotall: bool,
) -> None:
    """Show predefined plots of results in simulation file.

    \f

    Parameters
    ----------
    simulation_file: str
        input ResultsTable fits file.
    plot: list, optional
        Plot the simulation results.
    plotconfig: str, optional
        INI file to select plots for each module, as well as to specifiy global plot settings.
    plotall: bool, optional
        Plot all the available the simulation results plots.


    Examples
    --------
    Command line usage of the run command may work with the following invocation.

    `nuspacesim show_plot my_sim_results.fits -p taus_overview`
    """

    sim = ResultsTable.read(simulation_file)

    plot = (
        list(registry)
        if plotall
        else read_plot_config(plotconfig)
        if plotconfig
        else plot
    )

    simulation.geometry.region_geometry.show_plot(sim, plot)
    simulation.taus.taus.show_plot(sim, plot)
    simulation.eas_optical.eas.show_plot(sim, plot)
    plots.show_plot(sim, plot)


def read_plot_config(filename):
    plot_list = []
    cfg = configparser.ConfigParser()
    cfg.read(filename)
    for sec in cfg.sections()[1:]:
        for key in cfg[sec]:
            try:
                if cfg[sec].getboolean(key):
                    plot_list.append(key)
            except Exception as e:
                print(e, "Config file contains non-valid option")
    return plot_list


if __name__ == "__main__":
    cli()
