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

import click

from ... import simulation
from ...results_table import ResultsTable
from ...utils import dashboard_plot
from ...utils.plot_function_registry import registry
from ...utils.plot_wrapper import PlotWrapper


@click.command()
@click.option(
    "-p",
    "--plot",
    type=click.Choice(list(registry), case_sensitive=False),
    multiple=True,
    default=[],
    help="Available plotting functions. Select multiple plots with multiple uses of -p",
)
@click.option(
    "-pc",
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

    plot = list(registry) if plotall else plot
    plot_wrapper = PlotWrapper(to_plot=plot, plotconfig=plotconfig)

    simulation.geometry.region_geometry.show_plot(sim, plot_wrapper)
    simulation.taus.taus.show_plot(sim, plot_wrapper)
    simulation.eas_optical.eas.show_plot(sim, plot_wrapper)
    dashboard_plot.show_plot(sim, plot_wrapper)
