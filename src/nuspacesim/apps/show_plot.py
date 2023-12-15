import click
from astropy.table import Table as AstropyTable

from .. import simulation
from ..utils import plots
from ..utils.plot_function_registry import registry
from .utils import read_plot_config

__all__ = ["show_plot"]


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
        input nuspacesim results file AstropyTable fits file.
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

    sim = AstropyTable.read(simulation_file)

    plot = read_plot_config(registry, plotall, plotconfig, plot)

    simulation.geometry.region_geometry.show_plot(sim, plot)
    simulation.taus.taus.show_plot(sim, plot)
    simulation.eas_optical.eas.show_plot(sim, plot)
    plots.show_plot(sim, plot)
