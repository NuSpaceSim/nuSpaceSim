#
#
#

import click

from nuspacesim.configuration import (
    NssConfig,
    SimulationParameters,
    config_from_xml,
    create_xml,
)
from nuspacesim.simulate import simulate
from nuspacesim.storage import write_fits  # , write_hdf5


@click.group()
# @click.option("--debug/--no-debug", default=False)
def cli():
    pass
    # def cli(ctx, debug):
    #     # ctx.ensure_object(dict)
    #     # ctx.obj["DEBUG"] = debug


@cli.command()
@click.argument(
    "config_file", default="sample_input_file.xml", type=click.Path(exists=True)
)
@click.argument("count", type=float, default=0.0)
@click.argument("logevalue", type=float, default=8.0)
# @click.pass_context
def run(config_file, count, logevalue):
    """
    Main Simulator for nuspacesim.  Given a XML configuration file, and
    optionally a count of simulated nutrinos, runs nutrino simulation.
    """

    # User Inputs
    config = config_from_xml(config_file)
    config.simulation.N = int(config.simulation.N if count == 0.0 else count)
    config.simulation.nu_tau_energy = 10 ** logevalue

    simulation = simulate(config, verbose=True)
    write_fits(simulation)
    # write_hdf5(simulation)


@cli.command()
@click.option(
    "-n", "--numtrajs", type=float, default=100, help="number of trajectories."
)
@click.option("-e", "--energy", default=8.0, help="log10(nu_tau energy) in GeV")
@click.argument("filename")
# @click.pass_context
def create_config(filename, numtrajs, energy):
    """
    Generate a configuration file from the given parameters.
    """

    simulation = SimulationParameters(N=int(numtrajs), nu_tau_energy=(10 ** energy))

    create_xml(filename, NssConfig(simulation=simulation))


if __name__ == "__main__":
    cli()
