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

""" Command line client source code.  """

import click

from ..utils.plot_function_registry import registry


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
    type=click.Choice(registry, case_sensitive=False),
    multiple=True,
    default=[],
    help="Available plotting functions. Select multiple plots with multiple uses of -p",
)
@click.option("--plotall", is_flag=True, help="Show all result plots.")
@click.option(
    "-w",
    "--write-stages",
    is_flag=True,
    help="Write intermediate values after each simulation stage.",
)
@click.argument(
    "config_file", default="sample_input_file.xml", type=click.Path(exists=True)
)
@click.argument("count", type=float, default=0.0)
@click.argument("logevalue", type=float, default=8.0)
# @click.pass_context
def run(
    config_file: str,
    count: float,
    logevalue: float,
    output: str,
    write_stages: bool,
    plot: list,
    plotall: bool,
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
    logevalue: float, optional
        Log of nu tau Energy. Optionally override value in config_file.
    output: str, optional
        Name of the output file holding the simulation results.
    plot: list, optional
        Plot the simulation results.
    plotall: boo, optional
        Plot all the available the simulation results plots.


    Examples
    --------
    Command line usage of the run command may work with the following invocation.

    nuspacesim run sample_input_file.xml 1e5 8 -o my_sim_results.fits
    """

    from ..simulate import simulate
    from ..xml_config import config_from_xml

    # User Inputs
    config = config_from_xml(config_file)
    config.simulation.N = int(config.simulation.N if count == 0.0 else count)
    config.simulation.nu_tau_energy = 10 ** logevalue
    plot = registry if plotall else plot
    simulation = simulate(
        config,
        verbose=True,
        to_plot=plot,
        output_file=output,
        write_stages=write_stages,
    )
    simulation.write(output, overwrite=True)


@cli.command()
@click.option(
    "-n", "--numtrajs", type=float, default=100, help="number of trajectories."
)
@click.option(
    "-l", "--logenergy", type=float, default=8.0, help="log10(nu_tau_energy) in GeV"
)
@click.argument("filename")
# @click.pass_context
def create_config(filename: str, numtrajs: float, logenergy: float) -> None:
    """Generate a configuration file from the given parameters.

    \f

    Parameters
    ----------
    filename: str
        Name of output xml configuration file.
    numtrajs: float, optional
        Number of thrown trajectories. Optionally override value in config_file.
    logenergy: float, optional
        Log of nu tau Energy. Optionally override value in config_file.

    Examples
    --------
    Command line usage of the create_config command may work with the following invocation.

    nuspacesim create_config -n 100000 sample_input_file.xml
    """
    from .. import NssConfig, SimulationParameters
    from ..xml_config import create_xml

    simulation = SimulationParameters(N=int(numtrajs), nu_tau_energy=(10 ** logenergy))

    create_xml(filename, NssConfig(simulation=simulation))


if __name__ == "__main__":
    cli()
