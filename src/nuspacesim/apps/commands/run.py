import click

from ...compute import compute
from ...config import MonoSpectrum, PowerSpectrum  # FileSpectrum,
from ...utils.plot_function_registry import registry
from ...xml_config import config_from_xml


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

    plot = list(registry) if plotall else plot
    print("plotconfig", plotconfig)

    simulation = compute(
        config,
        verbose=True,
        plot=plot,
        plot_config=plotconfig,
        output_file=output,
        write_stages=write_stages,
    )
    if not no_result_file:
        simulation.write(output, overwrite=True)
