import click

from ... import NssConfig, SimulationParameters
from ...config import MonoSpectrum, PowerSpectrum  # FileSpectrum,
from ...xml_config import create_xml


@click.command()
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

    sim = SimulationParameters(N=int(numtrajs), spectrum=spec)

    create_xml(filename, NssConfig(simulation=sim))
