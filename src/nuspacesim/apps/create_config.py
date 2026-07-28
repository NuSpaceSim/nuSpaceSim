import click
import numpy as np

from ..config import NssConfig, create_toml
from .utils import (
    parse_cloud_options,
    parse_diffuse_integ_method_options,
    parse_source_info,
    parse_spectra_options,
    parse_sun_moon_cuts,
    parse_target_integ_method_options,
)


@click.command()
# @click.option(
#    "-n", "--numthrown", type=float, default=100, help="number of thrown events."
# )
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
    help="Apply cuts accounting for limited observing due to the Sun and the Moon.",
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
    "(determines the number of nodes) and the number of events per node.",
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
    "User should provide a month name, abbreviation, or number.",
)
@click.option(
    "--sourceinfo",
    nargs=5,
    type=click.Tuple([float, float, str, str, float]),
    default=[np.radians(0.0), np.radians(0.0), "2022-06-02T01:00:00", "isot", 86400],
    help="Source information. RA, Dec., observation date, date format, amount of observation time (s).",
)
@click.argument("filename")
def create_config(
    filename: str,
    # numthrown: float,
    diffuse: bool,
    target: bool,
    sunmooncuts: bool,
    sunmooncutparameters: click.Tuple,
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
    """Generate a configuration file from the given parameters.

    \f

    Parameters
    ----------
    filename: str
        Name of output toml configuration file.
    numthrown: float, optional
        Number of thrown trajectories. Optionally override value in config_file.

    Examples
    --------
    Command line usage of the create_config command may work with the following invocation.

    `nuspacesim create_config -n 1e5 sample_input_file.toml`
    """
    config = NssConfig()

    config.detector.sun_moon.id = (
        "apply_sun_moon_cuts" if sunmooncuts else "no_sun_moon_cuts"
    )

    if sunmooncuts:
        overwrite_sunmoon = parse_sun_moon_cuts(sunmooncutparameters)
        if overwrite_sunmoon:
            config.detector.sun_moon = overwrite_sunmoon

    config.simulation.mode = "Target" if target else "Diffuse"

    overwrite_integ_method = (
        parse_target_integ_method_options(montecarlo, targetapprox, cubature)
        if target
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

    if target:
        overwrite_source_info = parse_source_info(sourceinfo)
        if overwrite_source_info:
            config.simulation.target = overwrite_source_info

    # config.simulation.thrown_events = int(numthrown)
    config.simulation.num_time_bins = int(numtimebins)

    create_toml(filename, config)
