import click
from nuSpaceSim.params import NssConfig
from nuSpaceSim.region_geometry import RegionGeom
from nuSpaceSim.taus import Taus
from nuSpaceSim.eas import EAS
from nuSpaceSim.create_xml import create_xml


@click.group()
# @click.option("--debug/--no-debug", default=False)
def cli():
    pass
    # def cli(ctx, debug):
    #     # ctx.ensure_object(dict)
    #     # ctx.obj["DEBUG"] = debug


@cli.command()
@click.argument(
    "config_file",
    default="sample_input_file.xml",
    type=click.Path(exists=True)
)
@click.argument("count", type=float, default=0.0)
# @click.pass_context
def run(config_file, count):
    """
    Main Simulator for nuSpaceSim.  Given a XML configuration file, and
    optionally a count of simulated nutrinos, runs nutrino simulation.
    """

    # User Inputs
    config = NssConfig(config_file)
    numtrajs = int(config.N if count == 0.0 else count)

    # Initialized Objects
    geom = RegionGeom(config)
    tau = Taus(config)
    eas = EAS(config)

    # Run simulation
    betaArr = geom(numtrajs)
    tauBeta, tauLorentz, showerEnergy, tauexitprob = tau(betaArr)
    numPEs, costhetaChEff = eas(betaArr, tauBeta, tauLorentz, showerEnergy)
    # More modules here
    mcintegral, mcintegralgeoonly = geom.mcintegral(numPEs, costhetaChEff,
                                                    tauexitprob)

    print("Geom. Only MC Integral:", mcintegralgeoonly)
    print("mcintegral", mcintegral)


@cli.command()
@click.option("-n", "--numtrajs", type=float, default=100, help="number of trajectories.")
@click.argument("filename")
# @click.pass_context
def create_config(filename, numtrajs):
    """
    Generate a configuration file from the given parameters.
    """
    create_xml(filename, int(numtrajs))


if __name__ == "__main__":
    cli()
