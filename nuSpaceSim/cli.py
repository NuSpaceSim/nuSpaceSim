import click
from nuSpaceSim.params import NssConfig
from nuSpaceSim.detector_geometry import DetectorGeometry
from nuSpaceSim.taus import Taus
from nuSpaceSim.EAScherGen.eas import EAS
from nuSpaceSim.create_xml import create_xml


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug


@cli.command()
@click.argument('config_file', default="sample_input_file.xml.",
                type=click.Path(exists=True))
@click.argument('count', type=int)
@click.pass_context
def run(ctx, config_file, count):
    """
    Main Simulator for nuSpaceSim.  Given a XML configuration file, and
    optionally a count of simulated nutrinos, runs nutrino simulation.
    """

    # User Inputs
    config = NssConfig(config_file)
    # numtrajs = config['NumTrajs'] if count == 0 else count
    numtrajs = count

    # Initialized Objects
    detector = DetectorGeometry(config)
    tau = Taus(config)
    eas = EAS(config)

    # Run simulation
    detector.throw(numtrajs)
    betaArr = detector.betas()
    tauBeta, tauLorentz, showerEnergy, tauexitprob = tau(betaArr)
    numPEs, costhetaCh = eas(betaArr, tauBeta, tauLorentz, showerEnergy)
    mcintegral = detector.mcintegral(numPEs, costhetaCh, tauexitprob)
    print("mcintegral", mcintegral)


@cli.command()
@click.option('-n', '--numtrajs', default=10, help='number of trajectories.')
@click.argument('filename')
@click.pass_context
def create_config(ctx, filename, numtrajs):
    """
    Generate a configuration file from the given parameters.
    """
    create_xml(filename, numtrajs)

if __name__ == "__main__":
    cli(default_map={
        'run': {
            'count': 10
        }
    })
