import click
from .params import NssConfig
from .detector_geometry import DetectorGeometry
from .taus import Taus
from .eas import EAS


@click.command()
@click.option('--count', default=1_000,
              help="Number of trajectories to simulate.")
@click.argument('config_file', default="sample_input_file.xml")
def main(count, config_file):
    """
    Master Loop for nuSpaceSim.

    Given a count of simulated nutrinos and an XML configuration file, runs
    nutrino simulator.
    """
    numtrajs = count
    config = NssConfig(config_file)

    detector = DetectorGeometry(config)
    tau = Taus(config)
    eas = EAS(config)

    detector.throw(numtrajs)
    betaArr = detector.betas()
    tauBeta, tauLorentz, showerEnergy, tauExitProb = tau(betaArr)
    # eas_results = eas(betaArr, tauBeta, tauLorentz, showerEnergy)


if __name__ == '__main__':
    main()
