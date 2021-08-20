import click
from nuSpaceSim.create_xml import create_xml
from nuSpaceSim.eas import EAS
from nuSpaceSim.params import NssConfig
from nuSpaceSim.region_geometry import RegionGeom
from nuSpaceSim.sim_store import SimStore
from nuSpaceSim.taus import Taus
import nuSpaceSim.radio_antenna as radio_antenna
import numpy as np
import matplotlib.pyplot as plt

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
def run(config_file):
    """
    Main Simulator for nuSpaceSim.  Given a XML configuration file, and
    optionally a count of simulated nutrinos, runs nutrino simulation.
    """

    # User Inputs
    config = NssConfig(config_file)

    # Initialized Objects
    geom = RegionGeom(config)
    tau = Taus(config)
    eas = EAS(config)
    #store = SimStore(config)

    # Run simulation
    betaArr, thetaArr, pathLenArr = geom(config.N)
    #store('geom', ('betaArr', 'thetaArr'), betaArr, thetaArr)

    tauBeta, tauLorentz, showerEnergy, tauexitprob = tau(betaArr)
    #store('tau', ('tauBeta', 'tauLorentz', 'showerEnergy', 'tauexitprob'), tauBeta, tauLorentz, showerEnergy, tauexitprob)
    
    if config.method == 'Optical':
        numPEs, costhetaChEff = eas(betaArr, thetaArr, pathLenArr, tauBeta, tauLorentz, showerEnergy)
        #store( 'eas', ('numPEs', 'costhetaChEff'), numPEs, costhetaChEff)
        mcintegral, mcintegralgeoonly, numEvPass = geom.mcintegral(numPEs, 
                                                        costhetaChEff, tauexitprob)
        #store( 'mcintegral', ('mcintegral', 'mcintegralgeoonly', 'numEvPass'), mcintegral, mcintegralgeoonly, numEvPass)

    if config.method == 'Radio':
        EFields, decay_h = eas(betaArr, thetaArr, pathLenArr, tauBeta, tauLorentz, showerEnergy)
        #store( 'eas', ('EFields', 'decayHeight'), EFields, decay_h)
        snrs = radio_antenna.calculate_snr(EFields, config.detFreqRange, config.detectAlt, config.detNant, config.detGain)
        #print(EFields[snrs>5.0].min())
        #print(snrs, snrs.mean())
        #store( 'ant', ('snrs'), snrs)
        costhetaArr = np.cos(thetaArr)
        mcintegral, mcintegralgeoonly, numEvPass = geom.mcintegral(snrs, 
                                                        costhetaArr, tauexitprob)

    print("Geom. Only MC Integral:", mcintegralgeoonly)
    print("MC integral:", mcintegral)
    print("Fraction of Events Passing Selection Cuts:", numEvPass/float(len(tauBeta)))
    #store.close()

@cli.command()
@click.option("-n", "--numtrajs", type=float, default=100, help="number of trajectories.")
@click.option("-e", "--energy", default=9.0, help="log10(nu_tau energy) in GeV")
@click.argument("filename")
# @click.pass_context
def create_config(filename, numtrajs, energy):
    """
    Generate a configuration file from the given parameters.
    """
    create_xml(filename, int(numtrajs), energy)

if __name__ == "__main__":
    cli()
