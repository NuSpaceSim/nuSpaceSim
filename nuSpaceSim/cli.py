import click
from nuSpaceSim.create_xml import create_xml
from nuSpaceSim.eas import EAS
from nuSpaceSim.params import NssConfig
from nuSpaceSim.region_geometry import RegionGeom
from nuSpaceSim.sim_store import SimStore
from nuSpaceSim.taus import Taus
import numpy as np

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
@click.argument("evalue", type=float, default=8.0)
# @click.pass_context
def run(config_file, count, evalue):
    """
    Main Simulator for nuSpaceSim.  Given a XML configuration file, and
    optionally a count of simulated nutrinos, runs nutrino simulation.
    """

    # User Inputs
    config = NssConfig(config_file)
    config.N = int(config.N if count == 0.0 else count)
    config.logNuTauEnergy = evalue
    config.nuTauEnergy = 10 ** config.logNuTauEnergy

    # Initialized Objects
    geom = RegionGeom(config)
    tau = Taus(config)
    eas = EAS(config)
    store = SimStore(config)

    # Run simulation
    beta_tr = geom(config.N)
    store('geom', ('betaArr'), beta_tr)

    tauBeta, tauLorentz, showerEnergy, tauexitprob = tau(beta_tr)
    store('tau', ('tauBeta', 'tauLorentz', 'showerEnergy', 'tauexitprob'),
            tauBeta, tauLorentz, showerEnergy, tauexitprob)

    numPEs, costhetaChEff = eas(beta_tr, tauBeta, tauLorentz, showerEnergy)
    store( 'eas', ('numPEs', 'costhetaChEff'), numPEs, costhetaChEff)

    # More modules here
    mcintegral, mcintegralgeoonly, numEvPass = geom.mcintegral(numPEs, 
                                                    costhetaChEff, tauexitprob)
    store( 'mcintegral', ('mcintegral', 'mcintegralgeoonly', 'numEvPass'),
            mcintegral, mcintegralgeoonly, numEvPass)

    store.close()

    print("Geom. Only MC Integral:", mcintegralgeoonly)
    print("MC integral:", mcintegral)
    print("Number of Events Passing Selection Cuts:", numEvPass)

@cli.command()
@click.option("-n", "--numtrajs", type=float, default=100, help="number of trajectories.")
@click.option("-e", "--energy", default=8.0, help="log10(nu_tau energy) in GeV")
@click.argument("filename")
# @click.pass_context
def create_config(filename, numtrajs, energy):
    """
    Generate a configuration file from the given parameters.
    """
    create_xml(filename, int(numtrajs), energy)

if __name__ == "__main__":
    cli()
