import click
from nuSpaceSim.EAScherGen.Conex.conex_plotter import conex_plotter
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
    betaArr = geom(config.N)
    store('geom', ('betaArr'), betaArr)

    tauBeta, tauLorentz, showerEnergy, tauexitprob = tau(betaArr)
    store('tau', ('tauBeta', 'tauLorentz', 'showerEnergy', 'tauexitprob'),
            tauBeta, tauLorentz, showerEnergy, tauexitprob)

    numPEs, costhetaChEff = eas(betaArr, tauBeta, tauLorentz, showerEnergy)
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

#conex plotter functionality
@cli.command()
@click.argument("conex_filename")
@click.option("-d", "--dataset", type=str, help="name of data set inside file (key)" )
def conex_sampler(conex_filename, dataset): 
    """
    Generate sample plots from data set (key) inside given .h5 file. \n
    Place the .h5 file here: nuSpaceSim/DataLibraries/ConexOutputData/HDF5_data \n 
    Sample usage: $nuSpaceSim conex-sampler gamma_EAS_table.h5 --dataset EASdata_22 \n 
    """
    conex_plotter(conex_filename, dataset)


if __name__ == "__main__":
    cli()