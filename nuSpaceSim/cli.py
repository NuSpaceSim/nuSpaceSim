import click
from nuSpaceSim.EAScherGen.CompositeShowers.composite_eas import composite_eas
from nuSpaceSim.EAScherGen.Conex.conex_macros import conex_converter
from nuSpaceSim.EAScherGen.Conex.conex_plotter import conex_plotter
from nuSpaceSim.create_xml import create_xml
from nuSpaceSim.eas import EAS
from nuSpaceSim.params import NssConfig
from nuSpaceSim.EAScherGen.Pythia.pythia_macros import pythia_converter
from nuSpaceSim.EAScherGen.Pythia.pythia_plotter import pythia_plotter
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

#conex to .hdf5 converter 
@cli.command()
@click.argument("conex_filename", type=click.Path(exists=True))
@click.argument("conex_out_filename", type=str)
@click.argument("conex_out_dataname", type=str)
def conex_to_h5 (conex_filename, conex_out_filenam, conex_out_dataname): 
    """
    Convert a .txt or .dat Conex output to .hdf5.
    """
    conex_converter (conex_filename, conex_out_filenam, conex_out_dataname)
    
#pythia tables to .hdf5 converter
@cli.command()
@click.argument("pythia_filename", type=click.Path(exists=True))
@click.argument("pythia_out_filename", type=str)
@click.argument("pythia_out_dataname", type=str)
def pythia_to_h5 (pythia_filename, pythia_out_filename, pythia_out_dataname): 
    """
    Convert a machine-readable tau decay tables to flattened .hdf5.
    """
    pythia_converter (pythia_filename, pythia_out_filename, pythia_out_dataname)

#conex sampler functionality for CORSIKA/CONEX GH Fit files
@cli.command()
@click.argument("conex_filename", type=click.Path(exists=True))
@click.argument("key_name", type=str)
@click.option("-p", "--ptype", type=str, default='regular', 
              help="'regular', 'average', 'rebound', or 'histograms'")
@click.option("-s", "--start", type=int, default=0, 
              help="start row from CORSIKA/CONEX GH Fits to sample")
@click.option("-e", "--end", type=int, default=10, 
              help="end row from CORSIKA/CONEX GH Fits to sample")
@click.option("-x", "--xlim", type=int, default=2000, 
              help="ending slant depth [g/cm^2]")
@click.option("-b", "--bins", type=int, default=2000, 
              help="control grammage step, default to 2000 given xlim")
@click.option("-t", "--threshold", type=float, default=0.01, 
              help="decimal multiple of Nmax at which to stop rebound")
def conex_sampler(conex_filename, key_name, ptype, start, end, xlim, bins, threshold): 
    """
    Generate sample plots from a key inside given .h5 file path. Currently works for
    100PeV Conex outputs. \n
    For sample files: nuSpaceSim/DataLibraries/ConexOutputData/HDF5_data \n
    """
    conex_plotter(conex_filename, key_name, ptype, start, end, xlim, bins, threshold)
    
#tau tables sampler for Pythia8 tau decay tables
@cli.command()
@click.argument("pythia_filename", type=click.Path(exists=True))
@click.argument("data_name", type=str)
@click.option("-p", "--pid", 
              help="particle PID to filter (11, +/-211, etc.)")
@click.option("-e", "--energy", type=float, default=100., 
              help="energy of the generated pythia table")
@click.option("-o", "--occur", default='all', 
              help="number of occurrence per event ('all', 'multiple', 1)")
@click.option("-x", "--crossfilt", default= None, 
              help="cross filter with another pid") 
@click.option("-c", "--color", type=str, default= None, 
              help="recognized plt color string; random otherwise") 
def pythia_sampler(pythia_filename, 
                   data_name, 
                   pid, 
                   energy, 
                   occur, 
                   crossfilt, 
                   color): 
    """
    Generate histograms for energy of selected particle type. \n
    For sample files: nuSpaceSim/DataLibraries/PythiaDecayTables/HDF5_data \n
    """
    pythia_plotter(file_name=pythia_filename, 
                   data_name=data_name, 
                   particle_id=pid, 
                   table_energy_pev=energy, 
                   num_occurance=occur, 
                   cross_filter=crossfilt, 
                   color=color) 

#base composite showers
@cli.command()
@click.argument("write_to", type=str)
@click.option("-p","--sample_plt",  is_flag=True)
def composite_showers(write_to, sample_plt): 
    """
    Make composite showers based on sample data.
    """
    composite_eas(write_to, sample_plt)



if __name__ == "__main__":
    cli()