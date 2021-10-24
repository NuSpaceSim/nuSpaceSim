import click


r""" Miscellaneous command line client source code.

.. autosummary::
   :toctree:
   :recursive:

   conex_to_h5
   pythia_to_h5
   conex_sampler
   pythia_sampler
   composite_showers
"""
@click.group()
# @click.option("--debug/--no-debug", default=False)
def cli_misc():
    pass
    # def cli(ctx, debug):
    #     # ctx.ensure_object(dict)
    #     # ctx.obj["DEBUG"] = debug


#conex to .hdf5 converter 
@cli_misc.command()
@click.argument("conex_filename", type=click.Path(exists=True))
@click.argument("conex_out_filename", type=str)
@click.argument("conex_out_dataname", type=str)
def conex_to_h5 (conex_filename, conex_out_filenam, conex_out_dataname): 
    r"""
    Convert a .txt or .dat Conex output to .hdf5.
    """
    from ..utils.eas_cher_gen.conex_gh.conex_macros import conex_converter
    
    conex_converter (conex_filename, conex_out_filenam, conex_out_dataname)
    
    
#pythia tables to .hdf5 converter
@cli_misc.command()
@click.argument("pythia_filename", type=click.Path(exists=True))
@click.argument("pythia_out_filename", type=str)
@click.argument("pythia_out_dataname", type=str)
def pythia_to_h5 (pythia_filename, pythia_out_filename, pythia_out_dataname): 
    r"""
    Convert a machine-readable tau decay tables to flattened .hdf5.
    """
    from ..utils.eas_cher_gen.pythia_tau_decays.pythia_macros import pythia_converter
    
    pythia_converter (pythia_filename, pythia_out_filename, pythia_out_dataname)


#conex sampler functionality for CORSIKA/CONEX GH Fit files
@cli_misc.command()
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
    r"""
    Generate sample plots from a key inside given .h5 file path. Currently works for
    100PeV Conex outputs. 

    """
    from ..utils.eas_cher_gen.conex_gh.conex_plotter import conex_plotter
    
    conex_plotter(conex_filename, key_name, ptype, start, end, xlim, bins, threshold)
  
    
#tau tables sampler for Pythia8 tau decay tables
@cli_misc.command()
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
    r"""
    Generate histograms for energy of selected particle type. 
    """
    from ..utils.eas_cher_gen.pythia_tau_decays.pythia_plotter import pythia_plotter
    
    pythia_plotter(file_name=pythia_filename, 
                   data_name=data_name, 
                   particle_id=pid, 
                   table_energy_pev=energy, 
                   num_occurance=occur, 
                   cross_filter=crossfilt, 
                   color=color) 


#base composite showers
@cli_misc.command()
@click.argument("write_to", type=str)
@click.option("-p","--sample_plt",  is_flag=True)
def composite_showers(write_to, sample_plt): 
    r"""
    Make composite showers based on sample data.
    """
    from ..utils.eas_cher_gen.composite_showers.composite_eas import composite_eas
    
    composite_eas(write_to, sample_plt)



if __name__ == "__main__":
    cli_misc()