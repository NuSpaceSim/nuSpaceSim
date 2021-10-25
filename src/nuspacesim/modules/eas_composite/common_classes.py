import numpy as np
import h5py 


class FileConverter:
    '''
    > Convert Pythia and Conex default .txt or .dat output into .h5 \n
    > Convert .h5 files (with a given key) to an array
    '''
    def __init__(self, data_file, data_set_name = None):
        self.data_file = data_file 
        self.data_set_name = data_set_name 
    
    def convert_to_h5(self, hdf5_name = 'conex_converted', data_set_name = 'eas_dataset'): 
        old_conex =  np.genfromtxt(self.data_file)
        new_hdf5 = h5py.File(hdf5_name, 'w') 
        new_hdf5.create_dataset(data_set_name, data = old_conex)
        
    def h5_to_array(self): 
        h5_file = h5py.File(self.data_file, 'r')
        showers = h5_file.get(self.data_set_name)
        showers_array = np.array(showers)
        return showers_array
    
class ParameterReader: #add nmax reader like Generating Histograms from conex plotter (for loop)
    '''
    > Reads in data parameters for functions from files
    '''
    
    def __init__(self, data_array):
        self.data_array = data_array
        
    def gh_param_reader (self, row, x_lim, bin_number): #, showers, row, xlim
        '''
        > extracts the G-H parameters from a data set, \n
        > returns desired values for a given curve (row) in showers \n
        row = row of the table\n
        x_lim = bin limit, will be divided into 1000 bins unless specified otherwise \n
        '''
        pid = self.data_array[row, 0]    #particle ID
        lgE = self.data_array [row, 1]    #log10 of the primary energy in eV
        zenith = self.data_array [row, 2]     #zenith angle in degree
        azimuth =    self.data_array[row, 3]    #azimuth angle in degree
        
        n_max = self.data_array[row, 4] #max number of particles
        x_max = self.data_array[row, 5] #shower maximum at this depth
        x_0 = self.data_array[row, 6] #first interaction point for GH fit
        
        #parameters for the polynomial function of the gaisser-hillas
        p1 = self.data_array[row, 7]
        p2 = self.data_array[row, 8]
        p3 = self.data_array[row, 9]
        
        chi2 = self.data_array[row, 10] 
        
        #some of the X0 values are negative (physically doesn't make sense), this cut addresses that
        if x_0 >= 0:
            x = np.linspace( x_0 , x_lim, bin_number)
        else: 
            x = np.linspace( x_lim/ bin_number, x_lim, bin_number)   #slant  Depth 
         
        #calculating gaiser-hillas function
        gh_lambda = p1 + p2*x + p3*(x**2) 
        
        exp1 = (x_max - x_0) / gh_lambda 
        term1 = n_max * np.nan_to_num ( pow( (x - x_0) / (x_max - x_0), exp1) )
        
        exp2 = (x_max - x) / gh_lambda 
        term2 = np.exp(exp2) 
        
        f = term1 * term2
        
        gh_lambda_at_x_max = p1 + p2*x_max + p3*(x_max**2)
        
        return x, f, n_max, x_max, x_0, gh_lambda_at_x_max