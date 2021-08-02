import numpy as np
import matplotlib.pyplot as plt
import h5py 

#try to find a place for functions like these
def pad_row(uneven_list, fillval = np.nan):
    '''
    > pads an ununiform array with Nans \n
    https://stackoverflow.com/a/40571482
    '''
    lens = np.array([len(item) for item in uneven_list])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(uneven_list)
    return out  
 
def paramHist (param, title, roundby,  xlabel, color, histtype): 
    '''
    > takes an array of param values and computes & plots w/ approproiate bins \n  
    > https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule \n
    param = parameter to plot \n 
    title = string object of title \n
    xlabel = x axis label \n 
    roundby = round statistics to this decimal \n
    color = string of known colors  \n
    histype = if True, probability on the y axis, else, raw counts on the y axis  \n          
    '''
    q75, q25 = np.percentile(param, [75 ,25])
    IQR = q75 - q25
    binwidth = (2*IQR) / ( len(param)**(1/3) ) 
    bins = round( ( float (max(param)) - float (min(param)) ) / binwidth )
    
    plt.figure(figsize=(8, 5), dpi= 100)
    
    if histtype is True: 
        
        plt.hist (param, bins = bins, color = color, edgecolor='black', 
                  weights = np.ones(len(param)) / len(param) )
        
        plt.title( title  + '\n[Mean = ' + str(round (np.mean(param), roundby ) ) + 
                  ', Median = ' + str(round (np.median(param), roundby ) ) 
                  + ', StDev = ' + str(round (np.std(param), roundby  ) ) + ']'  ) 

        plt.annotate('Total Counts = ' + str(len(param)), 
                     xy=(0.75, 0.96), xycoords='axes fraction') 
        
        plt.xlabel(xlabel)
        plt.ylabel('Probability')
        plt.show()
        
    else:
        
        plt.hist (param, bins = bins, color = color, edgecolor='black' )
        
        plt.title( title  + ' \n [Mean = ' + str(round (np.mean(param), roundby ) ) + 
                  ', Median = ' + str(round (np.median(param), roundby ) ) 
                  + ', StDev = ' + str(round (np.std(param), roundby  ) ) + ']' ) 
        plt.annotate('Total Counts = ' + str(len(param)), 
                     xy=(0.75, 0.96), xycoords='axes fraction')  
        plt.xlabel(xlabel)
        plt.ylabel('Raw Counts')
        plt.show() 

    

####################################
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
        azimuth = self.data_array[row, 3]    #azimuth angle in degree
        
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
    


           
class ConexPlotter: 
    '''
    > Stand-alone class for Conex plotting routines directly from a .h5 data_file containing 
    a specific data_set_name. \n
    > Conex EAS Profile Plotter and Sampler, with the option of getting the particle content 
    and associated bins. \n
    > Can modify the plot after running; plt.show() is not called within the class
    '''
    def __init__(self, data_file, data_set_name = None):
        '''
        > uses two classes, the FileConverter (to turn input file to array) and the
        ParameterReader to read the G-H parameters from said array
        '''
        self.data_file = data_file 
        self.data_set_name = data_set_name 
        
        self.data = FileConverter(self.data_file, self.data_set_name)
        self.showers_array = self.data.h5_to_array()
        
        self.parameter_reader = ParameterReader(self.showers_array)
    
    
    def gh_regular_plot (self, number_of_showers, x_lim = 2000, start_row = 0, bin_number = 1000, 
                         get_data = None):
        '''
        > regular plotting routine \n 
        > can select the number of showers to be plotted, up to what depth to plot it to,
        what row to start, and whether to get the data for each curve
        '''
        rows = [] 
        bins = []
        shower_contents = []

        for row in range (start_row, start_row + number_of_showers): 
            x, f =  self.parameter_reader.gh_param_reader(row, x_lim, bin_number)[0:2] 
            
            rows.append('Row ' + str(row) )   
            shower_contents.append(f)
            bins.append(x)
        
        plt.figure(figsize=(8, 5), dpi= 120)
        for row, x, f in zip(rows, bins, shower_contents) : 
            plt.plot(x, f, label = row)
        
        title = self.data_file.split('/')[-1]
        plt.title(title +' | '+ self.data_set_name) 
        plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
        plt.xlim(left = 0 )
        plt.ylabel('Number of Particles')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend()
        
        if get_data is not None: 
            bins = np.array(bins)
            shower_contents = np.array(shower_contents)
            return bins, shower_contents, rows     
        
    def gh_rebound_plot (self, number_of_showers, x_lim = 2000, start_row = 0, bin_number = 1000,
                         rebound_threshold = 0.10, get_data = None):
        '''
        > plotting routine until certain rebound_threshold for G-H param \n 
        > can select the number of showers to be plotted, up to what depth to plot it to,
        what row to start, the number of bins, the threshold and whether to get the data 
        for each curve \n 
        > each shower data, if cut, is filled with NaNs so as to get an even array \n 
        '''
        rows = [] 
        bins = []
        shower_contents = []
        cutoff_depths = []
        
        for row in range (start_row, start_row + number_of_showers): 
            x, f, n_max =  self.parameter_reader.gh_param_reader(row, x_lim, bin_number)[0:3] 
            
            n_max_fraction =  n_max * float (rebound_threshold )
            #indeces where it is less than set fraction 
            places_to_cut = np.argwhere(f <= n_max_fraction) 
            #highest index where it is still less than (after or before peak, depending on xlimit) 
            cut_here = np.amax(places_to_cut) 
            #trims the arrays from beginning to where number of  particles < Nmax_fraction 
            f = np.array ( f[0 : cut_here ] ) 
            x = np.array ( x[0 : cut_here ] )
            cutoff_depth = np.amax(x) #get the slant depth when f exceeds Nmax_fraction
            
            rows.append ( 'Row ' + str(row) )   
            shower_contents.append (f)
            bins.append (x)
            cutoff_depths.append(cutoff_depth)
        
        bins = np.array(pad_row(bins))
        shower_contents = np.array(pad_row(shower_contents))
        
        plt.figure(figsize=(8, 5), dpi= 120)
        for row, x, f in zip(rows, bins, shower_contents) : 
            plt.plot(x, f, label = row)
        
        title = self.data_file.split('/')[-1]
        plt.title(title +' | '+ self.data_set_name + ' | ' + str(rebound_threshold) + 
                  ' Nmax Rebound Cut'  ) 
        plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
        plt.xlim(left = 0 )
        plt.ylabel('Number of Particles')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.yscale('log')
        plt.legend()
       
        if get_data is not None: 
            return bins, shower_contents, rows, cutoff_depths        
            
    def gh_average_plot (self, number_of_showers, x_lim = 2000, start_row = 0, bin_number = 1000,
                         get_data = None):  
        '''
        > averages each bin of all the selected rows and plots them with RMS error (average +/-
        RMS error )\n
        > the user has the option to return the averaged particle content and bins, as well 
        as the rms error itself. 
        '''
        rows = [] 
        bins = []
        shower_contents = []        
        
        for row in range (start_row, start_row + number_of_showers): 
            x, f =  self.parameter_reader.gh_param_reader(row, x_lim, bin_number)[0:2] 
            
            rows.append('Row ' + str(row) )   
            shower_contents.append(f)
            bins.append(x)
            
        shower_contents = np.array(shower_contents) #Makes the list of fs into a 2D matrix
        average_shower_contents = shower_contents.mean(axis = 0) #takes the mean of each row (each bin)
        
        bins = np.array(bins) 
        average_bins =  bins.mean(axis = 0)
        
        # to get RMS
        shower_contents_square = np.square (shower_contents.T) #squares the 2D matrix  
        mean_shower_contents = shower_contents_square.mean(axis = 1) #gets mean along each bin
        rms_shower_contents =  np.sqrt(mean_shower_contents) #gets root of each mean
        
        #draws average plot
        plt.figure(figsize=(8, 5), dpi= 120)
        plt.plot(average_bins, average_shower_contents,  '--k' , 
                 label = 'Average (Row ' + str (start_row) + 
                 ' to Row ' + str (start_row + number_of_showers) + ')')
        
        #draws rms plot
        #plt.plot(X, rms_fs, '--r', label = 'RMS')
        
        #here is a crude measurement for error, which is the RMS minus the average value
        #rms_error = rms_fs - np.mean( fs.T, axis = 1) 

        #here is the proper definition for rms_error
        #to get RMSE where the average is the estimated series and particleContent are values
        rms_error = np.sqrt(np.mean( ( average_shower_contents  - shower_contents )**2, axis = 0 ))
        
        #rms_error = np.sqrt(np.mean( ( average - particleContent )**2, axis = 0 ))
        plt.fill_between(average_bins, 
                         average_shower_contents - rms_error, average_shower_contents + rms_error,
                         alpha = 0.5, edgecolor='red', facecolor='crimson', 
                         label = 'Error')
        
        title = self.data_file.split('/')[-1]
        plt.title('Average Profile ' + title +' | '+ self.data_set_name ) 
        plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
        plt.xlim(left = 0 )
        plt.ylabel('Number of Particles')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend()
       
        if get_data is not None: 
            return average_bins, average_shower_contents, rms_error
        
    def __call__(self): 
        plt.show()
        

if __name__ == '__main__':
    
    #here is a test plot, extracting data of each curve
    gamma = ConexPlotter('Conex/HDF5_data/proton_EAStable.h5', 'EASdata_100')
    bins, shower_contents, rows = gamma.gh_regular_plot(number_of_showers = 10, 
                                                             get_data= True)
    