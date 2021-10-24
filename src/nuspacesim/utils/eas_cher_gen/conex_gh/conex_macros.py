import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
np.seterr(all='ignore')

 
def conex_converter (input_file, output_file = 'converted.h5', output_dataset = 'conex_output'):
    old_conex =  np.genfromtxt(input_file)
    new_hdf5 = h5py.File(output_file, 'w')
    new_hdf5.create_dataset(output_dataset, data = old_conex)
    print('File written to current directory...')

# def data_reader (file_name, data_name):
#     ref = importlib_resources.files(
#     'nuSpaceSim.DataLibraries.ConexOutputData.HDF5_data') / file_name
#     with importlib_resources.as_file(ref) as path:
#         data = h5py.File(path, 'r')
#         showers = data.get(data_name)
#         showers = np.array(showers)
#         return showers

def data_reader (file_path, data_name): 
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        print(f"{file_path} is not a valid path!")
        sys.exit()

    with h5py.File(file_path, "r") as f:
        showers = f.get(data_name)
        showers = np.array(showers)
        return showers

def gh_param_reader (row, x_limit, showers = None, file_name = None , data_name =None , bins = 2000): 
    '''
    > extracts the G-H parameters from a data set, \n
    > returns desired values for a given curve (row) in showers \n
    showers = 2D np array with columns to unpack
    row = row of the table\n
    xlim = bin limit, will be divided into 1000 bins\n
    '''
    if file_name and data_name is not None: 
        showers = data_reader(file_name, data_name)
    
    pid = showers[row, 0] #particle ID
    lgE = showers[row, 1] #log10 of the primary energy in eV
    zenith = showers[row, 2] #zenith angle in degree
    azimuth = showers[row, 3] #azimuth angle in degree
    

    n_max = (showers[row, 4])
    x_max = showers[row, 5] #G-H fit result for slant depth of the shower maximum (g/cm^2)
    x_0 = showers[row, 6] #first interaction point for GH fit
    
    #parameters for the polynomial function of the GH fit
    p1 = showers[row, 7]
    p2 = showers[row, 8]
    p3 = showers[row, 9]
    
    chi2 = showers[row, 10] #Chi squared / number of degree of freedom / sqrt (Nmax) for the fit
    
    #some of the X0 values are negative (physically doesn't make sense), this cut addresses that
    if x_0 >= 0:
        x = np.linspace( x_0, x_limit + 1, bins)
    else: 
        x = np.linspace( x_limit/ bins, x_limit + 1, bins)   #slant  Depth 
     
    #calculating gaiser-hillas function
    gh_lambda = p1 + p2*x + p3*(x**2) 
    
    exp1 = (x_max - x_0) / gh_lambda 
    term1 = n_max * np.nan_to_num ( pow( (x - x_0) / (x_max - x_0), exp1) )
    
    exp2 = (x_max - x) / gh_lambda
    term2 = np.exp(exp2) 
    
    f = term1 * term2
    
    gh_lambda_at_x_max = p1 + p2*x_max + p3*(x_max**2)

    return x, f,  n_max, x_max, x_0,  gh_lambda_at_x_max   


def gh_profile_plot(file_name, data_name, end_row, start_row = 0, regular_plot = None, 
                    average_plot = None, rebound_plot = None, n_max_cut_threshold = None, 
                    x_limit = 2000, bins = 2000):
    '''
    > plots the profiles based on gaisser-hillas parameters \n
    > uses the gaisser-hillas function \n
    '''
    file_name:str 
    data_name:str
    start_row:int
    end_row:int 
    
    showers = data_reader(file_name, data_name)
   
    if n_max_cut_threshold is not None: 
        #stopping in slant depth once the particle content f is Nmax_threshold of Nmax
        #note that for each profile, Nmax is different 
        rows = [] 
        fs = []
        xs = []
        
        for row in range (start_row, end_row + 1):  
            x, f, n_max, x_max = gh_param_reader(row = row, x_limit = x_limit, bins = bins,
                                       showers = showers)[0:4] 
            n_max_fraction = n_max * float (n_max_cut_threshold )
            #indeces where it is less than the fraction
            places_to_cut = np.argwhere(f <= n_max_fraction) 
            #highest index where it is still less than (after or before peak, depends on x_limit) 
            cut_here = np.amax(places_to_cut) 
            #trims the arrays from 0 to where number of  particles < n_max_fraction
            trimmed_f = np.array (f[0:cut_here] )
            trimmed_x = np.array (x[0:cut_here] )
            #get the slant depth when f exceeds n_max_fraction
            cutoff_depth = np.amax(trimmed_x) 
            
            #index closest to curve peak
            location_nearest_n_max = np.argmin(np.absolute(f - n_max))
            
            if places_to_cut[-1] > location_nearest_n_max and  f [-2] < f[-1]:
                #rebounds within given x_limit
                #pad trimmed_f so each vector has same dimensions
                trimmed_f = np.pad(trimmed_f, (0,bins - len(trimmed_f)), 
                                   'constant', constant_values = float('nan'))
                trimmed_x = np.pad(trimmed_x, (0,bins - len(trimmed_x)), 
                                   'constant', constant_values = float('nan'))
                rows.append ( 'Row ' + str(row))   
                fs.append (trimmed_f)
                xs.append (trimmed_x)
            else: 
                #does not rebound in range, just plut until x_limit, tag with NR
                rows.append ( 'Row ' + str(row) + ' -NR' )   
                fs.append (f)
                xs.append (x)
        
        
        if rebound_plot is not None:
            print("rebound_plot: " +  str(data_name) )
            plt.figure(figsize=(8, 5), dpi= 120)
            for row, f, x in zip(rows, fs, xs) : 
                plt.plot(x, f, label = row)   
            
            plt.title('G-H Plots | ' + str(file_name.split('/')[-1]) +'/'+ str(data_name) + 
                      ' ($ N_{rebound} \leq' + str(n_max_cut_threshold) + '\; Nmax$)' ) 
            plt.yscale('log')
            plt.ylabel('Number of Particles N')
            plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
            plt.xlim(left = 0 )
            plt.legend()
            
        else: 
            if np.isclose(cutoff_depth,  x_limit, atol = 1):
                #means there is no cut off depth since the graph doesn't reach the 
                #set n_max_fraction within the given x_limit
                return float('nan')
            else:
                #returns true cutoff depth
                return cutoff_depth           
           
    else: 
        
        if regular_plot is not None: 
            #regular plotting routine 
            rows = [] 
            fs = []
            xs = [] 
            for row in range (start_row, end_row + 1): 
                x, f = gh_param_reader(row = row, x_limit = x_limit, bins = bins,
                                       showers = showers)[0:2] 
                rows.append ( 'Row ' + str(row) )   
                fs.append (f)
                xs.append (x)
            print("regular_plot: " +  str(data_name))    
            plt.figure(figsize=(8, 5), dpi= 120)    
            for row, f, x in zip(rows, fs, xs) : 
                plt.plot(x, f, label = row) 

            plt.title('G-H Plots | ' + str(file_name.split('/')[-1]) +'/'+ str(data_name))
            plt.ylabel('Number of Particles N')
            plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
            plt.xlim(left = 0 )         
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
            plt.legend()
            
        if average_plot is not None:            
            rows = [] 
            fs = []
            xs = []   
            for row in range (start_row, end_row + 1): ####See if you can delete this
                x, f = gh_param_reader(row = row, x_limit = x_limit, bins = bins,
                                       showers = showers)[0:2] 
                rows.append ( 'Row ' + str(row) )   
                fs.append (f)
                xs.append (x)
                
            fs = np.array(fs) #makes the list of fs into a 2D matrix
            average_fs = fs.mean(axis = 0) #Takes the mean of each row (each bin)
            
            #to get RMS
            fs_square = np.square (fs.T) #squares the 2D matrix  
            mean_fs = fs_square.mean(axis = 1) #gets mean along each bin
            rms_fs =  np.sqrt(mean_fs) #gets root of each mean
            
            #note, RMS error is equal to the stdev if you are using the average as the predicted
            #value; i.e., np.std(param) = np.sqrt(np.mean(( np.mean(param) - param)**2))
            #RMS is greater than the mean, but RMS error is less than the mean
            
            #to get RMSE where the average is the estimated series
            rms_error = np.sqrt(np.mean( ( average_fs  - fs )**2, axis = 0 ))

            print("average_plot: " +  str(data_name))
            plt.figure(figsize=(8, 5), dpi= 120)
            plt.plot(x, average_fs,  '--k' , label = 'Average (Row ' + str (start_row) + 
                     ' to Row ' + str (end_row) + ')')
            #RMS error
            plt.fill_between(x, average_fs - rms_error, average_fs + rms_error,
                             alpha = 0.5, edgecolor='red', facecolor='crimson', 
                             label = 'Error')
            
            plt.title('G-H Plots | ' + str(file_name.split('/')[-1]) +'/'+ str(data_name))
            plt.ylabel('Number of Particles N')
            plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
            plt.xlim(left = 0 )         
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
            plt.legend()
    
 
    
def parameter_histogram(param, title, x_label, color = None, hist_type = 'counts', 
                        round_by = 0): 
    '''
    > takes a vector of param values and computes & plots w/ approproiate bins \n  
    param = parameter to plot \n 
    title = string object of title \n
    xlabel = x axis label \n 
    roundby = round statistics to this decimal \n
    color = string of known colors  \n
    histype = if True, probability on the y axis, else, raw counts on the y axis  \n          
    '''
    
    print('parameter_histogram:',  title)
    param = [param for param in param if str(param) != 'nan']
    param = np.array(param) 
    #Diaconis Rule 
    q75, q25 = np.percentile(param, [75 ,25])
    IQR = q75 - q25
    binwidth = (2*IQR) / ( param.size**(1/3) ) 
    bins = round( ( float (max(param)) - float (min(param)) ) / binwidth )
    
    if color is None:
        color = np.random.rand(3,1).T
   
    if hist_type != 'counts': 
        plt.figure(figsize=(8, 5), dpi= 120)
        plt.hist (param, bins = bins, color = color, edgecolor='black', 
                  weights = np.ones(len(param)) / len(param) )
        
        plt.title( title  + '\n[Mean = ' + str(round (np.mean(param), round_by ) ) + 
                  ', Median = ' + str(round (np.median(param), round_by ) ) 
                  + ', StDev = ' + str(round (np.std(param), round_by ) ) + ']'  ) 

        plt.annotate('Total Counts = ' + str(len(param)), 
                     xy=(0.75, 0.96), xycoords='axes fraction') 
        
        plt.xlabel(x_label)
        plt.ylabel('Probability')
        
    else:
        plt.figure(figsize=(8, 5), dpi= 120)
        plt.hist (param, bins = bins, color = color, edgecolor='black' )
        
        plt.title( title  + ' \n [Mean = ' + str(round (np.mean(param), round_by ) ) + 
                  ', Median = ' + str(round (np.median(param), round_by ) ) 
                  + ', StDev = ' + str(round (np.std(param), round_by  ) ) + ']' ) 
        plt.annotate('Total Counts = ' + str(len(param)), 
                     xy=(0.75, 0.96), xycoords='axes fraction')  
        plt.xlabel(x_label)
        plt.ylabel('Raw Counts')
        
    
   
    