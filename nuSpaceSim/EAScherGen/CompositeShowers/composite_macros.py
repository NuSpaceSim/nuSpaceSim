import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

np.seterr(all='ignore')

def data_reader (file_path, data_name): 
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        print(f"{file_path} is not a valid path!")
        sys.exit()

    with h5py.File(file_path, "r") as f:
        data_array = f.get(data_name)
        data_array = np.array(data_array)
        return data_array

#composite writer to h5
# new_hdf5 = h5py.File('sample_composite_showers.h5', 'w')
# new_hdf5.create_dataset("showers", data = composite_showers)
# new_hdf5.create_dataset("slantdepths", data = composite_bins)
# new_hdf5.close()
# print('File written to current directory...')

def greisen_param (conex_showers, row, x_limit, pythia_tables, table_energy = 100e15 ): 
    '''
    > uses the Greisen parametrization 
    > Cosmic Rays and Particle Physics by Thomas K. Gaisser, eq 15.28
    > depends solely on electron energies from Pythia Tables 
    '''
    gh_n_max = conex_showers[row, 4]  #n_max scaled by y 
    
    #read in Pythia Decays Data
    event_number = pythia_tables[row, 0]
    y = pythia_tables[row, -1] #energy decimal quantitiy of 100 PeV
    scaled_gh_n_max = gh_n_max * y 
    
    bins = x_limit
    x = np.linspace( x_limit/ bins, x_limit, bins)  #slant  Depth 
    
    #source: https://pdg.lbl.gov/2015/AtomicNuclearProperties/HTML/air_dry_1_atm.html
    #atomic and nuclear properties of air (dry, 1 atm)
    x_0 = 36.62 
    e_c = 87.92e6 #critical energy for electromagnetic cascades in the air
    e_0 = y * table_energy #energy of the photon initiated shower in eV pg 303 from Gaisser's Book
    beta_0 = np.log(e_0 / e_c)
    
    t = x / x_0 #t = x/x0, where x0 is the radiation length in air ~ 37 g/cm^2
    s = (0 + 3*t)/ (t + 2*beta_0 ) #age parameter, n = 0
    x_max = x_0 * np.log(e_0 / e_c)
    n_max = ( 0.31 / pow(beta_0, 1/2) ) * (e_0 / e_c)

    term1 = ( 0.31 / pow(beta_0, 1/2) )
    term2 = np.exp ( t - (t * (3/2)  * np.log (s) ) )
    
    f = term1 * term2 #particle content 
    
    return x, f, event_number, n_max

def composite_gh_param (conex_showers, row, x_limit, pythia_tables, bins = 2000): 
    '''
    > extracts the G-H parameters from a data set, \n
    > returns desired values for a given curve (row) in showers \n
    sconex_showers = 2D np array conex with columns to unpack 
    row = row of the table\n
    x_limit = bin limit, will be divided into 1000 bins\n
    pythia_tables = pythia tables of that specific particle 
    '''
    #read in Conex EAS Data
    pid = conex_showers[row, 0]    #particle ID
    log_e = conex_showers[row, 1]    #log10 of the primary energy in eV
    zenith = conex_showers[row, 2]     #zenith angle in degree
    azimuth = conex_showers[row, 3]    #azimuth angle in degree
    #chi squared / number of degree of freedom / sqrt (n_max) for the fit
    chi2 = conex_showers[row, 10] 
    
    n_max = (conex_showers[row, 4])  
    x_max = conex_showers[row, 5]  #GH fit result for slant depth of the shower maximum (g/cm^2)
    x_0 = conex_showers[row, 6]    # first interaction point for GH fit
    #parameters for the polynomial function of the GH fit
    p1 = conex_showers[row, 7]
    p2 = conex_showers[row, 8]
    p3 = conex_showers[row, 9]

    #read in Pythia Decays Data
    event_number = pythia_tables[row, 0]
    y = pythia_tables[row, -1] # energy decimal quantitiy of 100 PeV
   
    scaled_n_max = float (n_max * y) #n_max scaled by y 

    bins = x_limit
    
    #some of the x_0 values are negative (physically doesn't make sense)
    #this cut addresses that
    
    if x_0 >= 0:
       x = np.linspace( x_0, x_limit, bins)
    else: 
        x =  np.linspace( x_limit/ bins, x_limit, bins)  #slant  Depths 
     
    #calculating gaisser-hillas function
    Lambda = p1 + p2*x + p3*(x**2) 
    
    exp1 = (x_max - x_0) / Lambda
   
    term1 = scaled_n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0)) **exp1 )
    
    exp2 = (x_max - x) / Lambda
    term2 = np.exp(exp2) 
    
    f = np.nan_to_num (term1 * term2)
    
    LambdaAtx_max = p1 + p2*x_max + p3*(x_max**2)
    #t = (x - x_max)/36.62 #shower stage
    return x, f, event_number, scaled_n_max, y, x_max, x_0, LambdaAtx_max 


def composite_eas (conex_showers, pythia_tables, end_row, start_row = 0, bins = 2000, x_limit = 2000, 
                      sum_content_per_event = None, n_max_cut_threshold = None, rebound_plt = None, 
                      regular_plt = None, greisen_comparison = None, average_plt = None):
    '''
    > plots the profiles based on composite GH parameters \n
    > uses the composite_gh_param function \n
    conex_showers = the data to be plotted; np array \n
    start_row =  what row to start \n
    end_row =  what row to end \n
    pythia_tables = where the y scaling values are from, make sure they are the same particle type 
    as conex_showers \n 
    n_max_cut_threshold =  graph the profile up until n_max rebounds for that profile after the peak,
    if empyt just plots until x_limit \n
    rebound_plt = if true, draw the plots, else just returns cutoff slant depths for histogram \n
    regular_plt = 0 returns just the data, 1 plots it instead \n
    greisen_comparison = true, plot the greisen param if regular_plt is also true  \n
    sum_content_per_event = sum that event's composition for ONE type of particle 
    average_plt = true or false, plots the average w/ rms error of all selected rows \n
    '''
    
    if n_max_cut_threshold is not None: 
        # stopping in slant depth once the particle content f is n_max_cut_threshold of n_max
        # note that for each profile, n_max is different 
        rows = [] 
        fs = []
        xs = []   
        untrimmed_xs =[]
        plt.figure(figsize=(8, 5), dpi= 120) 
        for row in range (start_row, end_row + 1):  
            #x, f, n_max, x_max = composite_gh_param(conex_showers, row, x_limit)[0:4] 
            x, f, event_number, scaled_n_max = composite_gh_param (conex_showers = conex_showers, 
                                                       pythia_tables = pythia_tables, 
                                                       row = row, x_limit = x_limit, bins = bins)[0:4]

            untrimmed_xs.append(x)  
            n_max_fraction = scaled_n_max * n_max_cut_threshold
            places_to_cut = np.argwhere(f <= n_max_fraction) #indeces where it is less than 
            idx_near_scaled_n_max = np.argmin(np.absolute(f - scaled_n_max))
            #value of last place where it is still less than 1%
            cut_here = np.amax(places_to_cut) #(after or before peak, depends on x_limit) 
            #trims the arrays from 0 to where number of  particles < n_max_fraction 
            trimmed_f = np.array ( f[0 : cut_here ] ) 
            trimmed_x = np.array ( x[0 : cut_here ] )
            cutoff_depth = np.amax(x) #slant depth when f exceeds n_max_fraction (given x_limit)
            #tag the trimmed and untrimmed array with the event number
            trimmed_f = np.insert(trimmed_f, 0, event_number, axis = 0) 
            
            f = np.insert(f, 0, event_number, axis = 0)
            
            if places_to_cut[-1] > idx_near_scaled_n_max and  f [-2] < f [-1]:
                #print('rebounds')
                #print ('last index to cut', places_to_cut[-1])
                #print('peak occurs closest to this index', idx_near_scaled_n_max)
                #print ('second to last element of trimmed', f [-2]) 
                #print ('last element of trimmed', f [-1])
                trimmed_f = np.pad(trimmed_f, (0, bins - len(trimmed_f)), 
                                   'constant', constant_values = float('nan'))
                trimmed_x = np.pad(trimmed_x, (0, bins - len(trimmed_x)), 
                                   'constant', constant_values = float('nan'))
                
                #trimmed_f = trimmed_f.tolist()
                #trimmed_x = trimmed_x.tolist()
                
                #trimmed_f.extend( [float('nan')] * (1001 - len(trimmed_f)) )
                #trimmed_x.extend( [float('nan')] * (1000 - len(trimmed_x)) )
                
                #trimmed_f = np.nan_to_num (np.array(trimmed_f))
                #trimmed_x = np.nan_to_num (np.array(trimmed_x))
                
                fs.append (trimmed_f)
                xs.append (trimmed_x)
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (event_number) ) +  ')' )
             
            else:
                #print('no rebound')
                #Notes:
                #for rebound the following must be met:
                #the index of the last place to cut is greater than the peak (n_maxFraction) 
                #the curve must be increasing, not decreasing, meaning -2 is less than -1
                #the curve can get cut early (by x_limit) before it exceedes 1%, but 
                #make sure that if x_limit allows the curve to grow greater than 1% that it is cut short. 
                #if it doesn't rebound, just constrain it by x_limit

                #f = f.tolist()
                #x = x.tolist()

                fs.append (trimmed_f)
                xs.append (trimmed_x)
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (event_number) ) + ' - NR)' )
            
        if sum_content_per_event is True: 
            #based on https://stackoverflow.com/a/30041823
            #keep the summed curve going after one of the constituent curves stops, looks jagged.
            #tagged_fs = np.nan_to_num(tagged_fs) 
            
            tagged_fs = np.array(fs)
            sorted_data = tagged_fs[ np.argsort(tagged_fs[:,0]) ,: ]

            row_mask = np.append(np.diff(sorted_data[:,0], axis=0) != 0, [True] )

            cumsum_grps = sorted_data.cumsum(0) [row_mask,1:]
            sum_grps = np.diff(cumsum_grps, axis=0) 

            counts = np.concatenate((cumsum_grps[0,:][None],sum_grps),axis=0)

            summed_content = np.concatenate((sorted_data [row_mask,0] [:,None], counts), axis = 1)

            event_labels = summed_content[:,0]
            #event_labels = event_labels.tolist()
            
            summed_content =  np.delete(summed_content, 0, 1) #deletes the event tag 
            #summed_content = summed_content.tolist()
            '''
            xs = np.array(xs)
            biggestx = np.argmax(xs.max(axis = 1))
            xs_summed_content = xs[biggestx,:].T 
            xs_summed_content = xs_summed_content.tolist()
            '''
            #return fs,xs, summed_content, xs_summed_content

            for event, particleContent in zip(event_labels, summed_content): 
                #check how to implement the bins, same bins for now
                plt.plot(x, particleContent, '--', label = "Summed Event " + str (int (event) ) ) 
       
        if rebound_plt is True:
            
            fs =  np.delete( np.array (fs), 0, 1) #deletes the event tag
            
            for row, f, x in zip(rows, fs, xs): 
                plt.plot(x, f, label = row)      
            
        else: 
            return cutoff_depth           
       
      
    else:
            
        rows = [] 
        fs = []
        xs = [] 

        if regular_plt is not None: 
            # regular plotting routine
    
            for row in range (start_row, end_row + 1): 
                 
                x, f, event_number = composite_gh_param (conex_showers = conex_showers, 
                                                pythia_tables = pythia_tables, 
                                                row = row, x_limit = x_limit, bins = bins)[0:3]
                f = np.insert(f , 0, event_number, axis = 0)
                x = np.insert(x , 0, event_number, axis = 0)
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (event_number) ) +  ')' )   
                fs.append (f)
                xs.append (x)
                
                if greisen_comparison is True:
                    Greisenx, Greisenf, Greisenevent_number = greisen_param (conex_showers = conex_showers, 
                                                    pythia_tables = pythia_tables, 
                                                    row = row, x_limit = x_limit)[0:3]
                    Greisenf = np.insert(Greisenf , 0, Greisenevent_number, axis = 0)
                    Greisenx = np.insert(Greisenx, 0, event_number, axis = 0)
                    
                    rows.append ( 'Greisen Row ' + str(row) + ' (Event ' + str ( int (event_number) ) +  ')' )
                    fs.append (Greisenf)
                    xs.append (Greisenx)
            
            if regular_plt == 1: 
                
                plt.figure(figsize=(8, 5), dpi= 120)

                fs =  np.delete(np.array(fs), 0, 1)
                xs = np.delete(np.array(xs), 0, 1)
                
     
                for row, x, f in zip(rows, xs, fs):
                    plt.plot(x, f, label = row) 
            
            elif regular_plt == 0: 
                
                return np.array(xs), np.array(fs), np.array(rows)
    
        if average_plt is True:            
            # returns an average plot ONLY given rows of a file for one particle type and rms error
           
            for row in range (start_row, end_row + 1): 
                x, f, event_number= composite_gh_param (conex_showers = conex_showers, 
                                            pythia_tables = pythia_tables, 
                                            row = row, x_limit = x_limit, bins = bins )[0:3]
                
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (event_number) ) +  ')' )   
                fs.append (f)
                xs.append (x)
                
            fs = np.array(fs).T   #Makes the list of fs into a 2D matrix
            average_fs = fs.mean(axis = 1) #Takes the mean of each row (each bin)
            
            # to get RMS
            fs_square = np.square (fs) #squares the 2D matrix  
            mean_fs = fs_square.mean(axis = 1) #gets mean along each bin
            rms_fs =  np.sqrt(mean_fs) #gets root of each mean
            
            #draws average plot
            plt.plot(x, average_fs,  '--k' , label = 'Average (Row ' + str (start_row) + 
                     ' to Row ' + str (end_row) + ')')
            
            #here is a crude measurement for error, which is the RMS minus the average value
            #rms_error = rms_fs - average_fs 
            
            #here is a better definition for rms_error
            #to get RMSE where the average is the estimated series and particleContent are values
            rms_error = np.sqrt(np.mean( ( average_fs  - fs )**2, axis = 0 ))
            
            #draws error
            plt.fill_between(x, average_fs - rms_error, average_fs + rms_error,
                             alpha = 0.5, edgecolor='red', facecolor='crimson', 
                             label = 'Error')
        
        if sum_content_per_event is True:
            rows = [] 
            fs = []
            xs = [] 
        
            for row in range (start_row, end_row + 1): 
                x, f, event_number = composite_gh_param (conex_showers = conex_showers, 
                                              pythia_tables = pythia_tables, 
                                              row = row, x_limit = x_limit, bins = bins )[0:3] 
                
                f = np.insert(f, 0, event_number, axis = 0)
                
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (event_number) ) +  ')' )   
                fs.append (f)
                xs.append (x)
            #based on https://stackoverflow.com/a/30041823
            tagged_fs = np.array(fs)
    
            sorted_data = tagged_fs[ np.argsort(tagged_fs[:,0]) ,: ]
            
            row_mask = np.append(np.diff(sorted_data[:,0], axis=0) != 0, [True] )
            
            cumsum_grps = sorted_data.cumsum(0) [row_mask,1:]
            sum_grps = np.diff(cumsum_grps, axis=0) 
            
            counts = np.concatenate((cumsum_grps[0,:][None],sum_grps),axis=0)
            
            summed_content = np.concatenate((sorted_data [row_mask,0] [:,None], counts), axis = 1)
            
            event_labels = summed_content[:,0]
            #stack before converting to list
            #event_labels = event_labels.tolist()
            
            summed_content =  np.delete(summed_content, 0, 1)  # delete first column 
            
            #summed_content = summed_content.tolist()
            
            for event, particleContent, x  in zip(event_labels, summed_content, xs): 
                
                plt.plot(x, particleContent,'--', label = "Summed Event " + str (int (event) ) )
                


def composite_plotter (start_row, end_row, event_labels, event_bins, particle_content,
                      average_plt = None, composite_plt = None ): 
    '''
    > a specialized version of composite_eas() used only for average and rms plots as well as
    composite profiles given event summed data (content_per_event() on how to get composites).\n 
    
    '''
    event_labels = event_labels [start_row:end_row + 1]
    event_bins = event_bins  [start_row:end_row + 1 ,:]
    particle_content = particle_content [start_row:end_row + 1 ,:]
           
        
    
    
    # returns an average plot  and rms error ONLY given rows of a file for one particle type
    if average_plt is not None:
        #takes mean along the rows 
        average = particle_content.mean(axis = 0) 
        #to get RMS
        squared = np.square (particle_content) #squares the 2D matrix  
        mean = squared.mean(axis = 0)  #gets mean along each bin
        rms =  np.sqrt(mean) #gets root of each mean
        
        #draws average plot
        plt.figure(figsize=(8, 5), dpi= 120) 
        plt.plot(event_bins[0,:], average,  '--k', 
                 label = 'Average of ' + str(np.shape(particle_content)[0]) + ' Events')

       
        #here is a crude measurement for error, which is the RMS minus the average value
        #rms_error = rms - average 
        
        #here is a better definition for rms_error
        #to get RMSE where the average is the estimated series and particle_content are values
        rms_error = np.sqrt(np.mean( ( average - particle_content )**2, axis = 0 ))
        
        plt.fill_between( event_bins[0,:], average - rms_error, average + rms_error,
                         alpha = 0.5, edgecolor='red', facecolor='crimson', 
                         label = 'RMS Error')
    
    if composite_plt is not None:
        plt.figure(figsize=(8, 5), dpi= 120) 
        for event, event_id, shower_content,slant_depths  in \
                        zip(event_labels[:,0], event_labels[:,1], particle_content, event_bins): 
            
            plt.plot(slant_depths, shower_content,'-', label = "Summed Event " + str (int (event) ) 
                     + ' (' + str(int(event_id)) + ')' ) 

  
def content_per_event (pythia_decays, just_decay_codes = False, average_bins = None, return_std = None,
                       **kwargs):
    '''
    > takes any amount of pre-filtered particle data-- particle content-- and sums them per event,
    given that they have an event tag in the beginning of each row. \n
    > if you want decay code only-- no event tags-- (e.g., 300001), set just_decay_codes = True \n
    > particle_contents = shower contents/ sharged particles as a funtcion of slant depth \n 
    > slant_depths = corresponding slant depths
    '''
    particle_fs = kwargs.get('particle_contents')
    master_particle_contents  = np.concatenate((particle_fs), axis = 0)
    master_particle_contents  = master_particle_contents [ np.argsort(master_particle_contents [:,0]) ,: ]

    particle_xs = kwargs.get('slant_depths')
    master_bins = np.concatenate((particle_xs), axis = 0)
    master_bins =  master_bins[ np.argsort(master_bins[:,0]) ,: ] 

    #sum master events for each event 
    row_mask = np.append( np.diff(master_particle_contents [:, 0], axis = 0) != 0, [True] )
    cumsum_grps = master_particle_contents .cumsum(0) [row_mask,1:]
    sum_grps = np.diff(cumsum_grps, axis = 0) 
    counts = np.concatenate( (cumsum_grps[0,:][None], sum_grps), axis = 0)
    summed_content = np.concatenate((master_particle_contents  [row_mask,0] [:,None], counts), axis = 1)
    #average of bins per event, since each EAS in composite EAS may have non-identical bins
    if average_bins is not None: 
    #based on https://stackoverflow.com/a/54112125
        #sort array by groupby column
        b = master_bins
        #get interval indices for the sorted groupby col
        idx = np.flatnonzero(np.r_[True,b[:-1,0]!=b[1:,0], True])\
        #get counts of each group and sum rows based on the groupings & hence averages
        grpcounts = np.diff(idx)  
        avg = np.add.reduceat(b[:,1:],idx[:-1],axis=0)/grpcounts.astype(float)[:,None]
        #concatenate for the output in desired format
        event_bins = np.c_[b[idx[:-1],0],avg]
    else:    
        event_bins = np.unique(master_bins, axis = 0)
        
    decay_codes_event_numbers = np.delete(pythia_decays, np.s_[2:5], axis = 1) 
    decay_codes_event_numbers = np.unique(decay_codes_event_numbers, axis = 0)
    
    filtered_codes_numbers = decay_codes_event_numbers[np.isin(decay_codes_event_numbers [:,0], 
                                                               summed_content[:,0])] 
    decay_codes =  filtered_codes_numbers[:,1]
    
    untagged_particle_xs = event_bins
    untagged_particle_fs = summed_content
    
    if just_decay_codes is False: #outputs event number and decay code
        return untagged_particle_xs, untagged_particle_fs, filtered_codes_numbers
    else: #outputs decay code only 
        return event_bins, summed_content, decay_codes


def bin_nmax_xmax (bins, particle_content):
    '''
    > given an array of Slant Depths and Particle Content values for the same particle 
    (can be any number of events, but need to be same size), returns the Nmax and Xmax Values 
    per row (if composite showers and bins are inputted, per event) \n
    > intended to use for nmax and xmax distribution analysis \n
    '''

    try:
        bin_nmax = np.amax(particle_content, axis = 1) 
        bin_nmax_pos =  np.nanargmax(particle_content, axis = 1)
        bin_xmaxs = bins[np.arange(len(bins)), bin_nmax_pos]
    except: 
        bin_nmax = np.amax(particle_content) 
        bin_nmax_pos =  np.nanargmax(particle_content)
        bin_xmaxs = bins[bin_nmax_pos]
    
    return bin_nmax,  bin_xmaxs


def bin_nmax_xmax_rising (bins, particle_content, n_max_fraction: float): 
    '''
    gets the rising edge value of nmax*n_max_fraction and corresponding xmax
    '''
    particle_content = np.copy(particle_content) #copy the data
    
    try:
        #get the maximum particle content value per event(row)
        nmax = np.amax(particle_content, axis = 1) 
        #get the corresponding positions
        nmax_pos =  np.array ( np.argmax(particle_content, axis = 1) )
        #get the corresponding n_max_fraction per row
        nmax_fraction = nmax * n_max_fraction 
        #turn the entries before the nmax idx for a given row to nan
        particle_content[np.array(nmax_pos)[:,None]  < np.arange(particle_content.shape[1])] = np.nan
        #get the residuals by subtracting the particle content by the nmax_fraction
        residuals_pos1 = np.abs(particle_content.T  - nmax_fraction); residuals_pos1 = residuals_pos1.T
        #smallest risidual indicates the index of the value closes to nmax
        position1 = np.nanargmin(residuals_pos1, axis = 1 )
        #gets the corresponding values given the wanted index, per row
        nmax_pos1 = particle_content[np.arange(len(particle_content)), position1]
        xmax_pos1 = bins[np.arange(len(bins)), position1]
        
    except: 
        #if it's not an array
        nmax = np.amax(particle_content) 
        nmax_pos =  np.array ( np.argmax(particle_content) ) 
        nmax_fraction = nmax * n_max_fraction 
        particle_content[np.array(nmax_pos)[:,None]  < np.arange(particle_content.shape[1])] = np.nan
        residuals_pos1 = np.abs(particle_content.T  - nmax_fraction); residuals_pos1 = residuals_pos1.T
        position1 = np.nanargmin(residuals_pos1)
        nmax_pos1 = particle_content[np.arange(len(particle_content)), position1]
        xmax_pos1 = bins[np.arange(len(bins)), position1]

    return nmax_pos1,  xmax_pos1


def bin_nmax_xmax_falling (bins, particle_content, n_max_fraction: float): 
    '''
    gets the falling edge value of nmax*n_max_fraction and corresponding xmax
    '''
    particle_content = np.copy(particle_content) #copy the data
    
    try:
         nmax = np.amax(particle_content, axis = 1) 
         nmax_pos =  np.array ( np.argmax(particle_content, axis = 1) ) 
         nmax_fraction = nmax * n_max_fraction 
         particle_content[np.array(nmax_pos)[:,None]  > np.arange(particle_content.shape[1])] = np.nan
         residuals_pos2 = np.abs(particle_content.T  - nmax_fraction); residuals_pos2 = residuals_pos2.T
         position2 = np.nanargmin(residuals_pos2, axis = 1 )
         nmax_pos2 = particle_content[np.arange(len(particle_content)), position2]
         xmax_pos2 = bins[np.arange(len(bins)), position2]
         
    except:
        #if it's not an array
         nmax = np.amax(particle_content) 
         nmax_pos =  np.array ( np.argmax(particle_content) ) 
         nmax_fraction = nmax * n_max_fraction 
         particle_content[np.array(nmax_pos)[:,None]  > np.arange(particle_content.shape[1])] = np.nan
         residuals_pos2 = np.abs(particle_content.T  - nmax_fraction); residuals_pos2 = residuals_pos2.T
         position2 = np.nanargmin(residuals_pos2)
         nmax_pos2 = particle_content[np.arange(len(particle_content)), position2]
         xmax_pos2 = bins[np.arange(len(bins)), position2] 
         
    return nmax_pos2,  xmax_pos2

   
        