import numpy as np
import matplotlib.pyplot as plt
import h5py
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

def padRow(array, fillval=np.nan):
    '''
    > pads an ununiform array with Nans 
    '''
    lens = np.array([len(item) for item in array])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(array)
    return out    
        

def emGreisen (conexShowers, row, xlim, pythiaTables): 
    '''
    > uses the Greisen parametrization 
    > Cosmic Rays and Particle Physics by Thomas K. Gaisser, eq 15.28
    > depends solely on electron energies from Pythia Tables 
    '''
    GH_Nmax = (conexShowers[row, 4])  #Nmax scaled by y 
    
    #Read in Pythia Decays Data
    eventNum = pythiaTables[row, 0]
    y = pythiaTables[row, -1] # energy decimal quantitiy of 100 PeV
    scaled_GH_Nmax = GH_Nmax * y 
    
    bins = xlim
   
    X = np.linspace( xlim/ bins, xlim, bins)   #slant  Depth 
    
    #source: https://pdg.lbl.gov/2015/AtomicNuclearProperties/HTML/air_dry_1_atm.html
    #Atomic and nuclear properties of air (dry, 1 atm)
    t = X / 36.62 # t = X/X0, where X0 is the radiation length in air ~ 37 g/cm^2
    E_c = 87.92e6 # critical energy for electromagnetic cascades in the air
    
    E_0 = (y * 100e15) # energy of the photon initiated shower in eV pg 303 from Gaisser's Book
    #E_c = Nmax * E_0 
    Beta_0 = np.log(E_0 / E_c)
    
    s = (0 + 3*t)/ (t + 2*Beta_0 ) #age parameter, n = 0
    Nmax = ( 0.31 / pow(Beta_0, 1/2) ) * (E_0 / E_c)

    term1 = ( 0.31 / pow(Beta_0, 1/2) )
    term2 = np.exp ( t - (t * (3/2)  * np.log (s) ) )
    
    f = term1 * term2 #particle content 
    
    return X, f, eventNum, Nmax

def compositeGH (conexShowers, row, xlim, pythiaTables): 
    '''
    > extracts the G-H parameters from a data set, \n
    > returns desired values for a given curve (row) in showers \n
    sconexShowers = 2D np array conex with columns to unpack 
    row = row of the table\n
    xlim = bin limit, will be divided into 1000 bins\n
    pythiaTables = pythia tables of that specific particle 
    '''
    #Read in Conex EAS Data
    PID = conexShowers[row, 0]    #particle ID
    lgE = conexShowers[row, 1]    #log10 of the primary energy in eV
    zenith = conexShowers[row, 2]     #zenith angle in degree
    azimuth = conexShowers[row, 3]    #azimuth angle in degree
    
    Nmax = (conexShowers[row, 4])  
    Xmax = conexShowers[row, 5]  #GH fit result for slant depth of the shower maximum (g/cm^2)
    X0 = conexShowers[row, 6]    # first interaction point for GH fit
    
    #parameters for the polynomial function of the GH fit
    p1 = conexShowers[row, 7]
    p2 = conexShowers[row, 8]
    p3 = conexShowers[row, 9]
    #Chi squared / number of degree of freedom / sqrt (Nmax) for the fit
    chi2 = conexShowers[row, 10] 
    
    #Read in Pythia Decays Data
    eventNum = pythiaTables[row, 0]
    y = pythiaTables[row, -1] # energy decimal quantitiy of 100 PeV
   
    scaled_Nmax = float (Nmax * y) #Nmax scaled by y 

    bins = xlim
    
    #some of the X0 values are negative (physically doesn't make sense), this cut addresses that
    
    if X0 >= 0:
       X = np.linspace( X0, xlim, bins)
    else: 
        X =  np.linspace( xlim/ bins, xlim, bins)  #slant  Depth 
     
    #calculating gaisser-hillas function
    Lambda = p1 + p2*X + p3*(X**2) 
    
    exp1 = (Xmax - X0) / Lambda
   
    term1 = scaled_Nmax * np.nan_to_num ( pow( (X - X0) / (Xmax - X0), exp1) )
    
    exp2 = (Xmax - X) / Lambda
    term2 = np.exp(exp2) 
    
    f = np.nan_to_num (term1 * term2)
    
    LambdaAtXmax = p1 + p2*Xmax + p3*(Xmax**2)

    return X, f, eventNum, scaled_Nmax, y, Xmax, X0, LambdaAtXmax 


def compositeProfile (conexShowers, pythiaTables, rowStart, rowEnd, xlimit, 
                      eventSum = None, nmaxThreshold = None, thresholdPlot = None, 
                      regPlot = None, greiComp = None, avgPlot = None):
    '''
    > plots the profiles based on composite GH parameters \n
    > uses the compositeGH function \n
    conexShowers = the data to be plotted; np array \n
    rowStart =  what row to start \n
    rowEnd =  what row to end \n
    pythiaTables = where the y scaling values are from, make sure they are the same particle type 
    as conexShowers \n 
    nmaxThreshold =  graph the profile up until Nmax rebounds for that profile after the peak,
    if empyt just plots until xlimit \n
    thresholdPlot = if true, draw the plots, else just returns cutoff slant depths for histogram \n
    regPlot = 0 returns just the data, 1 plots it instead \n
    greiComp = true, plot the greisen param if regPlot is also true  \n
    eventSum = sum that event's composition for ONE type of particle 
    avgPlot = true or false, plots the average w/ rms error of all selected rows \n
    '''
    
    if nmaxThreshold is not None: 
        # stopping in slant depth once the particle content f is nmaxThreshold of Nmax
        # note that for each profile, Nmax is different 
        rows = [] 
        fs = []
        Xs = []   
        
        untrimmedXs =[]
        plt.figure(figsize=(8, 5), dpi= 120) 
        for row in range (rowStart, rowEnd + 1):  
            
            #X, f, Nmax, Xmax = compositeGH(conexShowers, row, xlimit)[0:4] 
            X, f, eventNum, scaled_Nmax = compositeGH (conexShowers = conexShowers, 
                                                       pythiaTables = pythiaTables, 
                                                       row = row, xlim = xlimit )[0:4]

            untrimmedXs.append(X)  
           
            Nmax_fraction = scaled_Nmax * nmaxThreshold
            
            places_to_cut = np.argwhere(f <= Nmax_fraction) #indeces where it is less than 
            
            #locationClosetoNmax = np.argwhere (  np.isclose (f, scaled_Nmax))
            locationClosetoNmax = np.argmin(np.absolute(f - scaled_Nmax))

            #value of last place where it is still less than 1%
            cut_here = np.amax(places_to_cut) #(after or before peak, depends on xlimit) 
            
            #trims the arrays from 0 to where number of  particles < Nmax_fraction 
            trimmedf = np.array ( f[0 : cut_here ] ) 
            trimmedX = np.array ( X[0 : cut_here ] )
            
            cutoff_depth = np.amax(X) #slant depth when f exceeds Nmax_fraction (given xlimit)
            
            #tag the trimmed and untrimmed array with the event number
            trimmedf = np.insert(trimmedf, 0, eventNum, axis = 0) 
            
            f = np.insert(f, 0, eventNum, axis = 0)
            
            if places_to_cut[-1] > locationClosetoNmax and  f [-2] < f [-1]:
                #print('rebounds')
                #print ('last index to cut', places_to_cut[-1])
                #print('peak occurs closest to this index', locationClosetoNmax)
                #print ('second to last element of trimmed', f [-2]) 
                #print ('last element of trimmed', f [-1])
                
                trimmedf = trimmedf.tolist()
                trimmedX = trimmedX.tolist()
                
                trimmedf.extend( [float('nan')] * (1001 - len(trimmedf)) )
                trimmedX.extend( [float('nan')] * (1000 - len(trimmedX)) )
                
                #trimmedf = np.nan_to_num (np.array(trimmedf))
                #trimmedX = np.nan_to_num (np.array(trimmedX))
                
                fs.append (trimmedf)
                Xs.append (trimmedX)
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (eventNum) ) +  ')' )
             
            else:
                #print('no rebound')
                #Notes:
                #for rebound the following must be met:
                #the index of the last place to cut is greater than the peak (nmaxFraction) 
                #the curve must be increasing, not decreasing, meaning -2 is less than -1
                #the curve can get cut early (by xlim) before it exceedes 1%, but 
                #make sure that if xlim allows the curve to grow greater than 1% that it is cut short. 
                #if it doesn't rebound, just constrain it by xlimit

                f = f.tolist()
                X = X.tolist()

                fs.append (f)
                Xs.append (X)
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (eventNum) ) + ' - NR)' )
            
        if eventSum is True: 
            #based on https://stackoverflow.com/a/30041823
            #keep the summed curve going after one of the constituent curves stops, looks jagged.
            #fsWithEventTag = np.nan_to_num(fsWithEventTag) 
            
            fsWithEventTag = np.array(fs)
            sortedData = fsWithEventTag[ np.argsort(fsWithEventTag[:,0]) ,: ]

            row_mask = np.append(np.diff(sortedData[:,0], axis=0) != 0, [True] )

            cumsum_grps = sortedData.cumsum(0) [row_mask,1:]
            sum_grps = np.diff(cumsum_grps, axis=0) 

            counts = np.concatenate((cumsum_grps[0,:][None],sum_grps),axis=0)

            summedParticleContent = np.concatenate((sortedData [row_mask,0] [:,None], counts), axis = 1)

            eventLabels = summedParticleContent[:,0]
            eventLabels = eventLabels.tolist()
            
            summedParticleContent =  np.delete(summedParticleContent, 0, 1) #deletes the event tag 
            summedParticleContent = summedParticleContent.tolist()
            '''
            Xs = np.array(Xs)
            biggestX = np.argmax(Xs.max(axis = 1))
            Xs_summedParticleContent = Xs[biggestX,:].T 
            Xs_summedParticleContent = Xs_summedParticleContent.tolist()
            '''
            #return fs,Xs, summedParticleContent, Xs_summedParticleContent

            for event, particleContent in zip(eventLabels, summedParticleContent): 
                #check how to implement the bins, same bins for now
                plt.plot(X, particleContent, '--', label = "Summed Event " + str (int (event) ) ) 
       
        if thresholdPlot is True:
            
            fs =  np.delete( np.array (fs), 0, 1) #deletes the event tag
            
            for row, f, X in zip(rows, fs, Xs): 
                plt.plot(X, f, label = row)      
            
        else: 
            return cutoff_depth           
       
      
    else:
            
        rows = [] 
        fs = []
        Xs = [] 

        if regPlot is not None: 
            # regular plotting routine
    
            for row in range (rowStart, rowEnd + 1): 
                 
                X, f, eventNum = compositeGH (conexShowers = conexShowers, 
                                                pythiaTables = pythiaTables, 
                                                row = row, xlim = xlimit )[0:3]
                f = np.insert(f , 0, eventNum, axis = 0)
                X = np.insert(X , 0, eventNum, axis = 0)
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (eventNum) ) +  ')' )   
                fs.append (f)
                Xs.append (X)
                
                if greiComp is True:
                    GreisenX, Greisenf, GreisenEventNum = emGreisen (conexShowers = conexShowers, 
                                                    pythiaTables = pythiaTables, 
                                                    row = row, xlim = xlimit)[0:3]
                    Greisenf = np.insert(Greisenf , 0, GreisenEventNum, axis = 0)
                    GreisenX = np.insert(GreisenX, 0, eventNum, axis = 0)
                    
                    rows.append ( 'Greisen Row ' + str(row) + ' (Event ' + str ( int (eventNum) ) +  ')' )
                    fs.append (Greisenf)
                    Xs.append (GreisenX)
            
            if regPlot == 1: 
                
                plt.figure(figsize=(8, 5), dpi= 120)

                fs =  np.delete(np.array(fs), 0, 1)
                Xs = np.delete(np.array(Xs), 0, 1)
                
     
                for row, X, f in zip(rows, Xs, fs):
                    plt.plot(X, f, label = row) 
            
            elif regPlot == 0: 
                
                return Xs, fs, rows
    
        if avgPlot is True:            
            # returns an average plot ONLY given rows of a file for one particle type and rms error
           
            for row in range (rowStart, rowEnd + 1): 
                X, f, eventNum= compositeGH (conexShowers = conexShowers, 
                                            pythiaTables = pythiaTables, 
                                            row = row, xlim = xlimit )[0:3]
                
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (eventNum) ) +  ')' )   
                fs.append (f)
                Xs.append (X)
                
            fs = np.array(fs).T   #Makes the list of fs into a 2D matrix
            average_fs = fs.mean(axis = 1) #Takes the mean of each row (each bin)
            
            # to get RMS
            fs_square = np.square (fs) #squares the 2D matrix  
            mean_fs = fs_square.mean(axis = 1) #gets mean along each bin
            rms_fs =  np.sqrt(mean_fs) #gets root of each mean
            
            #draws average plot
            plt.plot(X, average_fs,  '--k' , label = 'Average (Row ' + str (rowStart) + 
                     ' to Row ' + str (rowEnd) + ')')
            
            #here is a crude measurement for error, which is the RMS minus the average value
            #rms_error = rms_fs - average_fs 
            
            #here is a better definition for rms_error
            #to get RMSE where the average is the estimated series and particleContent are values
            rms_error = np.sqrt(np.mean( ( average_fs  - fs )**2, axis = 0 ))
            
            #draws error
            plt.fill_between(X, average_fs - rms_error, average_fs + rms_error,
                             alpha = 0.5, edgecolor='red', facecolor='crimson', 
                             label = 'Error')
        
        if eventSum is True:
            rows = [] 
            fs = []
            Xs = [] 
        
            for row in range (rowStart, rowEnd + 1): 
                X, f, eventNum = compositeGH (conexShowers = conexShowers, 
                                              pythiaTables = pythiaTables, 
                                              row = row, xlim = xlimit )[0:3] 
                
                f = np.insert(f, 0, eventNum, axis = 0)
                
                rows.append ( 'Row ' + str(row) + ' (Event ' + str ( int (eventNum) ) +  ')' )   
                fs.append (f)
                Xs.append (X)
            #based on https://stackoverflow.com/a/30041823
            fsWithEventTag = np.array(fs)
    
            sortedData = fsWithEventTag[ np.argsort(fsWithEventTag[:,0]) ,: ]
            
            row_mask = np.append(np.diff(sortedData[:,0], axis=0) != 0, [True] )
            
            cumsum_grps = sortedData.cumsum(0) [row_mask,1:]
            sum_grps = np.diff(cumsum_grps, axis=0) 
            
            counts = np.concatenate((cumsum_grps[0,:][None],sum_grps),axis=0)
            
            summedParticleContent = np.concatenate((sortedData [row_mask,0] [:,None], counts), axis = 1)
            
            eventLabels = summedParticleContent[:,0]
            #stack before converting to list
            eventLabels = eventLabels.tolist()
            
            summedParticleContent =  np.delete(summedParticleContent, 0, 1)  # delete first column 
            
            summedParticleContent = summedParticleContent.tolist()
            
            for event, particleContent, X  in zip(eventLabels, summedParticleContent, Xs): 
                
                plt.plot(X, particleContent,'--', label = "Summed Event " + str (int (event) ) )
                


def compositePlotter (rowStart, rowEnd, eventLabels, eventBins, particleContent,
                      avgPlot = None, compositePlot = None ): 
    '''
    > a specialized version of compositeProfile() used only for average and rms plots as well as
    composite profiles given event summed data (see contentPerEvent() on how to get composites).\n 
    > sample use: takes in an array of eventLables, bins with no event tags, particle content 
    with no event tags summed per event (already composite), and plots 4 rows from the matrices. \n
    compositePlotter (rowStart = 0, rowEnd = 3, eventLabels = eventLabels, 
    eventBins = binsPerEventNoTag, particleContent = summedContentNoTag, compositePlot = True)
    
    '''
    eventLabels = eventLabels [rowStart:rowEnd + 1]
    eventBins = eventBins  [rowStart:rowEnd + 1 ,:]
    particleContent = particleContent [rowStart:rowEnd + 1 ,:]
           
    # returns an average plot  and rms error ONLY given rows of a file for one particle type
       
    if avgPlot is not None:
        #takes mean along the rows 
        average = particleContent.mean(axis = 0) 
        #to get RMS
        squared = np.square (particleContent) #squares the 2D matrix  
        mean = squared.mean(axis = 0)  #gets mean along each bin
        rms =  np.sqrt(mean) #gets root of each mean
        
        #draws average plot
        plt.figure(figsize=(8, 5), dpi= 120) 
        plt.plot(eventBins[0,:], average,  '--k', 
                 label = 'Average of ' + str(np.shape(particleContent)[0]) + ' Events')

       
        #here is a crude measurement for error, which is the RMS minus the average value
        #rms_error = rms - average 
        
        #here is a better definition for rms_error
        #to get RMSE where the average is the estimated series and particleContent are values
        rms_error = np.sqrt(np.mean( ( average - particleContent )**2, axis = 0 ))
        
        plt.fill_between( eventBins[0,:], average - rms_error, average + rms_error,
                         alpha = 0.5, edgecolor='red', facecolor='crimson', 
                         label = 'RMS Error')
    
    if compositePlot is not None:
        plt.figure(figsize=(8, 5), dpi= 120) 
        for event, ID, Content, X  in zip(eventLabels[:,0], eventLabels[:,1], 
                                          particleContent, eventBins): 
            
            plt.plot(X, Content,'--', label = "Summed Event " + str (int (event) ) 
                     + ' (' + str(int(ID)) + ')' ) 


def NmaxANDXmax (particleContent, slantDepths):
    '''
    > given an array of Slant Depths and Particle Content values for the same particle 
    (can be any number of events, but need to be same size), returns the Nmax and Xmax Values 
    per row; if composite showers and bins are inputted, per event. \n
    > intended for use with paramHist() from the Conex gh_macros 
    '''
    compositeNmaxs = np.amax(particleContent, axis = 1) 
    compositeNmaxPositions =  np.argmax(particleContent, axis = 1)
    compositeXmaxs = slantDepths[np.arange(len(slantDepths)), compositeNmaxPositions]
    
    return compositeNmaxs,  compositeXmaxs
    
def contentPerEvent (pythiaDecays, noEventTags = None, avgBins = None, **kwargs):
    '''
    > takes any amount of pre-filtered particle data (particle content) and sums them per event,
    given that they have an event tag in the beginning of each row. \n 
    > particleFs handles the particle content and Particle Xs handles the bins. \n 
    > requires raw pythiaDecay tables for event decay code information. \n
    > can avgBins if bins are not the same per particle per event due to the X0 > 0 restriction.  
    '''
    particleFs = kwargs.get('particleFs')
    masterParticles  = np.concatenate((particleFs), axis = 0)
    masterParticles  = masterParticles [ np.argsort(masterParticles [:,0]) ,: ]

    particleXs = kwargs.get('particleXs')
    masterBins = np.concatenate((particleXs), axis = 0)
    masterBins =  masterBins[ np.argsort(masterBins[:,0]) ,: ] 

    #sum master events for each event 
    row_mask = np.append( np.diff(masterParticles [:, 0], axis = 0) != 0, [True] )
    cumsum_grps = masterParticles .cumsum(0) [row_mask,1:]
    sum_grps = np.diff(cumsum_grps, axis = 0) 
    counts = np.concatenate( (cumsum_grps[0,:][None], sum_grps), axis = 0)
    summedParticleContent = np.concatenate((masterParticles  [row_mask,0] [:,None], counts), axis = 1)

    if avgBins is not None: #average of bins per event
    #based on https://stackoverflow.com/a/54112125
        #sort array by groupby column
        b = masterBins
        #get interval indices for the sorted groupby col
        idx = np.flatnonzero(np.r_[True,b[:-1,0]!=b[1:,0], True])
        #get counts of each group and sum rows based on the groupings & hence averages
        grpcounts = np.diff(idx)
        avg = np.add.reduceat(b[:,1:],idx[:-1],axis=0)/grpcounts.astype(float)[:,None]
        #concatenate for the output in desired format
        binsPerEvent = np.c_[b[idx[:-1],0],avg]
    else:    
        binsPerEvent = np.unique(masterBins, axis = 0)
        
    EventCodes = np.delete(pythiaDecays, np.s_[2:5], axis = 1) 
    EventCodes = np.unique(EventCodes, axis = 0)
    
    eventLabels = EventCodes[np.isin(EventCodes [:,0], summedParticleContent[:,0])] 
    particleEventCodes =  eventLabels[:,1]
    
    binsPerEventNoTag = np.delete(binsPerEvent, 0, 1)
    summedContentNoTag =  np.delete(summedParticleContent, 0, 1)
    
    if noEventTags is not None: #outputs tag and decay code
        return binsPerEventNoTag, summedContentNoTag, eventLabels
    else: #outputs decay code only 
        return binsPerEvent, summedParticleContent, particleEventCodes
   
        