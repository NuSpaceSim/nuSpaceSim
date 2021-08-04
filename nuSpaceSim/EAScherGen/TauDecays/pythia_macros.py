import numpy as np
import h5py


def decayTabletoHDF5 (fileName, outputName, datasetName) : 
    '''
    > takes a pythia machine-readable text/ ascii file and converts it to hdf5 \n 
    > each daughter particle (row) is tagged with the the event # and decay code along with \n
    > other information \n
    > see https://drive.google.com/file/d/1MVj0FhWNI-075oZQwM8NWSwateT0xqJH/view?usp=sharing \n 
  
    > takes string type arguments\n
    
    fileName = Name of txt/ ascii file to be converted \n
    outputName = name of file with .h5 extension \n 
    datasetName = name of key (data set table) inside the .h5 file \n 
    
    the h5 tables have these mappings \n 
    [:0] = Event #  \n
    [:1] = Decay Code  \n
    [:2] = PID of daughter particle \n
    [:3] = y or mother particle \n
    [:4] = fractional energy (e.g., 100PeV * float) of the daughter particle  \n
    '''
    data = [] 
    with open(fileName) as f: #does this line by line, with output conventions used as clasifier.
        for line in f:
            if '#'  in line:
                event_number =  int (line [line.find('#')+ 1:])
                
            elif 'Event' not in line and '     0    0 0' not in line and len(line) == 9: 
                decay_code = int (line)
                
            elif '     0    0 0' not in line and 'Event' not in line and len(line) != 9:
                pid, mother, energy = [float(n) for n in line.split()]  
                
                data.append([event_number, decay_code, pid, mother, energy]) 
    
    dataArray = np.array(data)
    
    newData = h5py.File(outputName, 'w')
    newData.create_dataset(datasetName, data = dataArray)
    newData.close()

def pythiaReader (decays, row, input_PID): 
    '''
    > returns the energy values (and other information) associated with a particle \n
    > the following are some PIDs \n      
    \n 
    pion = (+/-) 211 \n
    kaon = (+/-) 321 \n
    electron = 11  \n
    muon = (-) 14  \n
    gamma = 22  \n
    '''
    eventNumber = decays [row, 0]
    decayCode = decays [row, 1]
    pid = decays [row, 2]
    mother = decays [row, 3]
    energy = decays [row, 4]
    
    if input_PID != 11: 
        #change this to filter + or - particles
        if pid == float(input_PID) or pid == -float(input_PID) :
            return  int (eventNumber), int(decayCode), int(pid), int(mother), energy
    
    elif input_PID == 11: #get only electrons, ignore positrons (-11)
        if pid == float(input_PID):
            return  int (eventNumber), int(decayCode), int(pid), int(mother), energy
    else: 
        return None 
 
    
 
def eventFilter (particleData, occurNum, energySum, crossFilter, crossFilterData, returnAll = None):
    '''
    > filters certain events that have certain number of occurrences of a certain daughter particle \n
    > can also impose a crossFilter that can return only events that have two 
    > certain daughter particles \n 
    > returns a vector containing filtered energy entries \n
    \n
    particleData = list to be filtered containing energies of a specific particle type \n
    occurNum = events with this number of daughter particle products will be passed through 
    can be 'all' \n 
    crossFilter = when True, cross filters with another particle type or not such that 
    the energies returned will be restricted by the occurNum AND if that event also has a particle 
    of certain type \n 
    crossFilterData = Data of the particle used to crossFilter 
    
    '''
    data = np.array(particleData)

    #unique tuples with number of instances for each event number 
    #event numbers have in duplicates in original data array 
    UniqueCounts = list( zip (*np.unique (data [:,0], return_counts = True) ) ) 
    
    #return event numbers with this many instances using various filters
    try:
        if 0 < occurNum < 5:
            EventsInstance = [event for (event, instances) in UniqueCounts if instances == occurNum]
        elif occurNum >= 5:
            EventsInstance = [event for (event, instances) in UniqueCounts if instances >= 5]
    except:
        if occurNum.lower() == 'all': 
            EventsInstance = [event for (event, instances) in UniqueCounts if instances >= 1]
        else: 
            EventsInstance = [event for (event, instances) in UniqueCounts if instances >= 2]

    #return the original data array but only the ones with specific number of instances,
    #as established by the zipped dictionary
    trimmedData = data[np.isin(data [:,0] , EventsInstance)] 
    
    if crossFilter is True: 
        #introduces a second filter restricting the events to have to meet both requirements 
        #it has to be both in data and crossFilterData 
        #currently, it is set to count if any number of daughter particles are in the cross
        #filter data, can change the >= 1 to be other number.
        
        #returns unique tuples with occurances per event
        filterUniqueCounts = list( zip (*np.unique ( np.array(crossFilterData )[:,0], return_counts = True) ) ) 
        
        filterEventsInstance = [event for (event, instances) in filterUniqueCounts if instances >= 1]
        
        #returns event number that both have the particle in data 
        commonEvents = np.array(sorted ( list(set(EventsInstance).intersection(filterEventsInstance)) ) )
        
        #array containing events that pass both filters (with other info) 
        trimmedData = data[np.isin (data [:,0], commonEvents)] 
           
    if energySum is True:
       
        trimmedData = data[np.isin(data [:,0] , EventsInstance)] 
        
        if crossFilter is True: 
            trimmedData = data[np.isin (data [:,0], commonEvents)] 
        
        #sums all energy per event 
        #https://stackoverflow.com/questions/30041286/sum-rows-where-value-equal-in-column
        # sort  based on first column
        sortedData = trimmedData[ np.argsort(trimmedData[:,0]) ,: ]
        # row mask where each group ends
        row_mask = np.append(np.diff(sortedData[:,0], axis=0) != 0, [True] )
        # get cummulative summations and then DIFF to get summations for each group
        cumsum_grps = sortedData.cumsum(0) [row_mask,1:]
        sum_grps = np.diff(cumsum_grps, axis=0) 
        # concatenate the first unique row with its counts
        counts = np.concatenate((cumsum_grps[0,:][None],sum_grps),axis=0)
        # concatenate the first column of the input array for final output
        
        trimmedData = np.concatenate((sortedData [row_mask,0] [:,None], counts), axis = 1)

    if returnAll is None: 
        trimmedData = np.delete(trimmedData, np.s_[0:4], axis = 1) 
        return trimmedData
    else: 
        return trimmedData
        #uniqueEventsCount =  np.shape(trimmedData)[0] / occurNum
    
    