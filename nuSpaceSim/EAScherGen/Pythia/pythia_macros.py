import numpy as np
import h5py


def pythia_converter (input_file, output_file = 'converted.h5', output_dataset = 'pythia_tables') : 
    '''
    > takes a pythia machine-readable text/ ascii file and converts it to hdf5 \n 
    > the h5 table have these mappings \n 
    [:0] = Event #  \n
    [:1] = Decay Code  \n
    [:2] = PID of daughter particle \n
    [:3] = mother particle \n
    [:4] = fractional energy  y (e.g., 100PeV * y) of the daughter particle  \n
    '''
    data = [] 
    with open(input_file) as f: #does this line by line, with output conventions used as classifier
        for line in f:
            if '#'  in line:
                event_number =  int (line [line.find('#')+ 1:])
            elif 'Event' not in line and '     0    0 0' not in line and len(line) == 9: 
                decay_code = int (line)
            elif '     0    0 0' not in line and 'Event' not in line and len(line) != 9:
                pid, mother, energy = [float(n) for n in line.split()]  
                data.append([event_number, decay_code, pid, mother, energy]) 
                
    data_array = np.array(data)
    newData = h5py.File(output_file, 'w')
    newData.create_dataset(output_dataset, data = data_array)
    newData.close()
    print('File written to current directory...')
 
    
 
def event_filter (particle_data, occurance_number, 
                  sum_energy = None, cross_filter_data = None, return_all = None):
    '''
    > filters events that have certain number of occurrences of a particle type per event \n
    > can impose cross_filter_data that returns only events that have two events in common \n
    > returns a vector containing filtered energy entries  or return the entire table \n
    particle_data = array to filter \n
    occurance_number = integers, can be 'all' or 'multiple' (>1)
    cross_filter_data = the particle needs to exist here too in a given event
    return_all = true if ruturning not just energies
    '''
    data = np.array(particle_data)

    #unique tuples with number of instances for each event number 
    #event numbers have duplicates in original data array 
    unique_counts = list( zip (*np.unique (data [:,0], return_counts = True) ) ) 
    
    #return event numbers with this many instances using various filters
    try:
        if 0 < occurance_number < 5:
            event_number_match = [event for (event, instances) in unique_counts \
                                   if instances == occurance_number]
                
        elif occurance_number >= 5:
            event_number_match = [event for (event, instances) in unique_counts \
                                    if instances >= 5]
            
    except:
        if occurance_number.lower() == 'all': 
            event_number_match = [event for (event, instances) in unique_counts \
                                    if instances >= 1]
            
        elif occurance_number.lower() == 'multiple': 
            event_number_match = [event for (event, instances) in unique_counts \
                                  if instances > 1 ]
                
        else: 
            print('Enter a valid occurance number.')

    #return the original data array but only the ones with specific number of instances,
    #as established by the zipped dictionary
    trimmed_data = data[np.isin(data [:,0] , event_number_match )] 
    
    if cross_filter_data is not None: 
        #introduces a second filter restricting the events to those in both particle datas
        #currently, it is set to pass any number of occurances in cross_filter_data >= 1
        filter_unique_counts = list( zip (*np.unique ( np.array(cross_filter_data)[:,0], \
                                                      return_counts = True) ) ) 
        filter_event_number_match = [event for (event, instances) in filter_unique_counts \
                                     if instances >= 1]
        #returns event number that both have the particle in each event 
        common_events = np.array(sorted ( list(set(event_number_match ). \
                                              intersection(filter_event_number_match)) ) )
        #array containing events that pass both filters
        trimmed_data = data[np.isin (data [:,0], common_events)] 
           
    if sum_energy is not None:
        #sums all energy per event 
        trimmed_data  = data[np.isin(data [:,0] , event_number_match )] 
        
        if cross_filter_data is not None: 
            trimmed_data  = data[np.isin (data [:,0], common_events)] 
        
        #based on: https://stackoverflow.com/a/30041823

        sortedData = trimmed_data [ np.argsort(trimmed_data [:,0]) ,: ]
        row_mask = np.append(np.diff(sortedData[:,0], axis=0) != 0, [True] )
        cumsum_grps = sortedData.cumsum(0) [row_mask,1:]
        sum_grps = np.diff(cumsum_grps, axis=0) 
        counts = np.concatenate((cumsum_grps[0,:][None],sum_grps),axis=0)
        trimmed_data  = np.concatenate((sortedData [row_mask,0] [:,None], counts), axis = 1)

    if return_all is None: 
        trimmed_data  = np.delete(trimmed_data , np.s_[0:4], axis = 1) 
        return trimmed_data 
    else: 
        return trimmed_data 

    
    