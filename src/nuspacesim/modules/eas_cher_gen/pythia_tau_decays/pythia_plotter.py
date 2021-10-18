import numpy as np
import matplotlib.pyplot as plt
#from nuSpaceSim.EAScherGen.Conex import conex_macros
#from nuSpaceSim.EAScherGen.Pythia import pythia_macros
import conex_macros
import pythia_macros
import sys

#can implement summing energy for each event in the future

def pythia_plotter (file_name, data_name, particle_id, table_energy_pev, num_occurance = 'all',
                    cross_filter = None, color = None ):

    decays = conex_macros.data_reader(file_name, data_name) 
    raw_particle_id = particle_id
    
    #extracts the pid in question, can be int or str: i.e. +/-211
    try: 
        if '/' in str(particle_id):
            particle_id = int(''.join(filter(str.isdigit, particle_id)))
    
            one_type = np.array([row for row in decays if row[2] == particle_id \
                                 or row[2] == -particle_id])
        else: 
            particle_id = int(particle_id)
            one_type = np.array([row for row in decays if row[2] == particle_id ])     
    except: 
        
        print('Please enter a valid particle ID') 
        sys.exit()
        
    if cross_filter is None:
        
        try:
            filtered_one_type = pythia_macros.event_filter(particle_data = one_type, 
                                                           occurance_number =  num_occurance)
            param = filtered_one_type * table_energy_pev
            title = f'Pythia Data PID: {raw_particle_id} | ' + \
                    str(file_name.split('/')[-1]) + '/' + str(data_name)
            conex_macros.parameter_histogram(param = param, 
                                             title = title, 
                                             x_label = 'Energy (PeV)',
                                             color = color)
        except: 
            print('No particles for this filter were found') 
            sys.exit()
                 
    else:
        
        #sets up the particle data for the filter, same as above but for the crossfilter pid
        try: 
            if '/' in str(cross_filter):
                cross_filter = int(''.join(filter(str.isdigit, cross_filter)))
        
                filter_type = np.array([row for row in decays if row[2] == cross_filter \
                                     or row[2] == -cross_filter])
            else: 
                cross_filter  = int(cross_filter )
                filter_type = np.array([row for row in decays if row[2] == cross_filter])
        except: 
            
            print('Please enter a valid cross filter particle ID') 
            sys.exit()        
        
        try:
            #now has the extra cross_filter_data field
            cross_filtered = pythia_macros.event_filter(particle_data = one_type, 
                                                        occurance_number =  num_occurance,
                                                        cross_filter_data = filter_type)
            param = cross_filtered * table_energy_pev
            title = f'Pythia Data PID: {raw_particle_id} | Cross-filter PID: {cross_filter} | ' + \
                    str(file_name.split('/')[-1]) + '/' + str(data_name)
            conex_macros.parameter_histogram(param = param, 
                                             title = title, 
                                             x_label = 'Energy (PeV)',
                                                 color = color)
        except: 
            print('No particles for this filter were found') 
            sys.exit()        
    
    # pions = np.array( [row for row in decays if row[2] == 211 or row[2] == -211 ])
    # kaons = np.array( [row for row in decays if row[2] == 321 or row[2] == -321 ])
    # electrons = np.array( [row for row in decays if row[2] == 11])
    # muons = np.array( [row for row in decays if row[2] == 14 or row[2] == -14 ])
    # gammas =  np.array( [row for row in decays if row[2]== 22])
    
    plt.show()
  
if __name__ == '__main__': 
    # pythia_plotter('DataLibraries/PythiaDecayTables/HDF5_data/new_tau_100_PeV.h5','tau_data', 
    #                 11, 100,1 )  
    pythia_plotter()

  