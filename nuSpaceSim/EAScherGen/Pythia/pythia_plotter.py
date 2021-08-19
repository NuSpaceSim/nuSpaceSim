import numpy as np
import matplotlib.pyplot as plt
import h5py
#from Conex import gh_macros as plotting_macros 
from nuSpaceSim.EAScherGen.Conex import conex_macros
from nuSpaceSim.EAScherGen.Pythia import pythia_macros
import sys
#under works
def pythia_plotter (file_name, data_name, particle_id, table_energy_pev, num_occurance = 'all',
                    cross_filter = None, color = None ):
    #can add sum energy
    # decays = conex_macros.data_reader('DataLibraries/PythiaDecayTables/HDF5_data/new_tau_100_PeV.h5',
    #                                 'tau_data') 
    decays = conex_macros.data_reader(file_name, data_name) 
    raw_particle_id = particle_id
    try: 
        if '/' in particle_id:

            particle_id = int(''.join(filter(str.isdigit, particle_id)))

            one_type = np.array([row for row in decays if row[2] == particle_id \
                                 or row[2] == -particle_id])
        else: 
            particle_id = int(particle_id)
            one_type = np.array([row for row in decays if row[2] == particle_id ])
    except: 
        print('Please enter a valid particle ID') 
        sys.exit()
    
    filtered_one_type = pythia_macros.event_filter(particle_data = one_type, 
                                                   occurance_number =  num_occurance)
    
    param = filtered_one_type * table_energy_pev
    title = f'Pythia Data PID : {raw_particle_id}| ' + str(file_name.split('/')[-1]) +'/'+ str(data_name)
    conex_macros.parameter_histogram(param, title, x_label = 'Energy (PeV)')
    
    
    # pions = np.array( [row for row in decays if row[2] == 211 or row[2] == -211 ])
    # kaons = np.array( [row for row in decays if row[2] == 321 or row[2] == -321 ])
    # electrons = np.array( [row for row in decays if row[2] == 11])
    # muons = np.array( [row for row in decays if row[2] == 14 or row[2] == -14 ])
    # gammas =  np.array( [row for row in decays if row[2]== 22])
    
if __name__ == '__main__': 
    pythia_plotter('DataLibraries/PythiaDecayTables/HDF5_data/new_tau_100_PeV.h5','tau_data', 
                   '(+/-) 211', 100 )  

#%% Count events with one, two, three, four, five+,… gamma rays
gammaTrimmed = pyM.eventFilter (particleData = gammaData , occurNum = 1, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '1 Gamma Product Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)



gammaTrimmed = pyM.eventFilter (particleData = gammaData , occurNum = 2, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '2 Gamma Products Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)

gammaTrimmed = pyM.eventFilter (particleData = gammaData , occurNum = 2, energySum = True,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '2 Gamma Products Per Event Summed', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)



gammaTrimmed = pyM.eventFilter (particleData = gammaData , occurNum = 3, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '3 Gamma Products Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)

gammaTrimmed = pyM.eventFilter (particleDataa = gammaData , occurNum = 3, energySum = True,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '3 Gamma Products Per Event Summed', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)



gammaTrimmed = pyM.eventFilter (particleData = gammaData , occurNum = 4, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '4 Gamma Products Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)

gammaTrimmed = pyM.eventFilter (particleData = gammaData , occurNum = 4, energySum = True,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '4 Gamma Products Per Event Summed', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)




gammaTrimmed = pyM.eventFilter (particleData = gammaData , occurNum = 5, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '$\geq$ 5 Gamma Products Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)

gammaTrimmed = pyM.eventFilter (particleData = gammaData , occurNum = 5, energySum = True,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * gammaTrimmed, title = '$\geq$ 5 Gamma Products Per Event Summed', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'crimson', histtype = False)


    
#%%Count events with one, two, three, four, five+… charged pions.

pionTrimmed = pyM.eventFilter (particleData = pionData, occurNum = 1, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * pionTrimmed, title = '1 Pion Product Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)


pionTrimmed= pyM.eventFilter (particleData = pionData , occurNum = 2, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * pionTrimmed, title = '2 Pion Products Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)

pionTrimmed = pyM.eventFilter (particleData = pionData , occurNum = 2, energySum = True,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * pionTrimmed, title = '2 Pion Products Per Event Summed', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)


pionTrimmed= pyM.eventFilter (particleData = pionData , occurNum = 3, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * pionTrimmed, title = '3 Pion Products Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)

pionTrimmed = pyM.eventFilter (particleData = pionData , occurNum = 3, energySum = True,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * pionTrimmed, title = '3 Pion Products Per Event Summed', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)

'''
#pions do not contain event with 4 pion daughters 

pionTrimmed= pyM.eventFilter (particleData = pionData , occurNum = 4, energySum = False,
                                crossFilter = None, crossFilterData = None)  

plotting_macros.paramHist(param = 100 * pionTrimmed, title = 'Four Pion Products Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)

pionTrimmed = pyM.eventFilter (particleData = pionData , occurNum = 4, energySum = True,
                                crossFilter = None, crossFilterData = None)  

plotting_macros.paramHist(param = 100 * pionTrimmed, title = 'Four Pion Products Per Event Summed', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)
'''

pionTrimmed= pyM.eventFilter (particleData = pionData , occurNum = 5, energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * pionTrimmed, title = '$\geq$ 5 Pion Products Per Event', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)

pionTrimmed = pyM.eventFilter (particleData = pionData , occurNum = 5, energySum = True,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * pionTrimmed, title = '$\geq$ 5 Pion Products Per Event Summed', 
                          roundby = 3, xlabel = 'Energy (PeV)'  , color = 'royalblue', histtype = False)


#%%Count events with one, two, three, four, five+… charged pions w/at least one gamma-ray. 

pionTrimmedwGamma = pyM.eventFilter (particleData = pionData, occurNum = 2, energySum = False,
                                crossFilter = True, crossFilterData = gammaData)  
plotting_macros.paramHist(param = 100 * pionTrimmedwGamma, 
                          title = '1 Pion Product Per Event With at Least One Gamma', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Purple', 
                          histtype = False)

pionTrimmedwGamma = pyM.eventFilter (particleData = pionData , occurNum = 2, energySum = False,
                                crossFilter = True, crossFilterData = gammaData)  
plotting_macros.paramHist(param = 100 * pionTrimmedwGamma, 
                          title = '2 Pion Product Per Event With at Least One Gamma', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Purple', 
                          histtype = False)


pionTrimmedwGamma = pyM.eventFilter (particleData = pionData , occurNum = 2, energySum = True,
                                crossFilter = True, crossFilterData = gammaData)  
plotting_macros.paramHist(param = 100 * pionTrimmedwGamma, 
                          title = '2 Pion Product Per Event With at Least One Gamma Summed', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Purple', 
                          histtype = False)


pionTrimmedwGamma = pyM.eventFilter (particleData = pionData , occurNum = 3, energySum = False,
                                crossFilter = True, crossFilterData = gammaData)  
plotting_macros.paramHist(param = 100 * pionTrimmedwGamma, 
                          title = '3 Pion Product Per Event With at Least One Gamma', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Purple', 
                          histtype = False)


pionTrimmedwGamma = pyM.eventFilter (data = pionData , occurNum = 3, energySum = True,
                                crossFilter = True, crossFilterData = gammaData)  
plotting_macros.paramHist(param = 100 * pionTrimmedwGamma, 
                          title = '3 Pion Product Per Event With at Least One Gamma Summed', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Purple', 
                          histtype = False)


pionTrimmedwGamma = pyM.eventFilter (particleData = pionData , occurNum = 5, energySum = False,
                                crossFilter = True, crossFilterData = gammaData)  
plotting_macros.paramHist(param = 100 * pionTrimmedwGamma, 
                          title = '$\geq$ 5 Pion Product Per Event With at Least One Gamma', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Purple', 
                          histtype = False)


pionTrimmedwGamma = pyM.eventFilter (particleData = pionData , occurNum = 5, energySum = True,
                                crossFilter = True, crossFilterData = gammaData)  
plotting_macros.paramHist(param = 100 * pionTrimmedwGamma, 
                          title = '$\geq$ 5 Pion Product Per Event With at Least One Gamma Summed', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Purple', 
                          histtype = False)


#%%Count events with a kaon  in the final state.

kaonTrimmed = pyM.eventFilter (particleData = kaonData, occurNum ='all', energySum = False,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * kaonTrimmed, 
                          title = '1 Kaon Daughter Particles', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Orange', 
                          histtype = False)


kaonTrimmed = pyM.eventFilter (particleData = kaonData, occurNum = 'all', energySum = True,
                                crossFilter = None, crossFilterData = None)  
plotting_macros.paramHist(param = 100 * kaonTrimmed, 
                          title = 'Kaon Daughter Particles Summed Per Event', 
                          roundby = 3, 
                          xlabel = 'Energy (PeV)' , 
                          color = 'Orange', 
                          histtype = False)
#%%
plotting_macros.paramHist(param = pionData, title = 'Pion Energies ', roundby = 3, 
                          xlabel = 'Energy (100 PeV)'  , color = 'royalblue', histtype = True)  

plotting_macros.paramHist(param = kaonData, title = 'Kaon Energies ', roundby = 3, 
                          xlabel = 'Energy (100 PeV)'  , color = 'orange', histtype = True)

plotting_macros.paramHist(param = electronData, title = 'Electron Energies ', roundby = 3, 
                          xlabel = 'Energy (100 PeV)'  , color = 'green', histtype = True)

plotting_macros.paramHist(param = muonData, title = 'Muon Energies ', roundby = 3, 
                          xlabel = 'Energy (100 PeV)'  , color = 'purple', histtype = True)

plotting_macros.paramHist(param = gammaData, title = 'Gamma Energies ', roundby = 3, 
                          xlabel = 'Energy (100 PeV)'  , color = 'crimson', histtype = True)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    