import numpy as np
import matplotlib.pyplot as plt
import h5py
from Conex import gh_macros as plotting_macros 
from TauDecays import pythia_macros as pyM


#.txt to .h5 

#pyM.decayTabletoHDF5('TauDecays/tau_100_PeV_07192021_machine.txt', 'new_tau_100_PeV.h5', 'tau_data')   


#.h5 to array
data = h5py.File('TauDecays/HDF5_data/new_tau_100_PeV.h5', 
                 'r')
decays = data.get('tau_data')
decays = np.array(decays)

pionData = []
kaonData = [] 
electronData = [] 
muonData= []
gammaData = [] 

#read in all data for each particle type 
for row in range (0, np.shape(decays)[0]):
    
    a_pion =  pyM.pythiaReader(decays, row, input_PID = 211) 
    a_kaon =  pyM.pythiaReader(decays, row,  input_PID = 321)
    a_electron = pyM.pythiaReader(decays, row, input_PID = 11)
    a_muon = pyM.pythiaReader(decays, row, input_PID = 14)
    a_gamma = pyM.pythiaReader(decays, row, input_PID = 22)
    
    if a_pion is not None:
        pionData.append(a_pion)
        
    elif a_kaon is not None:
        kaonData.append(a_kaon)
        
    elif a_electron is not None : 
        electronData.append(a_electron) 
        
    elif a_muon is not None: 
        muonData.append(a_muon)
        
    elif a_gamma is not None: 
        gammaData.append(a_gamma) 

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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    