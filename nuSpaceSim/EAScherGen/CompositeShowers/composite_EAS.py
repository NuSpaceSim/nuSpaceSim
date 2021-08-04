import numpy as np
import matplotlib.pyplot as plt
import h5py
from TauDecays import pythia_macros as pyM 
from Conex import gh_macros as ghM
from CompositeShowers import composite_macros as coM 

#read in the conex files
conexFile = h5py.File('Conex/HDF5_data/pion_EAStable.h5','r')
showers = conexFile.get('EASdata_211')
pionEAS = np.array(showers)

conexFile = h5py.File('Conex/HDF5_data/electron_EAStable.h5','r')
showers = conexFile.get('EASdata_11')
electronEAS = np.array(showers)

conexFile = h5py.File('Conex/HDF5_data/gamma_EAStable.h5', 'r')
showers = conexFile.get('EASdata_22')
gammaEAS = np.array(showers)
       
#read in the pythia files
pythiaFile = h5py.File('TauDecays/HDF5_data/new_tau_100_PeV.h5', 'r')
pythiaDecays = pythiaFile.get('tau_data')
pythiaDecays = np.array(pythiaDecays)

def readInPythia ():
    '''
    > unpacks the pythia decay tables and isolates specific particle types \n 
    > each data array contains event #, event decay codes, y values, and any other pertinent info. 
    '''
    pionData = []
    kaonData = [] 
    gammaData = [] 
    electronData = []
    for row in range (0, np.shape(pythiaDecays)[0]):
        
        a_pion =  pyM.pythiaReader(pythiaDecays, row, input_PID = 211) 
        a_kaon =  pyM.pythiaReader(pythiaDecays, row,  input_PID = 321) 
        a_gamma = pyM.pythiaReader(pythiaDecays, row, input_PID = 22)
        a_electron =  pyM.pythiaReader(pythiaDecays, row, input_PID = 11)
        
        if a_pion is not None:
            pionData.append(a_pion)
            
        elif a_kaon is not None:
            kaonData.append(a_kaon)
             
        elif a_gamma is not None: 
            gammaData.append(a_gamma)
            
        elif a_electron is not None: 
            electronData.append(a_electron)
    
    return np.array(pionData), np.array(kaonData), np.array(gammaData), np.array(electronData)

#flattened pythia tables
pionPythiaData, kaonPythiaData, gammaPythiaData , electronPythiaData = readInPythia() 

#kaon and pion treated the same-- both can scale pionEAS; stacked for easier parsing
KaonPionPythiaData = np.vstack((pionPythiaData, kaonPythiaData))
KaonPionPythiaData = KaonPionPythiaData[ np.argsort(KaonPionPythiaData[:,0]) ,: ]

#get the particle contents and bins
pionXs, pionFs, _ = coM.compositeProfile (rowStart = 0, rowEnd = np.shape(pionEAS)[0] - 1,
                                            xlimit = 2000, 
                                            conexShowers = pionEAS, 
                                            pythiaTables = KaonPionPythiaData, 
                                            regPlot = 0)
gammaXs, gammaFs, _ = coM.compositeProfile (rowStart = 0, rowEnd = np.shape(gammaEAS)[0] - 1,
                                            xlimit = 2000, 
                                            conexShowers = gammaEAS, 
                                            pythiaTables = gammaPythiaData, 
                                            regPlot = 0)
electXs, electFs, _ = coM.compositeProfile (rowStart = 0, rowEnd = np.shape(electronEAS)[0] - 1,
                                            xlimit = 2000, 
                                            conexShowers = electronEAS, 
                                            pythiaTables = electronPythiaData, 
                                            regPlot = 0)

pionFs = np.array(pionFs)
gammaFs = np.array(gammaFs)
electFs = np.array(electFs)

pionXs = np.array(pionXs)
gammaXs = np.array(gammaXs)
electXs = np.array(electXs)

#combine all the particle data per event to make composites 
binsPerEventNoTag, summedContentNoTag, eventLabels = coM.contentPerEvent (pythiaDecays = pythiaDecays,
                                                      particleFs = (pionFs, gammaFs, electFs),
                                                      particleXs = (pionXs, gammaXs, electXs),
                                                      noEventTags = True, avgBins = True)
#change pythiaEventCodes to something else
binsPerEvent, summedParticleContent, eventCodes = coM.contentPerEvent (pythiaDecays = pythiaDecays,
                                                   particleFs = (pionFs, gammaFs, electFs),
                                                   particleXs = (pionXs, gammaXs, electXs),
                                                   avgBins = True) 


#%% multiple event plot 
coM.compositePlotter(rowStart = 0, rowEnd = 3, 
                 eventLabels = eventLabels, 
                 eventBins= binsPerEventNoTag, 
                 particleContent = summedContentNoTag,
                 compositePlot = True)

plt.title('Composite Showers from Final State Particles \n' + 
          '$\pm$ 211, $\pm$ 321, 22, 11')
# x-axis
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0)
# y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ylim(1e-4, 1e8)
plt.ylabel('Number of Particles')
#plt.yscale('log')

plt.legend()
plt.show()

#avg and rms plotter
coM.compositePlotter(rowStart = 0, rowEnd = np.shape(summedContentNoTag)[0], 
                 eventLabels = eventLabels, 
                 eventBins= binsPerEventNoTag, 
                 particleContent = summedContentNoTag,
                 avgPlot = True)

plt.title('Average and RMS Profile for All Uniquely Sampled Composites \n' + 
          '$\pm$ 211, $\pm$ 321, 22, 11')
# x-axis
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0)
# y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ylim(1e-4, 1e8)
plt.ylabel('Number of Particles')
#plt.yscale('log')

plt.legend()
plt.show()

#%% Nmax and Xmax of all Composite Showers
compositeNmax, compositeXmax = coM.NmaxANDXmax(summedContentNoTag, binsPerEventNoTag) 
ghM.paramHist(param = compositeNmax, 
              title = 'Greisen Nmax Distribution for Composite Showers', 
              roundby = 0, 
              xlabel = 'Nmax Values (Particle Content)' , 
              color = 'deepskyblue', 
              histtype = True)
ghM.paramHist(param = compositeXmax, 
              title = 'Greisen Xmax Distribution for Composite Showers', 
              roundby = 0, 
              xlabel = 'Slant Depth t ' + '($g \; cm^{-2}$)', 
              color = 'palegreen', 
              histtype = True) 

#%%  Look at the electron decay channel only, calc mean and rms. 

singleElectronChannels = electronPythiaData[electronPythiaData[:,1] == 300001]

#invert these to just get composites with electrons (and pions, kaons, and gammas)
singleElectronXs = binsPerEvent[np.isin(summedParticleContent [:,0], 
                                        singleElectronChannels[:,0], invert = False)] 
singleElectronFs = summedParticleContent[np.isin(summedParticleContent [:,0], 
                                                 singleElectronChannels[:,0], invert = False)]


singleElectronCodes = eventLabels[np.isin(eventLabels [:,0], singleElectronFs[:,0])]
singleElectronXs = np.delete(singleElectronXs, 0, 1)
singleElectronFs =  np.delete(singleElectronFs, 0, 1)

#############################################################################################
electronNmax, electronXmax = coM.NmaxANDXmax(singleElectronFs, singleElectronXs)
ghM.paramHist(param = electronNmax, 
              title = 'Greisen Nmax Distribution Without Single Electron Decay Channels (300001)', 
              roundby = 0, 
              xlabel = 'Nmax Values (Particle Content)' , 
              color = 'turquoise', 
              histtype = True)
ghM.paramHist(param = electronXmax, 
              title = 'Greisen Xmax Distribution Without Single Electron Decay Channels (300001)', 
              roundby = 0, 
              xlabel = 'Slant Depth t ' + '($g \; cm^{-2}$)', 
              color = 'crimson', 
              histtype = True) 
#############################################################################################

coM.compositePlotter(rowStart = 0, rowEnd = np.shape(singleElectronFs)[0], 
                 eventLabels = singleElectronCodes, 
                 eventBins= singleElectronXs, 
                 particleContent = singleElectronFs,
                 avgPlot = True)
#plt.title('Average and RMS Composite Profile for \n  Single Electron Decay Channel (300001) ')
plt.title('Average and RMS Composite Profile \n  Without Single Electron Decay Channel (300001) ')
# x-axis
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0)
# y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ylim(1e-4, 1e8)
plt.ylabel('Number of Particles')
#plt.yscale('log') 
plt.legend()
plt.show()
#%% Look at single pion decay channel only, calc mean and rms

singlePionChannels = pyM.eventFilter (particleData = pionPythiaData, occurNum = 1, 
                                      energySum = None, crossFilter = None, crossFilterData = None,
                                      returnAll = True ) 
singlePionwGamma = pyM.eventFilter (particleData = pionPythiaData, occurNum = 1, 
                                      energySum = None, crossFilter = True, crossFilterData = gammaPythiaData,
                                      returnAll = True )  
singleKaonChannels = pyM.eventFilter (particleData = kaonPythiaData, occurNum = 1, 
                                      energySum = None, crossFilter = None, crossFilterData = None,
                                      returnAll = True )
singleKaonInPion = pyM.eventFilter (particleData = kaonPythiaData, occurNum = 1, 
                                      energySum = None, crossFilter = True, crossFilterData = pionPythiaData,
                                      returnAll = True )

exclude = np.vstack((singleKaonInPion, singlePionwGamma ))
exclude = exclude [ np.argsort(exclude[:,0]) ,: ]
#include single channel pions and kaons 
singlePionandKaonChannels = np.vstack((singlePionChannels, singleKaonChannels)) 
singlePionandKaonChannels =  singlePionandKaonChannels[ np.argsort(singlePionandKaonChannels[:,0]) ,: ]

#invert for pure single pion and kaon 
singlePionandKaonChannels = singlePionandKaonChannels[np.isin(singlePionandKaonChannels [:,0], 
                                                              exclude[:,0], invert = False)]

singlePionChannels = singlePionandKaonChannels #redefine for below

singlePionXs = binsPerEvent[np.isin(summedParticleContent [:,0], singlePionChannels[:,0])] 
singlePionFs = summedParticleContent[np.isin(summedParticleContent [:,0], singlePionChannels[:,0])]

singlePionXs= np.delete(singlePionXs, 0, 1)
singlePionFs =  np.delete(singlePionFs, 0, 1)
singlePionLabels = eventLabels[np.isin(eventLabels [:,0], singlePionFs [:,0])]

#############################################################################################
pionNmax, pionXmax = coM.NmaxANDXmax(singlePionFs, singlePionXs)
ghM.paramHist(param = pionNmax, 
              title = 'Greisen Nmax Distribution of Single Pion and Kaon in Composites', 
              roundby = 0, 
              xlabel = 'Nmax Values (Particle Content)' , 
              color = 'turquoise', 
              histtype = True)
ghM.paramHist(param = pionXmax, 
              title = 'Greisen Xmax Distribution of Single Pion and Kaon in Composites', 
              roundby = 0, 
              xlabel = 'Slant Depth t ' + '($g \; cm^{-2}$)', 
              color = 'crimson', 
              histtype = True) 
#############################################################################################

coM.compositePlotter(rowStart = 0, rowEnd = np.shape(singlePionFs)[0], 
                 eventLabels = singlePionLabels, 
                 eventBins= singlePionXs, 
                 particleContent = singlePionFs ,
                 avgPlot = True)

plt.title('Average and RMS Composite Profile for \n Single Pion Decay Channel')
#plt.title('Average and RMS Composite Profile for \n Single Pion and Kaon in Composites ')
# x-axis
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0)
# y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ylim(1e-4, 1e8)
plt.ylabel('Number of Particles')
#plt.yscale('log')

plt.legend()
plt.show()

#%%Look at multiple charged pion only composite, calc mean and rms.

multiplePionChannels = pyM.eventFilter (particleData = pionPythiaData, occurNum = 'multiple', 
                                      energySum = None, crossFilter = None, crossFilterData = None,
                                      returnAll = True ) 

multipleKaonChannels = pyM.eventFilter (particleData = kaonPythiaData, occurNum = 'multiple', 
                                      energySum = None, crossFilter = None, crossFilterData = None,
                                      returnAll = True )

anyKaonInPion = pyM.eventFilter (particleData = kaonPythiaData, occurNum = 'any', 
                                      energySum = None, crossFilter = True, crossFilterData = pionPythiaData,
                                      returnAll = True )

multiplePionChannels = np.concatenate((multiplePionChannels, multipleKaonChannels, anyKaonInPion), axis = 0)
multiplePionChannels =  multiplePionChannels[ np.argsort(multiplePionChannels[:,0]) ,: ] 


multiplePionXs = binsPerEvent[np.isin(summedParticleContent [:,0], multiplePionChannels[:,0])] 
multiplePionFs = summedParticleContent[np.isin(summedParticleContent [:,0], multiplePionChannels[:,0])]

multiplePionXs = np.delete(multiplePionXs, 0, 1)
multiplePionFs =  np.delete(multiplePionFs, 0, 1)

multiplePionLabels = eventLabels[np.isin(eventLabels [:,0], multiplePionFs [:,0])]



#############################################################################################
multiplePionNmax, multiplePionXmax = coM.NmaxANDXmax(multiplePionFs, multiplePionXs)
ghM.paramHist(param = multiplePionNmax, 
              title = 'Greisen Nmax Distribution of Multiple Pion and Kaon Composites', 
              roundby = 0, 
              xlabel = 'Nmax Values (Particle Content)' , 
              color = 'turquoise', 
              histtype = True)
ghM.paramHist(param = multiplePionXmax, 
              title = 'Greisen Xmax Distribution of Multiple Pion and Kaon Composites', 
              roundby = 0, 
              xlabel = 'Slant Depth t ' + '($g \; cm^{-2}$)', 
              color = 'crimson', 
              histtype = True) 
#############################################################################################


coM.compositePlotter(rowStart = 0, rowEnd = np.shape(multiplePionFs)[0], 
                 eventLabels = multiplePionLabels, 
                 eventBins= multiplePionXs, 
                 particleContent = multiplePionFs ,
                 avgPlot = True)

plt.title('Average and RMS Composite Profile for \n Multiple Pion Composite')
#plt.title('Average and RMS Composite Profile for \n Multiple Pion Composite Kaon Included')

# x-axis
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0)
# y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ylim(1e-4, 1e8)
plt.ylabel('Number of Particles')
#plt.yscale('log')

plt.legend()
plt.show()  


#%%Deprecated composite profile routine
'''
Deprecated composite profile routine; just sums certain particles per event. 
Can come in handy when doing specific particle types
Not true compsite showers
'''
#Pion

coM.compositeProfile (rowStart = 0, rowEnd = 1, xlimit = 5000, 
                       conexShowers = pionEAS, 
                       pythiaTables = pionPythiaData, 
                       nmaxThreshold = None, thresholdPlot = None, 
                       eventSum = None,
                       avgPlot = None,
                       regPlot = 1,
                       greiComp = True)

plt.title('Old "Composite" Profile' )
#plt.title('Pion Comparison of Greissen and G-H Parameterization')
# x-axis
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0)

# y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ylim(1e-4, 1e8)

plt.ylabel('Number of Particles')
plt.yscale('log')


plt.legend()
plt.show()

#Electron
coM.compositeProfile ( rowStart = 0, rowEnd = 0, xlimit = 50000, 
                        conexShowers = electronEAS,
                        pythiaTables = electronPythiaData, 
                        nmaxThreshold = None, thresholdPlot = None,
                        eventSum = None, 
                        avgPlot = None,
                        regPlot = True,
                        greiComp = True)

plt.title('Scaled Electron Profiles ($ f \leq 0.01 \; Nmax$) from G-H Parameters')
#plt.title('Electron Comparison of Greissen and G-H Parameterization') 

# x-axis
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0)

# y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ylim(1e-4, 1e8)

plt.ylabel('Number of Particles')
plt.yscale('log')


plt.legend()
plt.show()

#Gamma
coM.compositeProfile ( rowStart = 0, rowEnd = 0, xlimit = 2000, 
                        conexShowers = gammaEAS,
                        pythiaTables = gammaPythiaData, 
                        nmaxThreshold = None, thresholdPlot = None,
                        eventSum = None, 
                        avgPlot = None,
                        regPlot = True,
                        greiComp = True)

plt.title('Scaled Gamma Profiles ($ f \leq 0.01 \; Nmax$) from G-H Parameters')
plt.title('Gamma Comparison of Greissen and G-H Parameterization') 
# x-axis
plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
plt.xlim(left = 0)

# y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ylim(1e-4, 1e8)

plt.ylabel('Number of Particles')
plt.yscale('log')


plt.legend()
plt.show()