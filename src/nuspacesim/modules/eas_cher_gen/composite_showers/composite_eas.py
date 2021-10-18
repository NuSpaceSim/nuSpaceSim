import numpy as np
import matplotlib.pyplot as plt
from nuSpaceSim.EAScherGen.CompositeShowers import composite_macros
# from nuSpaceSim.EAScherGen.Conex import conex_macros
# from nuSpaceSim.EAScherGen.Pythia import pythia_macros
# import sys

def composite_eas (output_data = None, sample_plt = None):
    
    electron_gh, gamma_gh, pion_gh, tau_decays = composite_macros.load_sample_hdf5_data()
    
    pion_pyth_decay = np.array( [row for row in tau_decays if row[2] == 211 or row[2] == -211 ])
    kaon_pyth_decay = np.array( [row for row in tau_decays if row[2] == 321 or row[2] == -321 ])
    electron_pyth_decay = np.array( [row for row in tau_decays if row[2] == 11])
    gamma_pyth_decay =  np.array( [row for row in tau_decays if row[2]== 22])
    #muon_pyth_decay = np.array( [row for row in tau_decays if row[2] == 14 or row[2] == -14 ])
    #kaon and pion treated the same-- both can scale pionEAS, stacked for easier parsing
    pion_pyth_decay = np.vstack((pion_pyth_decay, kaon_pyth_decay))
    pion_pyth_decay = pion_pyth_decay[ np.argsort(pion_pyth_decay[:,0]) ,: ]
    
    pion_depth, pion_content, _ = composite_macros.composite_eas(conex_showers = pion_gh, 
                                                        pythia_tables = pion_pyth_decay, 
                                                        end_row = np.shape(pion_gh)[0] - 1,
                                                        regular_plt = 0)                                               
    
    gamma_depth, gamma_content, _ = composite_macros.composite_eas(conex_showers = gamma_gh, 
                                                        pythia_tables = gamma_pyth_decay, 
                                                        end_row = np.shape(gamma_gh)[0] - 1,
                                                        regular_plt = 0)  
    
    electron_depth, electron_content, _ = composite_macros.composite_eas(conex_showers = electron_gh, 
                                                        pythia_tables = electron_pyth_decay, 
                                                        end_row = np.shape(electron_gh)[0] - 1,
                                                        regular_plt = 0)  
                                                
    composite_bins, composite_showers, event_labels = \
        composite_macros.content_per_event (pythia_decays = tau_decays,
                                            particle_contents = (pion_content, 
                                                                 gamma_content, 
                                                                 electron_content),
                                            slant_depths = (pion_depth, 
                                                            gamma_depth,
                                                            electron_depth),
                                            average_bins = True)
    
    
    if output_data is not None: 
        composite_macros.composite_showers_to_h5(file_name_out = output_data, 
                                                 composite_showers = composite_showers, 
                                                 composite_depths = composite_bins)
    
    if sample_plt is not None:
        composite_macros.composite_plotter(start_row = 10, end_row = 20, 
                                           event_labels = event_labels, 
                                           event_bins = composite_bins, 
                                           particle_content = composite_showers, 
                                           composite_plt = True)
        plt.title('Composite Showers from Final State Particles \n' + 
                  '$\pm$ 211, $\pm$ 321, 22, 11')
        # x-axis
        plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
        plt.xlim(left = 0)
        # y-axis
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #plt.ylim(1e0, 1e8)
        plt.ylabel('Number of Particles')
        #plt.yscale('log')
        
        plt.legend()
        plt.show() 

if __name__ == '__main__': 
    composite_eas (sample_plt=True)