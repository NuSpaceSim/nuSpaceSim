import matplotlib.pyplot as plt
import numpy as np
from composite_ea_showers import CompositeShowers
from fitting_composite_eas import FitCompositeShowers

make_composites = CompositeShowers(shower_end=30000, grammage=10)
comp_showers, comp_depths =  make_composites(filter_errors=False)  
trimmed_showers, _ = make_composites.shower_end_cuts(
    composite_showers=comp_showers, composite_depths=comp_depths, separate_showers=False)
#make_composites = CompositeShowers(shower_end=2000, grammage=1)

# electron_gh, pion_gh, gamma_gh= make_composites.conex_params()
# electron_e, pion_e, gamma_e = make_composites.tau_daughter_energies()

# gamm_showers, gamm_depths = make_composites.single_particle_showers(
#     tau_energies=gamma_e, gh_params=gamma_gh
#     )
# elec_showers, elec_depths = make_composites.single_particle_showers(
#     tau_energies=electron_e, gh_params=electron_gh
#     )

# pion_showers, pion_depths = make_composites.single_particle_showers(
#     tau_energies=pion_e, gh_params=pion_gh
#     )

#%%
# get_fits = FitCompositeShowers(comp_showers, comp_depths,)
# fits = get_fits()
#
for depths,showers  in zip( comp_depths[0:10,], trimmed_showers[0:10,]):
    event_num = depths[0]
    decay_code = depths[1]
    plt.plot(depths[2:], showers[2:],'--', label = str(event_num)+"|"+ str(decay_code) )

plt.yscale('log')

plt.legend()
#%%

def mean_rms_plot(showers, bins, **kwargs):
    comp_showers = np.copy(showers[:,2:])
    bin_lengths = np.nansum(np.abs(bins[:, 2:]), axis = 1) 
    longest_shower_idx = np.argmax(bin_lengths)
    longest_shower = bins[longest_shower_idx, 2:]
    average_composites = np.nanmean(comp_showers, axis=0) 
    
    rms_error = np.sqrt(np.nanmean((average_composites  - comp_showers)**2, axis = 0 ))
    plt.plot(longest_shower, average_composites,  '--k', **kwargs) 
    
    plt.fill_between(longest_shower, 
                      average_composites - rms_error, 
                      average_composites + rms_error,
                      alpha = 0.5, 
                      edgecolor='red', 
                      facecolor='crimson',
                      interpolate=True,
                      label='RMS Error') 

    plt.title('Mean and RMS Error')
    plt.ylabel('Number of Particles')
    plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  
    # plt.yscale('log')
    plt.ylim(bottom=1) 
    plt.legend()
    
mean_rms_plot(trimmed_showers,comp_depths, label='mean generated composite showers')