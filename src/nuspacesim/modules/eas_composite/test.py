import matplotlib.pyplot as plt
import numpy as np
from composite_ea_showers import CompositeShowers
from fitting_composite_eas import FitCompositeShowers

make_composites_0km =  CompositeShowers( 
    alt=0, shower_end=5000, grammage=1
    ) 

comp_showers_0km, comp_depths_0km =  make_composites_0km(filter_errors=False) 
 
trimmed_showers_0km, test_depths = make_composites_0km.shower_end_cuts(
    composite_showers=comp_showers_0km, 
    composite_depths=comp_depths_0km, 
    separate_showers=False
    )
mask = (trimmed_showers_0km > 0) & (trimmed_showers_0km < 1)
trimmed_showers_0km[mask] = 0

#trimmed_showers_0km[trimmed_showers_0km == 0] = np.nan
# make_composites_15km = CompositeShowers( 
#     alt=15, shower_end=20000, grammage=10
#     )

# comp_showers_15km, comp_depths_15km =  make_composites_15km(filter_errors=False) 
 
# trimmed_showers_15km, _ = make_composites_15km.shower_end_cuts(
#     composite_showers=comp_showers_15km, 
#     composite_depths=comp_showers_15km, 
#     separate_showers=False
#     )

#%% RMS histogram analysis 
import scipy

sample_shower_column = trimmed_showers_0km[:,50::700].T
sample_depth_column = comp_depths_0km[:,50::700].T

for (depth,showers) in zip(sample_depth_column,sample_shower_column ): 
    #print(depth)
    print(showers.shape) 
    plt.figure(figsize=(8,6)) 

    plt.hist(showers, 
             alpha = .5, 
             edgecolor='black', linewidth=.5,
             label='{:g} g/cm^2'.format(depth[418]), 
             bins = 10)
    plt.title('Distribution of Composite values')
    plt.xlabel('Particle Content (N)')
    plt.ylabel('# of composite showers')
    plt.xscale('log')
    plt.legend()

#%%
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
plt.figure(figsize=(8,6), dpi=(200)) 
# for depths,showers  in zip(comp_depths_0km[0:5,], trimmed_showers_0km[0:5,]):
#     event_num = depths[0]
#     decay_code = depths[1]
#     plt.plot(
#             depths[2:], 
#             showers[2:], 
#             label = str(event_num)+"|"+ str(decay_code) 
#             )
    

for depths,showers  in zip(comp_depths_15km, trimmed_showers_15km):
    event_num = depths[0]
    decay_code = depths[1]
    plt.plot(
             depths[2:], 
             showers[2:],
             '--', 
             label = str(event_num)+"|"+ str(decay_code) 
             )

plt.yscale('log')
#plt.xlim(left= -30, right=100)
plt.ylabel("N Particles")
plt.xlabel("Shower Stage")
#plt.legend()
#%%

def mean_rms_plot(showers, bins, **kwargs): 
    comp_showers = np.copy(showers[:,2:])
    bin_lengths = np.nansum(np.abs(bins[:, 2:]), axis = 1) 
    longest_shower_idx = np.argmax(bin_lengths)
    longest_shower = bins[longest_shower_idx, 2:]
    # take average a long each bin, ignoring nans
    average_composites = np.nanmean(comp_showers, axis=0) 
    
    test = average_composites  - comp_showers
    # take the square root of the mean of the difference between the average
    # and each particle content of each shower for one bin, squared
    rms_error = np.sqrt(
        np.nanmean((average_composites  - comp_showers)**2, axis = 0 )
        )
    rms = np.sqrt(
        np.nanmean((comp_showers)**2, axis = 0 ) 
        )
    std = np.nanstd(comp_showers, axis = 0)  
    err_in_mean = np.nanstd(comp_showers, axis = 0) / np.sqrt(np.sum(~np.isnan( comp_showers), 0)) 
    #plt.figure(figsize=(8,6))  
    plt.plot(longest_shower, average_composites,  '--k', label = 'mean') 
    plt.plot(longest_shower, rms_error ,  '--r', label='rms error') 
    plt.plot(longest_shower, rms ,  '--g', label='rms') 
    plt.plot(longest_shower, std ,  '--y', label='std')  
    plt.plot(longest_shower, err_in_mean ,  '--b', label='error in mean')
    
    rms_low = average_composites - rms_error 
    rms_high = average_composites + rms_error
    

    plt.fill_between(longest_shower, 
                      rms_low, 
                      rms_high,
                      alpha = 0.5, 
                      #facecolor='crimson',
                      interpolate=True,
                      **kwargs) 

    plt.title('Mean and RMS Error 0km')
    plt.ylabel('Number of Particles')
    plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  

    plt.yscale('log') 
    plt.grid(True, which='both', linestyle='--')
    #plt.ylim(bottom=1) 
    #plt.xlim(right=1500)
    plt.legend()
    plt.show()
    
    return rms_low, rms_high, test

plt.figure(figsize=(8,6))    
low, high,test = mean_rms_plot(
    trimmed_showers_0km, comp_depths_0km, label='mean +/- rmse ')
# mean_rms_plot(trimmed_showers_15km,
#               comp_depths_15km, label='15 km', facecolor='green')
#%%  
decay_channels = np.unique(comp_depths_0km[:,1]) [0:2]

for dc in decay_channels: 
    x = trimmed_showers_0km[trimmed_showers_0km[:,1] == dc]
    y = comp_depths_0km[comp_depths_0km[:,1] == dc]
    
    
    plt.figure(figsize=(8,6))   
    for depths,showers  in zip(y, x):

        event_num = depths[0]
        decay_code = depths[1]
        plt.plot(
                 depths[2:], 
                 showers[2:],
                 alpha=0.3, 
                 #label = str(event_num)+"|"+ str(decay_code) 
                 )
    plt.title('{}'.format(dc))
    #plt.legend()
    
    plt.ylabel("N Particles")
    plt.xlabel("Shower Stage")   
    plt.yscale('log') 