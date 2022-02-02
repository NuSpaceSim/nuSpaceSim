import matplotlib.pyplot as plt
import numpy as np
from composite_ea_showers import CompositeShowers
from fitting_composite_eas import FitCompositeShowers

def mean_rms_plot(showers, bins, **kwargs): 
    comp_showers = np.copy(showers[:,2:])
    bin_lengths = np.nansum(np.abs(bins[:, 2:]), axis = 1) 
    longest_shower_idx = np.argmax(bin_lengths)
    longest_shower_bin = bins[10, 2:]
    # take average a long each bin, ignoring nans
    average_composites = np.nanmean(comp_showers, axis=0) 
    
    #test = average_composites  - comp_showers
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
    plt.plot(longest_shower_bin, average_composites,  '--k', label = 'mean') 
    # plt.plot(longest_shower, rms_error ,  '--r', label='rms error') 
    # plt.plot(longest_shower, rms ,  '--g', label='rms') 
    # plt.plot(longest_shower, std ,  '--y', label='std')  
    # plt.plot(longest_shower, err_in_mean ,  '--b', label='error in mean')
    
    rms_low = average_composites - rms_error
    rms_high = average_composites + rms_error
    

    plt.fill_between(longest_shower_bin, 
                      rms_low, 
                      rms_high,
                      alpha = 0.2, 
                      #facecolor='crimson',
                      interpolate=True,
                      **kwargs) 

    plt.title('Mean and RMS Error') 
    plt.ylabel('Number of Particles')
    #plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  
    plt.xlabel('Shower stage')
    plt.yscale('log') 
    plt.grid(True, which='both', linestyle='--')
    plt.ylim(bottom=1) 
    #plt.xlim(right=1500)
    plt.legend()    
    #plt.show()
    
    return  longest_shower_bin, average_composites, rms_low, rms_high




make_composites_15km = CompositeShowers( 
    alt=15, shower_end=20000, grammage=10, tau_table_start=3000
    )
comp_showers_15km, comp_depths_15km =  make_composites_15km(filter_errors=False) 
trimmed_showers_15km, _ = make_composites_15km.shower_end_cuts(
    composite_showers=comp_showers_15km, 
    composite_depths=comp_depths_15km, 
    separate_showers=False
    )

plt.figure(figsize=(8,6), dpi=200)  
bin_15km, mean_15km, rms_low, rms_high = mean_rms_plot(
    trimmed_showers_15km,comp_depths_15km, label='15 km', facecolor='tab:red'
    )


sample_shower_column = trimmed_showers_15km[:,500::500].T
sample_depth_column = comp_depths_15km[:,500::500].T

for (depth,showers) in zip(sample_depth_column,sample_shower_column ): 
    #print(depth)
    print(showers.shape) 
    plt.figure(figsize=(8,6),dpi=200) 
    x = showers / np.nanmean(showers)
    plt.hist(x, 
             alpha = .5, 
             edgecolor='black', linewidth=.5,
             label='{:g} g/cm^2'.format(depth[418]), 
             bins = 100)
    plt.title('Distribution of Composite values')
    plt.xlabel('Particle Content/ Avg particle Content (N)')
    #plt.xlabel('Particle Content (N)')
    plt.ylabel('# of composite showers')
    #plt.xscale('log')
    plt.legend()
