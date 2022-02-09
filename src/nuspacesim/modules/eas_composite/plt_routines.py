import numpy as np
import matplotlib.pyplot as plt


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
