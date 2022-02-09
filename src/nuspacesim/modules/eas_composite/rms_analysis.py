import matplotlib.pyplot as plt
import numpy as np
from composite_ea_showers import CompositeShowers
from fitting_composite_eas import FitCompositeShowers
from plt_routines import mean_rms_plot


make_composites_00km =  CompositeShowers( 
    alt=0, shower_end=1000, grammage=10
    ) 

comp_showers_00km, comp_depths_00km =  make_composites_00km(filter_errors=False) 
 
trimmed_showers_00km, test_depths = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km, 
    composite_depths=comp_depths_00km, 
    separate_showers=False
    )

plt.figure(figsize=(8,6), dpi=200)  
bin_00km, mean_00km, rms_low, rms_high = mean_rms_plot(
    trimmed_showers_00km,comp_depths_00km, label='15 km', facecolor='tab:red'
    )


sample_shower_column = trimmed_showers_00km[:,500::500].T
sample_depth_column = comp_depths_00km[:,500::500].T

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
