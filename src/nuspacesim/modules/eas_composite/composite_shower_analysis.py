import matplotlib.pyplot as plt
import numpy as np

from composite_ea_showers import CompositeShowers, FitCompositeShowers

make_composites = CompositeShowers(shower_end=30000, grammage=10)
comp_showers, depths =  make_composites()     

cut_off_depths = np.empty([np.shape(comp_showers)[0]])
# get rebound x
for row,(shower,bins) in enumerate(zip(comp_showers,depths)):
    max_val = np.max(shower) 
    threshold = max_val * 0.01
    # max_pos = int(np.argwhere (shower == max_val))
    
    # shower =  shower[max_pos:]
    # bins = bins[max_pos:]
    
    # shower_mask = (shower < threshold)
    # masked_shower = shower[shower_mask]
    # masked_depths = bins[shower_mask]
    # #print(masked_depths)
    places_to_cut = np.argwhere(shower[2:] <= threshold) 
    #highest index where it is still less than (after or before peak, depends on x_limit) 
    cut_here = np.amax(places_to_cut) 
    #trims the arrays from 0 to where number of  particles < n_max_fraction
    trimmed_f = np.array (shower[0:cut_here] )
    trimmed_x = np.array (bins[0:cut_here] )
    #get the slant depth when f exceeds n_max_fraction
    cutoff_depth = np.amax(trimmed_x[2:]) 
    cut_off_depths[row] = cutoff_depth
    #print(cutoff_depth )
#%%
plt.figure(figsize=(8, 5), dpi= 120)  
plt.hist (cut_off_depths[cut_off_depths < 10000 ], bins = 30) 
plt.title ('Shower Depth for Rebound Cutoff (<0.01 Nmax)')
plt.ylabel('Counts')
plt.xlabel('Depth (g/cm^2)')

#%%
make_composites = CompositeShowers(shower_end=2000, grammage=1)
comp_showers, depths =  make_composites()     
get_fits = FitCompositeShowers(comp_showers, depths)
fits = get_fits()

#%%

def giasser_hillas( x, n_max, x_max, x_0, gh_lambda): 
    
    particles = (n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0))  \
                                    **((x_max - x_0)/gh_lambda) )  )   \
            *                                                           \
            ( np.exp((x_max - x)/gh_lambda) )    
    
    return particles

plt.figure(figsize=(8, 5), dpi= 120)
plt.plot(depths[1,2:], comp_showers[1,2:], label = 'generated composite shower')
plt.plot(depths[1,2:],giasser_hillas(depths[10,2:], *fits[10,2:]), label ='shower fit, const. lambda')
plt.yscale('log')
plt.ylim(bottom = 1)
plt.legend()
#plt.draw()
#%%
fitted_showers = np.empty([comp_showers.shape[0], 2000]) 
for row,fit in enumerate(fits):
    fitted_showers[row,:] = giasser_hillas(depths[row,2:], *fits[row,2:])
#%%
plt.figure(figsize=(8, 5), dpi= 120)
average_composites = np.mean(comp_showers[:,2:], axis=0) #Takes the mean of each row (each bin)

# #to get RMS
square = np.square (comp_showers.T) #squares the 2D matrix  
mean = square.mean(axis = 1) #gets mean along each bin
rms =  np.sqrt(mean) #gets root of each mean

rms_error = np.sqrt(np.mean( ( average_composites  - comp_showers[:,2:])**2, axis = 0 ))


plt.figure(figsize=(8, 5), dpi= 120)
plt.plot(depths[1,2:], average_composites,  '--k', label= 'Mean Generated Showers') 

plt.fill_between(depths[1,2:], average_composites - rms_error, average_composites+ rms_error,
                  alpha = 0.5, edgecolor='red', facecolor='crimson', 
                  label = 'Error')

# #plt.title('G-H Plots | ' + str(file_name.split('/')[-1]) +'/'+ str(data_name))
# plt.ylabel('Number of Particles N')
# plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
# plt.xlim(left = 0 )         
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  
plt.yscale('log')
plt.legend()
#%%
plt.figure(figsize=(8, 5), dpi= 120)
average_composites = np.mean(fitted_showers, axis=0) #Takes the mean of each row (each bin)

# #to get RMS
square = np.square (fitted_showers .T) #squares the 2D matrix  
mean = square.mean(axis = 1) #gets mean along each bin
rms =  np.sqrt(mean) #gets root of each mean

rms_error = np.sqrt(np.mean( ( average_composites  - fitted_showers )**2, axis = 0 ))



plt.plot(depths[1,2:], average_composites,  '--k', label= 'Mean Fitted Showers') 

plt.fill_between(depths[1,2:], average_composites - rms_error, average_composites+ rms_error,
                  alpha = 0.5, edgecolor='green', facecolor='green', 
                  label = 'Error')

# #plt.title('G-H Plots | ' + str(file_name.split('/')[-1]) +'/'+ str(data_name))
# plt.ylabel('Number of Particles N')
# plt.xlabel('Slant Depth t ' + '($g \; cm^{-2}$)')
# plt.xlim(left = 0 )         
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  
plt.yscale('log')
plt.legend() 

#%%