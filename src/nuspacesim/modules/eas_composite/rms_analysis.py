import matplotlib.pyplot as plt
import numpy as np
from composite_ea_showers import CompositeShowers
from fitting_composite_eas import FitCompositeShowers
from plt_routines import mean_rms_plot


make_composites_00km =  CompositeShowers( 
    alt=0, shower_end=8e3, grammage=1
    ) 

comp_showers_00km, comp_depths_00km = make_composites_00km(filter_errors=False) 
 
trimmed_showers_00km, test_depths = make_composites_00km.shower_end_cuts(
    composite_showers=comp_showers_00km, 
    composite_depths=comp_depths_00km, 
    separate_showers=False
    )
#%% 

# sample_shower_column = trimmed_showers_00km[:,500::500].T
# sample_depth_column = comp_depths_00km[:,500::500].T

plt.figure(figsize=(8,6), dpi=200)  
bin_00km, mean_00km, rms_low, rms_high = mean_rms_plot(
    trimmed_showers_00km,comp_depths_00km, label='15 km', facecolor='tab:red'
    )

max_shwr_col = trimmed_showers_00km[:, np.argmax(mean_00km)].T
max_dpth_col = comp_depths_00km[:, np.argmax(mean_00km)].T

# mean and rms plot params
plt.xlim(right=2000)
plt.yscale('log')
plt.axvline(max_dpth_col[1136])


plt.figure(figsize=(8,6),dpi=200) 
#x = showers / np.nanmean(showers)
plt.hist(
    max_shwr_col, 
    alpha=.5, 
    edgecolor='black', linewidth=.5,
    label='{:g} g/cm^2'.format(max_dpth_col[0]), 
    bins = 100
    )
# plt.scatter(bin_ctr, freq, c='k') 
plt.title('Distribution of Composite values')
plt.xlabel('Particle Content/ Avg particle Content (N)')
#plt.xlabel('Particle Content (N)')
plt.ylabel('# of composite showers')
#plt.xscale('log')
plt.legend()

#%% Dart board monte carlo
n = 10
bins = 20



freq, bin_edgs = np.histogram(max_shwr_col, bins=bins)
bin_ctr = 0.5*(bin_edgs[1:] +bin_edgs[:-1])
bin_size = bin_ctr[1] - bin_ctr[0]


#get random values within the plotting window
rdom_y_ax = np.random.uniform(low=0.0, high=(np.max(freq)+2), size=n)
rdom_x_ax = np.random.uniform(low=0.0, high=(8e7), size=n)

plt.figure(figsize=(8,6),dpi=200) 
plt.hist(
    max_shwr_col, 
    alpha=.5, 
    edgecolor='black', linewidth=.5,
    label='{:g} g/cm^2'.format(max_dpth_col[0]), 
    bins = bins
    )
plt.scatter(bin_ctr, freq, s=2, c='k') 
plt.scatter(rdom_x_ax, rdom_y_ax, s=2, c='r')
plt.xlim(right=8e7)
plt.ylim(top=np.max(freq)+2)

x_residuals = np.abs(rdom_x_ax - bin_ctr[:, np.newaxis]) 
clst_x_idx = np.argmin(x_residuals, axis=0)#x_residuals < bin_size
smlst_resid = np.take_along_axis(x_residuals.T, clst_x_idx[:,None], axis=1) 
within_bin_msk = smlst_resid < bin_size

accepted_bin_idxs = np.unique(clst_x_idx[:,None][within_bin_msk]) 





