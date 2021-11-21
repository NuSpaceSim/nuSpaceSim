from nuspacesim.utils.eas_cher_gen.composite_showers.composite_macros import bin_nmax_xmax
import numpy as np
from scipy import optimize

class FitCompositeShowers():    
    r""" 
    a
    """
    def __init__(self, composite_showers, slant_depths):  
        self.showers  = composite_showers
        self.depths = slant_depths
        
    def modified_gh (self, x, n_max, x_max, x_0, p1, p2, p3): 
        
        particles = (n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0))                  \
                                        **( (x_max - x_0)/(p1 + p2*x + p3*(x**2)) )  ) )   \
                *                                                                           \
                ( np.exp((x_max - x)/(p1 + p2*x + p3*(x**2))) )
                
        return particles

    def gaisser_hillas(self, x, n_max, x_max, x_0, gh_lambda): 
        
        particles = (n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0))  \
                                        **((x_max - x_0)/gh_lambda) )  )   \
                *                                                           \
                ( np.exp((x_max - x)/gh_lambda) )    
        
        return particles

    # def gaisser_hillas (x, n_max, x_max, x_0, gh_lambda): 
        
    #     exp_term = (x_max - x_0)/gh_lambda
    #     exponential = np.exp( (x_max - x) / gh_lambda)
    
    #     base_term = np.nan_to_num ( (x - x_0) / (x_max - x_0) )
        
    #     shower_content = n_max * (base_term**exp_term) * exponential

    def fit_quad_lambda (self, comp_shower, depth): 
        event_tag =  comp_shower[0]
        decay_tag_num =  comp_shower[1]
        
        comp_shower = comp_shower[2:]
        depth = depth[2:]
        
        nmax, xmax = bin_nmax_xmax( bins=depth, particle_content=comp_shower)
        
        fit_params, covariance = optimize.curve_fit(
                            f=self.modified_gh, 
                            xdata=depth, 
                            ydata=comp_shower,
                            p0=[nmax,xmax,0,70,-0.01,1e-05], 
                            bounds=([0,0,-np.inf,-np.inf,-np.inf,-np.inf], 
                                    [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
                            )
        
        fits = np.array([event_tag, decay_tag_num, *fit_params])
        return fits
    
    def fit_const_lambda (self, comp_shower, depth): 
        event_tag =  comp_shower[0]
        decay_tag_num =  comp_shower[1]
        
        comp_shower = comp_shower[2:]
        depth = depth[2:]
        
        nmax, xmax = bin_nmax_xmax(bins=depth, particle_content=comp_shower)
        
        fit_params, covariance = optimize.curve_fit(
                            f=self.gaisser_hillas, 
                            xdata=depth, 
                            ydata=comp_shower,
                            p0=[nmax,xmax,0,70], 
                            bounds=([0,0,-np.inf,-np.inf], 
                                    [np.inf,np.inf,np.inf,np.inf])
                            )
        
        
        fits = np.array([event_tag, decay_tag_num, *fit_params])
        return fits
    
    def reco_showers (self, fit_params, depth): 
        r"""
        Reconstruct a composite shower based on fits
        """
        event_tag =  fit_params[0]
        decay_tag_num =  fit_params[1]
        
        depth = depth[2:]
        
        reconstructed = self.gaisser_hillas(depth, *fit_params[2:])
        #reco_shower = np.array([event_tag, decay_tag_num, reconstructed])
        reco_shower  = np.r_[event_tag, decay_tag_num,reconstructed]
        return reco_shower
    
    def reco_chi (self, composite_shower, reco_shower): 
        
        event_tag =  reco_shower[0]
        decay_tag_num =  reco_shower[1]
        
        composite_shower = composite_shower[2:]
        reco_shower = reco_shower[2:]
        
        chisquare = np.sum((composite_shower- reco_shower)**2 /  reco_shower) 
        dof =  np.size(reco_shower) - 4
        reduced_chisquare = chisquare/dof
        
        from scipy import stats
        # our constant lambda gh has 4 fitting parameters
        p_value = stats.chi2.sf (chisquare, dof)
        
        fit_results = np.r_[event_tag, decay_tag_num, reduced_chisquare, p_value]
        return fit_results
    
    def __call__ (self): 
        
        gh_fits = np.empty([self.showers.shape[0], 6]) 
        
        for row,(shower, depth) in enumerate(zip(self.showers, self.depths)):
            
            shower_fit = self.fit_const_lambda(comp_shower=shower, depth=depth)
            gh_fits[row,:] = shower_fit
            print('Fitting', row)
        return gh_fits