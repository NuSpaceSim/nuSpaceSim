import h5py 
import importlib_resources
import numpy as np 
import time  
from scipy import optimize
# from ...utils.eas_cher_gen.composite_showers.composite_macros import bin_nmax_xmax

np.seterr(all='ignore')
try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files
 
class ShowerParameterization: 
    r""" Class calculating particle content per shower for 
    GH and Greissen parametrizations. 
    """
    def __init__(self, table_decay_e, event_tag, decay_tag):
        
        # pythia related params
        self.table_decay_e = table_decay_e
        self.event_tag = event_tag
        self.decay_tag = decay_tag
        
    def greisen(self, shower_end = 2000, grammage = 1, table_energy = 100e15):
        #slant depths 
        x = np.linspace( 1, shower_end, int(shower_end / grammage))  
      
        # x0 is the radiation length in air ~ 37 g/cm^2 
        #(pdg.lbl.gov/2015/AtomicNuclearProperties/HTML/air_dry_1_atm.html
        x_naught = 36.62 
        
        #critical energy for electromagnetic cascades in the air
        e_c = 87.92e6 
        
        # energy of photon initiated shower (eV) Gaisser 303 
        e_0 = self.table_decay_e * table_energy
        beta_0 = np.log(e_0 / e_c)
        t = x / x_naught 
        
        # age parameter, n = 0
        s = (0 + 3*t)/ (t + 2*beta_0 ) 
       
        x_max = x_naught * np.log(e_0 / e_c)
        n_max = ( 0.31 / pow(beta_0, 1/2) ) * (e_0 / e_c)
    
        term1 = ( 0.31 / pow(beta_0, 1/2) )
        term2 = np.exp ( t - (t * (3/2)  * np.log (s) ) )
        
        content = term1 * term2 # particle content 
         
        return x, content, self.event_tag
        
    def gaisser_hillas(
        self, n_max, x_max, x_0, p1, p2, p3, shower_end = 2000, grammage = 1
        ):
        
        scaled_n_max = n_max * self.table_decay_e
        
        # allows negative starting depths
        # bins = int ( (shower_end + np.abs(x_0) ) / grammage)
        # x = np.linspace( int(x_0), shower_end, bins, endpoint=False) #slant depths g/cm^2 
        
        # constrains starting depths 
        x = np.linspace( 1, shower_end, shower_end) #slant depths g/cm^2    
        
        #calculating gaisser-hillas function
        gh_lambda = p1 + p2*x + p3*(x**2) 
        
        exp1 = (x_max - x_0) / gh_lambda
       
        term1 = scaled_n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0)) **exp1 )
        
        exp2 = (x_max - x) / gh_lambda
        term2 = np.exp(exp2) 
        
        f = np.nan_to_num (term1 * term2) 
        
        #LambdaAtx_max = p1 + p2*x_max + p3*(x_max**2)
        #t = (x - x_max)/36.62 #shower stage
        
        #tag the outputs with the event number
        x = np.r_[self.event_tag,self.decay_tag,x]
        f = np.r_[self.event_tag,self.decay_tag,f]
        return x, f
    
    # def gaisser_hillas(
    #     self, n_max, x_max, x_0, p1, p2, p3, shower_end=2000, grammage=1
    # ):

    #     scaled_n_max = n_max * self.table_decay_e

    #     # constrains starting depths
    #     x = np.linspace(1, shower_end, shower_end)  # slant depths g/cm^2

    #     # calculating gaisser-hillas function
    #     print(p1.shape, p2.shape, x.shape, p3.shape)
    #     v2 = np.multiply.outer(p2, x)
    #     v3 = np.multiply.outer(p3, x ** 2)
    #     gh_lambda = p1[:, None] + v2 + v3
    #     print(gh_lambda.shape)

    #     exp1 = (x_max - x_0)[:, None] / gh_lambda

    #     a1 = x - x_0[:, None]
    #     a2 = x_max - x_0
    #     a3 = a1 / a2[:, None]
    #     a4 = a3 ** exp1
    #     term1 = scaled_n_max[:, None] * np.nan_to_num(a4)

    #     exp2 = (x_max[:, None] - x) / gh_lambda
    #     term2 = np.exp(exp2)

    #     f = np.nan_to_num(term1 * term2)

    #     return x, f, self.event_tag
        
    # def single_particle_showers(self, gh_params = sefelectron_gh, tau_energies):
    #     shower = ShowerParameterization(
    #         table_decay_e=tau_energies, event_tag=tau_energies
    #     )
    #     depths, _, _ = shower.gaisser_hillas(
    #         n_max=gh_params[:, 4],
    #         x_max=gh_params[:, 5],
    #         x_0=gh_params[:, 6],
    #         p1=gh_params[:, 7],
    #         p2=gh_params[:, 8],
    #         p3=gh_params[:, 9],
    #     )
    #     return depths
  
    
class CompositeShowers():    
    r""" Make composite showers with constituent electrons, gamma, and pions, 
    contributions scaled by sampled tau energies. 
    
    """
    def __init__(self, shower_end: int = 2000, grammage: int = 1):  
        
        with as_file(
            files('nuspacesim.data.conex_gh_params') / 'electron_EAS_table.h5'
        ) as path:
            data = h5py.File(path, 'r')
            electron_gh = np.array(data.get('EASdata_11'))
            
        with as_file(
            files('nuspacesim.data.conex_gh_params') / 'gamma_EAS_table.h5'
        ) as path:
            data = h5py.File(path, 'r')
            gamma_gh = np.array(data.get('EASdata_22'))
            
        with as_file(
            files('nuspacesim.data.conex_gh_params') / 'pion_EAS_table.h5'
        ) as path:
            data = h5py.File(path, 'r')
            pion_gh = np.array(data.get('EASdata_211'))        
            
        with as_file(
            files('nuspacesim.data.pythia_tau_decays') / 'new_tau_100_PeV.h5'
        ) as path:
            data = h5py.File(path, 'r')
            tau_decays = np.array(data.get('tau_data'))  
        
        self.electron_showers = electron_gh
        self.gamma_showers = gamma_gh
        self.pion_showers = pion_gh
        
        self.tau_tables = tau_decays
        
        # shower development characterisitics
        self.shower_end = shower_end
        self.grammage = grammage
        
    def conex_params(self):
        r""" Return sampled GH paramters for electrons, pions, and gammas.
        """
        return self.electron_showers, self.pion_showers, self.gamma_showers
        
    
    def tau_daughter_energies(self):
        r"""Isolate energy contributions of pion, kaon, electron, 
        and gamma daughters.
        
        Used to scale Nmax of respective GH showers.  
        
        """
        electron_mask = self.tau_tables[:,2] == 11
        electron_energies = self.tau_tables[electron_mask] [:,[0,1,-1]] 
    
        gamma_mask = self.tau_tables[:,2] == 22
        gamma_energies = self.tau_tables[gamma_mask ] [:,[0,1,-1]] 
        
        # kaons and pions treated the same 
        pion_kaon_mask = ((self.tau_tables[:,2] == 211) | (self.tau_tables[:,2] == -211)) | \
                         ((self.tau_tables[:,2] == 321) | (self.tau_tables[:,2] == -321))
        pion_energies = self.tau_tables[pion_kaon_mask] [:,[0,1,-1]]
        
        # each row has [event_num, energy ] 
        return electron_energies, pion_energies, gamma_energies 
        
    def single_particle_showers(self, tau_energies, gh_params): 
        r""" Create single particle showers Nmax scaled by pythia energies
        from same PID.
        """
        
        # pre-allocate arrays, make room for event tag and decay tag 
        showers = np.empty([ gh_params.shape[0], int((self.shower_end/ self.grammage) + 2)]) 
        depths =  np.empty([ gh_params.shape[0], int((self.shower_end/ self.grammage) + 2)]) 
        
        for row,(shower_params,tau_dec_e) in enumerate(zip(gh_params, tau_energies)):
            
            shower = ShowerParameterization (
                table_decay_e=tau_dec_e[-1], event_tag=tau_dec_e[0], decay_tag=tau_dec_e[1]
            )
            
            depth, shower_content= shower.gaisser_hillas(
                                    n_max = shower_params[4],
                                    x_max = shower_params[5],
                                    x_0 = shower_params[6],
                                    p1 = shower_params[7],
                                    p2 = shower_params[8],
                                    p3 = shower_params[9],
                                    shower_end = self.shower_end,
                                    grammage = self.shower_end)
            
            showers[row,:] = shower_content
            depths[row,:] = depth 
            
        return showers, depths
    
    
    def composite_showers(self, **kwargs):
        r""" From single particle showers, create composite showers by
        summing each event. 
        
        Returns:
            Composite Showers in # of Charged Particles with columns 0 and 1
            being the event number and decay ID, repsectively.
            
            Uniform bins of the showers, all constrained to be uniform, 
            set by end_shower and grammage.
    
        """
        
        # read in all arrays and get a main array containing all of them. 
        single_showers = kwargs.get('single_showers')
        single_shower_bins = kwargs.get('shower_bins')
        single_showers = np.concatenate((single_showers), axis=0) 
        single_shower_bins = np.concatenate((single_shower_bins), axis=0) 
        # sort by event number
        single_showers = single_showers[single_showers[:,0].argsort()]
        single_shower_bins  = single_shower_bins[single_shower_bins[:,0].argsort()]
            
        grps, idx = np.unique(single_showers[:,0], return_index=True, axis=0)
        unique_event_tags = np.take(single_showers[:,1], idx)
        counts = np.add.reduceat(single_showers[:, 2:], idx)
        composite_showers = np.column_stack((grps,unique_event_tags,counts))
        composite_depths = np.unique(single_shower_bins, axis = 0)

        return  composite_showers, composite_depths
    
    def __call__ (self):
        
        electron_gh, pion_gh, gamma_gh= self.conex_params()
        electron_e, pion_e, gamma_e = self.tau_daughter_energies()
        
        elec_showers, elec_depths = self.single_particle_showers(
            tau_energies=electron_e, gh_params=electron_gh
            )
    
        pion_showers, pion_depths = self.single_particle_showers(
            tau_energies=pion_e, gh_params=pion_gh
            )
    
        gamm_showers, gamm_depths = self.single_particle_showers(
            tau_energies=gamma_e, gh_params=gamma_gh
            )
        
        comp_showers, depths = self.composite_showers( 
            single_showers = (elec_showers, pion_showers, gamm_showers), 
            shower_bins = (elec_depths, pion_depths, gamm_depths)
        )
            
        return comp_showers, depths



def bin_nmax_xmax (bins, particle_content):
    r"""
    given an array of Slant Depths and Particle Content values for the same particle 
    (can be any number of events, but need to be same size), returns the Nmax and Xmax Values 
    
    per row (if composite showers and bins are inputted, per event) 
    
    intended to use for nmax and xmax distribution analysis 
    """

    try:
        bin_nmax = np.amax(particle_content, axis = 1) 
        bin_nmax_pos =  np.nanargmax(particle_content, axis = 1)
        bin_xmaxs = bins[np.arange(len(bins)), bin_nmax_pos]
    except: 
        bin_nmax = np.amax(particle_content) 
        bin_nmax_pos =  np.nanargmax(particle_content)
        bin_xmaxs = bins[bin_nmax_pos]
    
    return bin_nmax,  bin_xmaxs


def modified_gh (x, n_max, x_max, x_0, p1, p2, p3): 
    
    particles = (n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0))                  \
                                    **( (x_max - x_0)/(p1 + p2*x + p3*(x**2)) )  ) )   \
            *                                                                           \
            ( np.exp((x_max - x)/(p1 + p2*x + p3*(x**2))) )
            
    return particles


def const_lambda (x, n_max, x_max, x_0, gh_lambda): 
    
    particles = (n_max * np.nan_to_num ( ((x - x_0) / (x_max - x_0))                  \
                                   **((x_max - x_0)/gh_lambda) )  )                    \
            *                                                                           \
            ( np.exp((x_max - x)/gh_lambda) )    
    
    return  particles

def fit_composites (comp_shower, depth): 
    event_tag =  comp_shower[0]
    decay_tag_num =  comp_shower[1]
    
    comp_shower = comp_shower[2:]
    depth = depth[2:]
    
    nmax, xmax = bin_nmax_xmax(
        bins=depth, particle_content=comp_shower
        )
    
    fit_params, covariance = optimize.curve_fit(
                        f=modified_gh, 
                        xdata=depth, 
                        ydata=comp_shower,
                        p0=[nmax,xmax,0,70,-0.01,1e-05], 
                        bounds=([0,0,-np.inf,-np.inf,-np.inf,-np.inf], 
                                [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
                        )
    
    fit_n_max = fit_params[0]
    fit_x_max = fit_params[1]
    fit_x_0 = fit_params[2]
    fit_p1 = fit_params[3]
    fit_p2 = fit_params[4]
    fit_p3 = fit_params[4]
    
    fits = np.array([event_tag, decay_tag_num, fit_n_max, fit_x_max, fit_x_0, fit_p1, fit_p2, fit_p3])
    return fits
    
def fit_composites_1 (comp_shower, depth): 
    event_tag =  comp_shower[0]
    decay_tag_num =  comp_shower[1]
    
    comp_shower = comp_shower[2:]
    depth = depth[2:]
    
    nmax, xmax = bin_nmax_xmax(
        bins=depth, particle_content=comp_shower
        )
    
    fit_params, covariance = optimize.curve_fit(
                        f=const_lambda, 
                        xdata=depth, 
                        ydata=comp_shower,
                        p0=[nmax,xmax,0,70], 
                        bounds=([0,0,-np.inf,-np.inf], 
                                [np.inf,np.inf,np.inf,np.inf])
                        )
    
    fit_n_max = fit_params[0]
    fit_x_max = fit_params[1]
    fit_x_0 = fit_params[2]
    fit_gh_lambda = fit_params[3]
    # fit_p2 = fit_params[4]
    # fit_p3 = fit_params[4]
    
    fits = np.array([event_tag, decay_tag_num, fit_n_max, fit_x_max, fit_x_0, fit_gh_lambda])
    return fits  


if __name__ == '__main__': 
    t0 = time.time()
    
    x = CompositeShowers()
    comp_showers, depths = x() 
    
    fits = np.empty([comp_showers.shape[0], 6]) 
    
    # do next: lambda with a one percent cut.
    # show distributions of chi squares. 
    # constant lambda plots rebounds 
    # 
    
    for row,(shower, depth) in enumerate(zip(comp_showers, depths)):
        try:
            shower_fit = fit_composites_1( comp_shower=shower, depth= depth )
        
            fits[row,:] = shower_fit
        except: 
            print("Can't fit shower", row)
            #a = np.empty((0,6))
            #fits[row,:] = a.fill(np.nan)
    
    
    
    
    t1 = time.time()
    total = t1-t0 
    print(total)

    