import numpy as np   
    
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
        # (pdg.lbl.gov/2015/AtomicNuclearProperties/HTML/air_dry_1_atm.html
        x_naught = 36.62 
        # critical energy for electromagnetic cascades in the air
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
        self, n_max, x_max, x_0, p1, p2, p3, shower_end:int = 2000, grammage:int = 1
        ):
        
        
        scaled_n_max = n_max * self.table_decay_e
        
        # allows negative starting depths
        bins =  int((shower_end + np.round(np.abs(x_0)) ) / grammage) 
        padded_vec_len = (shower_end/ grammage) + 600
        #x = np.linspace( np.round(x_0), shower_end, bins, endpoint=True) #slant depths g/cm^2 
        x = np.arange(np.round(x_0), shower_end + 1, grammage)
        # constrains starting depths 
        # x = np.linspace( 0, shower_end, int(shower_end/grammage)) #slant depths g/cm^2    
        
        #calculating gaisser-hillas function
        gh_lambda = p1 + p2*x + p3*(x**2) 
        
        exp1 = (x_max - x_0) / gh_lambda
       
        term1 = scaled_n_max *  np.nan_to_num( ((x - x_0) / (x_max - x_0)) **exp1 )
        
        exp2 = (x_max - x) / gh_lambda
        term2 = np.exp(exp2) 
        
        f = np.nan_to_num(term1 * term2).astype(int)
        #print(np.min(f))
        if np.min(f) < 0:
            break_point = int(np.argwhere(f < 0)[0])
            #print(break_point)
            f[break_point:] = 0
        #LambdaAtx_max = p1 + p2*x_max + p3*(x_max**2)
        #t = (x - x_max)/36.62 #shower stage
        #print(int(padded_vec_len), len(x))
        x = np.pad(x, (int(padded_vec_len - len(x) ), 0), 'constant')
        f = np.pad(f, (int(padded_vec_len - len(f) ), 0), 'constant')
        #tag the outputs with the event number
        # x = np.r_[self.event_tag,self.decay_tag,x].astype(int)
        # f = np.r_[self.event_tag,self.decay_tag,f].astype(int)
        x = np.r_[x, self.event_tag, self.decay_tag].astype(int)
        f = np.r_[f, self.event_tag, self.decay_tag].astype(int)

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