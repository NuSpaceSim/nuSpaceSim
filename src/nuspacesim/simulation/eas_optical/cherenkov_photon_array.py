import numpy as np

class CherenkovPhotonArray:
    """A class for using the full array of CherenkovPhoton values
    at a series of stages, t, and atmospheric delta values.
    """

    def __init__(self,gg):
        """Create a CherenkovPhotonArray object from a npz file. The
        npz file should contain the Cherenkov angular distributions
        for a set of stages and delta values. It should also contain
        arrays of the values for t, delta, and theta.

        Parameters:
            npzfile: The input file containing the angular distributions
                and values.

        The npz file should have exactly these keys: "gg_t_delta_theta",
        "t", "delta", and "theta".
        """
        self.gg_t_delta_theta = gg['gg_t_delta_theta']
        self.t = gg['t']
        self.delta = gg['delta']
        self.theta = gg['theta']

    def __repr__(self):
        return "gg: %d %.0f<t<%.0f, %d %.2e<delta<%.2e, %d %.2e<theta<%.2e"%(
            len(self.t),self.t.min(),self.t.max(),
            len(self.delta),self.delta.min(),self.delta.max(),
            len(self.theta),self.theta.min(),self.theta.max())

    def angular_distribution(self,t,delta):
        """Return the intepolated angular distribution at the
        given value of t and delta.

        The t interpolation is arithmetic, while the delta
        interpolation is geometric.

        Parameters:
            t: Shower stage
            delta: Atmospheric delta

        Returns:
            ng_t_delta_Omega: Angular spectrum array at t & delta
        """
        if t > self.t.max():
            t = self.t.max()
        it = np.searchsorted(self.t,t,side='right')
        jt = it-1
        if jt<0:
            jt = 0
            it = 1
        elif it>=len(self.t):
            jt = len(self.t)-2
            it = len(self.t)-1
        st = (t - self.t[jt])/(self.t[it]-self.t[jt])

        id = np.searchsorted(self.delta,delta,side='right')
        jd = id-1
        if jd<0:
            jd = 0
            id = 1
        elif id>=len(self.delta):
            jd = len(self.delta)-2
            id = len(self.delta)-1
        sd = np.log(          delta/self.delta[jd] ) / \
             np.log( self.delta[id]/self.delta[jd] )
        if sd > 1.e1:
            sd = 1.e1
        gg4 = self.gg_t_delta_theta[jt:jt+2,jd:jd+2]
        gg2 = gg4[0]*(1-st) + gg4[1]*st
        zeromask = gg2 == 0
        gg2[zeromask] = 1.e-8 #occasionally there will be some zeros in this array, we dont want to divide by zero
        frac = gg2[1]/gg2[0]
        gg  = gg2[0]*np.sign(frac)*(np.abs(frac))**sd
        return gg

    def interpolate(self,t,delta,theta):
        """Return the intepolated value of the angular distribution at the
        given value of t, delta and theta.

        The t interpolation is arithmetic, while the delta and theta
        interpolations are geometric.
        """
        gg = self.angular_distribution(t,delta)

        iq = np.searchsorted(self.theta,theta,side='right')
        jq = iq-1
        if jq<0:
            jq = 0
            iq = 1
        elif iq>=len(self.theta):
            jq = len(self.theta)-2
            iq = len(self.theta)-1
        sq = np.log(          theta/self.theta[jq] ) / \
             np.log( self.theta[iq]/self.theta[jq] )

        gg2 = gg[jq:jq+2]
        gg1 = gg2[0]*(gg2[1]/gg2[0])**sq
        return gg1
