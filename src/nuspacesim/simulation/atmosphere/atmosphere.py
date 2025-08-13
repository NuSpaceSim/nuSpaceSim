import numpy as np
from scipy.integrate import quad
from scipy.constants import Avogadro
from abc import ABC, abstractmethod
from importlib.resources import as_file, files


class Atmosphere(ABC):
    '''This is the abstract base class for creating atmosphere objects. It
    contains the necessary methods related to the atmosphere for use in the
    rest of CHASM.
    '''
    temperature_sea_level = 288.15    # K
    pressure_sea_level    = 101325    # Pa
    density_sea_level     = 1.225     # kg/m3
    air_mol_weight        = 28.9644   # amu
    gravity               = 9.80665   # m/s2
    gas_constant          = 8.31432   # J/MolK
    gMR           = gravity * air_mol_weight / gas_constant
    avo = Avogadro

    def number_density(self,h):
        '''
        This method returns the approximate number density of air molecules as
        a function of height.

        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            N - number density [N/m3]
        '''
        return self.density(h) * 1.e3 *  self.avo / self.air_mol_weight

    def depth(self,h1,h2=None):
        """
        This function returns atmospheric depth. It is the integral of atmospheric density between two heights.

        Parameters:
        These parameters can be ndarrays or single values.

        h1 - height 1 in atmosphere. This can be an ndarray or a single value. [m]
        h2 - height 2; Default is hMaxAtm. This can be an ndarray or a single value [m]

        If both h1 and h2 are ndarrays, they must be the same size (the length
        of the shorter array is used).

        If h1 or h2 is greater than hMaxAtm, hMaxAtm is used.

        Returns:
        The integral of rho from h1 to h2. The result is converted into g/cm2.

        """
        if h2 is None:
            h2 = self.maximum_height*np.ones_like(h1)

        if type(h1) is not np.ndarray and type(h2) is not np.ndarray:
            h1 = np.array([h1],dtype=float)
            h2 = np.array([h2],dtype=float)
            nin = 0
        elif type(h2) is not np.ndarray:
            h2 = h2*np.ones_like(h1)
            nin = len(h1)
        elif type(h1) is not np.ndarray:
            h1 = h1*np.ones_like(h2)
            nin = len(h2)
        else:
            nin = min(len(h1),len(h2))

        A = h1.copy()
        B = h2.copy()
        A[A<self.minimum_height] = self.minimum_height
        B[B<self.minimum_height] = self.minimum_height
        A[A>self.maximum_height] = self.maximum_height
        B[B>self.maximum_height] = self.maximum_height

        depth = np.array([quad(self.density,a,b)[0] for a,b in zip(A,B)])
        depth /= 10. # 1 km/m2 == 1000/10,000 g/cm2

        if nin == 0:
            return depth[0]
        else:
            return depth
        
    @property
    @abstractmethod
    def name(self) -> str:
        '''This property should the name of the atm implementation'''

    @property
    @abstractmethod
    def altitudes(self) -> np.ndarray:
        '''This property should return an array of the altitudes.'''

    @property
    @abstractmethod
    def maximum_height(self) -> float:
        '''This property should return the maximum tabulated height.'''

    @abstractmethod
    def density(self,h):
        """
        This function returns density as a function of height.

        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            rho - density [kg/m3]
        """

    @abstractmethod
    def thickness(self) -> np.ndarray:
        '''This method should return an array of the thicknesses between
        altitudes.
        '''

    @abstractmethod
    def delta(self,h):
        """
        This function returns the difference of the index-of-refraction from unity.

        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            delta - equal to n - 1.
        """

class USStandardAtmosphere(Atmosphere):
    """
    Class containing constants and methods for using the US Standard Atmosphere of 1976

    The temperature is assumed to change linearly with height above sea level. From this
    and the assumption of hybrdostatic equilibrium the pressure and density are calculated.

    While instantiated, the default parameters give the US Standard Atmosphere. Other
    atmospheres can be provided
    """

    # Class constants for 1976 US Standard Atmosphere
    temperature_sea_level = 288.15    # K
    pressure_sea_level    = 101325    # Pa
    density_sea_level     = 1.225     # kg/m3
    air_mol_weight        = 28.9644   # amu
    gravity               = 9.80665   # m/s2
    gas_constant          = 8.31432   # J/MolK
    gMR           = gravity * air_mol_weight / gas_constant
    avo = Avogadro

    def __init__(self,altitudes=None,rel_pressure=None,temperatures=None,temp_gradient=None):
        """
        Create and instance of an atmospheric model.

        If no parameters are provided, the US Standard Atmosphere is
        used. It uses these values
            altitudes    = np.array([0.,11000.,20000.,32000.,47000.,51000.,71000.,84852.])
            rel_pressure = np.array([1.,
                                     2.23361105092158e-1,5.40329501078488e-2,8.56667835929167e-3,
                                     1.09456013377711e-3,6.60635313285837e-4,3.90468337334393e-5,
                                     3.68500952357479e-6])
            temperatures = np.array([288.15,216.65,216.65,228.65,270.65,270.65,214.65,186.946])
            temp_gradient = np.array([-0.0065,0.,0.001,0.0028,0.,-0.0028,-0.002,0.])

        If another atmosphere is to be created, each of the parameters should be an identical length
        ndarray. All the parameters must be specified or none of them. The altitudes array must
        be ordered.
        """
        if altitudes is None and rel_pressure is None and \
           temperatures is None and temp_gradient is None:
            self.altitudes    = np.array([0.,11000.,20000.,32000.,47000.,51000.,71000.,84852.])            # m above SL
            self.rel_pressure = np.array([1.,
                                       2.23361105092158e-1,5.40329501078488e-2,8.56667835929167e-3,
                                       1.09456013377711e-3,6.60635313285837e-4,3.90468337334393e-5,
                                       3.68500952357479e-6])
            self.temperatures = np.array([288.15,216.65,216.65,228.65,270.65,270.65,214.65,186.946]) # K
            self.temp_gradient = np.array([-0.0065,0.,0.001,0.0028,0.,-0.0028,-0.002,0.])    # K/m
        else:
            self.altitudes     = altitudes
            self.rel_pressure  = rel_pressure
            self.temperatures  = temperatures
            self.temp_gradient = temp_gradient
        self.maximum_height = self.altitudes[-1]
        self.minimum_height = self.altitudes[0]
        self.name = 'USSTANDARDATM1976'

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def maximum_height(self):
        return self._maximum_height
    
    @maximum_height.setter
    def maximum_height(self, value: float):
        self._maximum_height = value

    @property
    def altitudes(self):
        return self._altitudes
    
    @altitudes.setter
    def altitudes(self, value: np.ndarray):
        self._altitudes = value

    def atmosphere(self,h):
        """
        This function returns atmospheric temperature, pressure, and density as a function of height.

        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            T   - temperature [K]
            P   - pressure [Pa]
            rho - density [kg/m3]
        """
        if type(h) is not np.ndarray:
            h = np.array([h],dtype=float)
            nin = 0
        else:
            nin = len(h)

        # Find the entry in the tables for each height
        too_low  = h < self.minimum_height
        too_high = h > self.maximum_height
        indx  = np.searchsorted(self.altitudes,h,side='right')
        idx = indx - 1

        # Find the temperature at height
        altitude        = self.altitudes[idx]
        base_temp       = self.temperatures[idx]
        temp_gradient   = self.temp_gradient[idx]
        delta_altitude  = h - altitude
        temperature     = base_temp + temp_gradient*delta_altitude

        # Find the relative pressure at height
        base_rel_pressure = self.rel_pressure[idx]
        flat = np.abs(temp_gradient) < 1.e-10
        rel_pressure = np.empty_like(h)
        rel_pressure[flat]  = base_rel_pressure[flat]  * \
                              np.exp(-self.gMR/1000*delta_altitude[flat]/base_temp[flat])
        rel_pressure[~flat] = base_rel_pressure[~flat] * \
                              (base_temp[~flat]/temperature[~flat])**(self.gMR/1000/temp_gradient[~flat])
        pressure = rel_pressure * self.pressure_sea_level
        density  = rel_pressure * self.density_sea_level * self.temperature_sea_level/temperature

        temperature[too_low] = self.temperature_sea_level
        pressure[too_low]    = self.pressure_sea_level
        density[too_low]     = self.density_sea_level
        temperature[too_high] = 0.
        pressure[too_high]    = 0.
        density[too_high]     = 0.

        T = temperature
        P = pressure
        rho = density

        if nin == 0:
            return T[0],P[0],rho[0]
        else:
            return T,P,rho

    def temperature(self,h):
        """
        This function returns temperature as a function of height.

        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            T - temperature [K]
        """
        T,_,_ = self.atmosphere(h)
        return(T)

    def pressure(self,h):
        """
        This function returns pressure as a function of height.

        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            P - pressure [Pa]
        """
        _,P,_ = self.atmosphere(h)
        return(P)

    def density(self,h):
        """
        This function returns density as a function of height.

        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            rho - density [kg/m3]
        """
        _,_,rho = self.atmosphere(h)
        return(rho)

    def delta(self,h):
        """
        This function returns the difference of the index-of-refraction from unity.

        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            delta - equal to n - 1.
        """
        T,P,_ = self.atmosphere(h)
        P /= 1000.       # Pa -> kPa
        return 7.86e-4*P/T

    def thickness(self) -> np.ndarray:
        '''This method returns an array of the thicknesses between
        altitudes.
        '''
        thicks = np.empty(self.altitudes.size, dtype=np.float64)
        thicks[:-1] = self.depth(self.altitudes[:-1],self.altitudes[1:])
        thicks[-1] = 0.
        return thicks