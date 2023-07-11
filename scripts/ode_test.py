import numpy as np
from scipy.integrate import solve_ivp

import nuspacesim.simulation.eas_optical.atmospheric_models as atm
from nuspacesim.utils.ode import dormand_prince_rk54

z_sol_Xs, z_sol_zs, yf = dormand_prince_rk54(
    func=lambda t, y, theta: atm.slant_depth_inverse_func(y, theta),
    t_span=[np.array([0.0]), np.array([1000.0])],
    y0=np.array([0.5]),
    args=np.array([0.25 * np.pi]),
    t_eval=[np.array([0.10]), np.array([1000.0])],
    rtol=1e-6,
    atol=1e-3,
)

# solve_ivp(
#     fun    = lambda t, y, theta: atm.slant_depth_inverse_func(y, theta),
#     t_span = [0.0, 1000.0],
#     y0     = np.array([0.5]),
#     args   = np.array([0.25 * np.pi]),
#     t_eval = np.array([0.10, 1000.0]),
#     rtol   =1e-6,
#     atol   =1e-3
# )
