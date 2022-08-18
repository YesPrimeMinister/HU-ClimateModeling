import numpy as np
import xarray as xr
import math
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt


class Combined_model:

    def __init__(self, Nx, Nz_atmo, Nz_soil, x_size, z_size_atmo, z_size_soil, z, z_0, time, dt):

        self.time = time                        # how long to run the model for
        self.dt = dt                            # time step for the model
        self.n_iters = int(time // dt)          # number of iterations

        self.Nx = Nx
        self.Nz_atmo = Nz_atmo
        self.Nz_soil = Nz_soil

        self.dz_atmo  = z_size_atmo/Nz_atmo     # Distance between z grid points in atmo
        self.dz_soil  = z_size_soil/Nz_soil     # Distance between z grid points in soil
        self.dx       = x_size/Nx               # Distance between x grid points

        # 3D Temperature arrays - (z,x,t)
        self.T_atmo = 280 * np.ones((Nz_atmo, Nx, self.n_iters))
        self.T_soil = np.ones((Nz_soil, Nx, self.n_iters))

        # Constants
        # Soil module
        self.K   = 1.2e-6               # Conductivity

        # Surface energy balance module
        self.c_p = 1004.            # specific heat [J kg^-1 K^-1]
        self.kappa = 0.41          # Von Karman constant [-]
        self.sigma = 5.67e-8       # Stefan-Bolzmann constant
        self.L = 2.83e6            # latent heat for sublimation

        self.z = z                 # Height for temperature measurement - 2 m
        self.z_0 = z_0             # Surface roughness
        # Bulk coefficients



    def _init_index_arrs(self):
        self.idx_atmo = {
            'z':   np.arange(1, self.Nz_atmo-1),
            'z_d': np.arange(0, self.Nz_atmo-2),
            'z_u': np.arange(2, self.Nz_atmo),
            }
        self.idx_soil = {
            'z':   np.arange(1, self.Nz_soil-1),
            'z_d': np.arange(0, self.Nz_soil-2),
            'z_u': np.arange(2, self.Nz_soil),
            }
        self.idx_len = {
            'x':   np.arange(1, self.Nx-1),
            'x_l': np.arange(0, self.Nx-2),
            'x_r': np.arange(2, self.Nx),
            }

    def _init_surface(self):
        """Initialises surface parameters"""

        self.surf_T_now = np.zeros(self.Nx)      # surface temp at a timestep

        # add spatial variability
        self.albedo = 0.3 * np.ones(self.Nx)     # albedo
        self.surf_f = 0.7 * np.ones(self.Nx)     # Relative humidity
        self.surf_rho = 1.1 * np.ones(self.Nx)   # Air density
        self.surf_U = 2.0 * np.ones(self.Nx)     # Wind velocity
        self.surf_z_0 = 1e-3 * np.ones(self.Nx)  # surface Roughness length
        self.surf_p = 1013 * np.ones(self.Nx)    # Pressure

        self.Cs_t = self.kappa**2 / (np.log(self.z/self.surf_z_0)**2)
        self.Cs_q = self.Cs_t

        # add temporal variability
        self.surf_G = 700.0 * np.ones(self.Nx)        # Incoming shortwave radiation


    def surface_balance(self, T_a):
        def E_sat(T):
            """ Saturation water vapor equation """
            Ew = 6.112 * np.exp((17.67*(T-273.16)) / ((T-29.66)))
            return Ew

        def EB_fluxes(T_0,T_a,f,albedo,G,p,rho,U_L,Cs_t,Cs_q):
            """ This function calculates the energy fluxes"""
            # Correction factor for incoming longwave radiation
            eps_cs = 0.23 + 0.433 * np.power(100*(f*E_sat(T_a))/T_a, 1.0/8.0)

            # Calculate turbulent fluxes
            print(f'H_0 = {rho} * {self.c_p} * {Cs_t} * {U_L} * ({T_0} - {T_a})')
            H_0 = rho * self.c_p * Cs_t * U_L * (T_0 - T_a)
            #E_0 = rho * L*0.622/p * Cs_q * U_L * (E_sat(T_0) - f*E_sat(T_0))
            E_0 = rho * self.L*0.622/p * Cs_q * U_L * E_sat(T_0)*(1 - f)

            # Calculate radiation budget
            L_d = eps_cs * self.sigma * T_a**4
            L_u = 1.0 * self.sigma * T_0**4
            Q_0 = (1-albedo) * G

            return (Q_0, L_d, L_u, H_0, E_0)


        def optim_T0(x,T_a,f,albedo,G,p,rho,U_L,Cs_t,Cs_q):
            """ Optimization function for surface temperature:

            Input:
            T_0       : Surface temperature, which is optimized [K]
            T_a       : Air tempearature at height self.z
            """

            Q_0, L_d, L_u, H_0, E_0 = EB_fluxes(x,T_a,f,albedo,G,p,rho,U_L,Cs_t,Cs_q)

            # Get residual for optimization
            print(f'Fluxes: Q_0={Q_0}, L_d={L_d}, L_u={L_u}, H_0={H_0}, E_0={E_0}')
            res = np.abs(Q_0+L_d-L_u-H_0-E_0)
            print('res: ' + str(res))

            # return the residuals
            return res

        # optimise temperature in each point in the x direction

        for x in range(len(self.surf_T_now)):
            print('Computing point number ' + str(x))
            T_0 = 283. # temperature which to optimise from
            surf_args = (T_a[x],self.surf_f[x],self.albedo[x],self.surf_G[x],self.surf_p[x],
                         self.surf_rho[x],self.surf_U[x],self.Cs_t[x],self.Cs_q[x],
                        )

            print('Surface params: ' + str(surf_args))
            res = minimize(optim_T0,x0=T_0,args=surf_args,bounds=((None,400),), \
                         method='L-BFGS-B',options={'eps':1e-8})

            T_0 = res.x[0]

            print(f'Minimised temperatures: {res.x[0]}')
            self.surf_T_now = res.x[0]

        return self.surf_T_now


    def boundary_layer(self, T):
        return T

    def heat_equation(self, T):
        # Set lower BC - Neumann condition
        T[-1, :] = T[-2, :]

        for x in range(T.shape[1]):

                # Update temperature using indices arrays
            T[self.idx_soil['z']] = T[self.idx_soil['z']] + \
                ((T[self.idx_soil['z_u']] - 2*T[self.idx_soil['z']] + \
                T[self.idx_soil['z_d']])/self.dz_soil**2) * self.dt * self.K

            # Copy the new temperature als old timestep values (used for the
            # next time loop step)
            #T[1:-1] = Tnew[1:-1].copy()

        return T


    def step(self, iter_id):
        self.T_atmo[0, :, iter_id+1] = self.surface_balance(self.T_atmo[0, :, iter_id])
        self.T_soil[-1,:, iter_id+1] = self.T_atmo[0, :, iter_id]
        print(self.T_atmo[0, :, iter_id+1])

        self.T_atmo[:,:,iter_id+1] = self.boundary_layer(self.T_atmo[:, :, iter_id])
        self.T_soil[:,:,iter_id+1] = self.heat_equation(self.T_soil[:, :, iter_id])


    def run(self):

        self._init_index_arrs()
        self._init_surface()

        for idx in range(self.n_iters-1):
            self.step(idx)

    def test_surf_balance(self):
        self._init_index_arrs()
        self._init_surface()
        print('starting temp: ' + str(self.T_atmo[0, :, 5]))
        print(self.surface_balance(self.T_atmo[0, :, 5]))


if __name__ == '__main__':
    Nx = 10
    Nz_atmo = 40
    Nz_soil = 20
    x_size = 10_000
    z_size_atmo = 10 * Nz_atmo
    z_size_soil = 20
    z = 2.0
    z_0 = 1e-3
    time = 3600 * 24 * 1/24 # run for an hour
    dt = 60

    model = Combined_model(Nx, Nz_atmo, Nz_soil, x_size, z_size_atmo, z_size_soil, z, z_0, time, dt)
    model.test_surf_balance()
