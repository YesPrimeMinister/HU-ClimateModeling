import math
import numpy as np
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
            print(f'H_0 = {rho} * {c_p} * {Cs_t} * {U_L} * ({T_0} - {T_a})')
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

            Q_0, L_d, L_u, H_0, E_0 = EB_fluxes(T_0,T_a,f,albedo,G,p,rho,U_L,Cs_t,Cs_q)

            # Get residual for optimization
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
                         method='Nelder-Mead',options={'eps':1e-8})

            self.surf_T_now[x] = res.x[0]

        return self.surf_T_now

    def test_surf_balance(self):
        self._init_index_arrs()
        self._init_surface()
        print('starting temp: ' + str(self.T_atmo[0, :, 5]))
        print(self.surface_balance(self.T_atmo[0, :, 5]))

class surf_test:
    def __init__(self, albedo):
        self.albedo = albedo

        self.c_p = 1004            # specific heat [J kg^-1 K^-1]
        self.kappa = 0.41          # Von Karman constant [-]
        self.sigma = 5.67e-8       # Stefan-Bolzmann constant
        self.L = 2.83e6            # latent heat for sublimation
        self.G = 700 * np.ones(20)

    def surface_balance(self, T_a):
        def E_sat(T):
            """ Saturation water vapor equation """
            Ew = 6.112 * np.exp((17.67*(T-273.16)) / ((T-29.66)))
            return Ew

        def EB_fluxes(T_0,T_a,f,albedo,G,p,rho,U_L):
            """ This function calculates the energy fluxes from following quantities:

            Input:
            T_0       : Surface temperature, which is optimized [K]
            f         : Relative humdity as fraction, e.g. 0.7 [-]
            albedo    : Snow albedo [-]
            G         : Shortwave radiation [W m^-2]
            p         : Air pressure [hPa]
            rho       : Air denisty [kg m^-3]
            z         : Measurement height [m]
            z_0       : Roughness length [m]

            """

                # Correction factor for incoming longwave radiation
            eps_cs = 0.23 + 0.433 * np.power(100*(f*E_sat(T_a))/T_a,1.0/8.0)

            # Select the appropriate latent heat constant
            L = 2.83e6 # latent heat for sublimation

            # Calculate turbulent fluxes
            H_0 = rho * self.c_p * Cs_t * U_L * (T_0 - T_a)
            #E_0 = rho * L*0.622/p * Cs_q * U_L * (E_sat(T_0) - f*E_sat(T_0))
            E_0 = rho * self.L*0.622/p * Cs_q * U_L * E_sat(T_0)*(1 - f)

            # Calculate radiation budget
            L_d = eps_cs * self.sigma * T_a**4
            L_u = 1.0 * self.sigma * T_0**4
            Q_0 = (1-albedo) * G

            return (Q_0, L_d, L_u, H_0, E_0)



        def optim_T0(x,T_a,f,albedo,G,p,rho,U_L):
            """ Optimization function for surface temperature:

            Input:
            T_0       : Surface temperature, which is optimized [K]
            f         : Relative humdity as fraction, e.g. 0.7 [-]
            albedo    : Snow albedo [-]
            G         : Shortwave radiation [W m^-2]
            p         : Air pressure [hPa]
            rho       : Air denisty [kg m^-3]
            z         : Measurement height [m]
            z_0       : Roughness length [m]

            """

            Q_0, L_d, L_u, H_0, E_0 = EB_fluxes(x,T_a,f,albedo,G,p,rho,U_L)

            # Get residual for optimization
            res = np.abs(Q_0+L_d-L_u-H_0-E_0)
            # return the residuals
            return res

        for i in range(20):
            test_args = (T_a,f,self.albedo,self.G[i],p,rho,U)
            res = minimize(optim_T0,x0=T_0,args=test_args,bounds=((None,400),), \
                           method='L-BFGS-B',options={'eps':1e-8})
            print(res.x[0])

        return res.x[0]


if __name__ == '__main__':

    # Test the SEB function
    # Define necessary variables and parameters
    kappa = 0.41          # Von Karman constant [-]

    T_0 = 283.0   # Surface temperature
    T_a = 280.0   # Air temperature
    f = 0.7       # Relative humidity
    albed = 0.3  # albedo
    rho = 1.1     # Air density
    U = 2.0       # Wind velocity
    z =  2.0      # Measurement height
    z0 = 1e-3     # Roughness length
    p = 1013      # Pressure

    # Bulk coefficients
    Cs_t = kappa**2 / (np.log(z/z0)**2)
    Cs_q = Cs_t

    model = surf_test(albed)
    print(model.surface_balance(T_a))
