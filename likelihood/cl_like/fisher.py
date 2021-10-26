import numpy as np
import yaml
import pyccl as ccl
from cl_like.theory_vector import posterior
from numpy.linalg import multi_dot
import numpy.linalg as LA 

class Fisher():
    
    def __init__ (self,input_forecast_yaml):
        self.a = posterior('forecast.yml')
        # Fitted Cosmology parameter
        self.parms = self.a.pars
        # theory part of ccl 
        self.TH = self.a.theory
        # inverse of date covariance
        self.data_invc = self.a.inv_cov


    # Function initialize ccl cosmo object to be passed into posterior calculation
    def utility_func_ccl(self,parms):
        '''
        Utility function that calls ccl cosmology and bias parameters
        Parameters:
        ----
        parms: dict
            A dictionary stores the updated parameter values to be calculated for derivative
        '''
        Cosmo = ccl.Cosmology(
            Omega_c=parms['Omega_c'], Omega_b=parms['Omega_b'], h=parms['h'], n_s=parms['n_s'],
            sigma8=parms['sigma8'],T_CMB=2.7255,m_nu=parms['m_nu'],
            transfer_function=self.TH['transfer_function'],
            matter_power_spectrum=self.TH['matter_pk'],baryons_power_spectrum=self.TH['baryons_pk'])
        Bias_parms = {}
        for p in parms:
            if self.a.input_params_prefix in p:
                Bias_parms[p] = parms[p]
        Theory = {}
        Theory['Cosmo'] = Cosmo
        Theory['bias_parms'] = Bias_parms
        return(Theory)

    def five_point_stencil_deriv(self,parms,par_name = 'sigma8',step_factor = 0.01):
        '''
        Calculating first derivative of the theory data vector with respect to 
        a single parameter based on five point stencil approximation
        Parameters:
        ----
        parms: dict
            A dictionry that stores fitted value for each parameter.
        par_name: string
            The parameter the derivative corresponds to.
        step_factor: float
            The factor that determine the step size of finite difference.
        verbose: bool
            For checks only.
        
        '''
        
        # typical variations of each parameter
        typ_var = {"sigma8": 0.1,"Omega_c": 0.5,"Omega_b": 0.2,"h": 0.5,"n_s": 0.2,"m_nu": 0.1,
               "clk_cl1_b1": 0.1,"clk_cl2_b1": 0.1,"clk_cl3_b1": 0.1,
               "clk_cl4_b1": 0.1,"clk_cl5_b1": 0.1,"clk_cl6_b1": 0.1, 
               "clk_cl1_b1p": 0.1,"clk_cl2_b1p": 0.1,"clk_cl3_b1p": 0.1,
               "clk_cl4_b1p": 0.1,"clk_cl5_b1p": 0.1,"clk_cl6_b1p": 0.1, 
               "clk_cl1_b2": 0.1,"clk_cl2_b2": 0.1,"clk_cl3_b2": 0.1,
               "clk_cl4_b2": 0.1,"clk_cl5_b2": 0.1,"clk_cl6_b2": 0.1, 
               "clk_cl1_bs": 0.1,"clk_cl2_bs": 0.1,"clk_cl3_bs": 0.1,
               "clk_cl4_bs": 0.1,"clk_cl5_bs": 0.1,"clk_cl6_bs": 0.1} 

        step_size = typ_var[par_name]*step_factor
        # -h
        left_params_1d = parms.copy()
        left_params_1d[par_name] = left_params_1d[par_name] - step_size
        # -2h
        left_params_2d = parms.copy()
        left_params_2d[par_name] = left_params_1d[par_name] - step_size

        # +h
        right_params_1d = parms.copy()
        right_params_1d[par_name] = right_params_1d[par_name] + step_size
        # +2h
        right_params_2d = parms.copy()
        right_params_2d[par_name] = right_params_1d[par_name] + step_size

        left_1d_cosmo = self.utility_func_ccl(parms=left_params_1d)
        right_1d_cosmo = self.utility_func_ccl(parms=right_params_1d)
        left_2d_cosmo = self.utility_func_ccl(parms=left_params_2d)
        right_2d_cosmo = self.utility_func_ccl(parms=right_params_2d)

        right_1d = self.a._get_theory(Theory = right_1d_cosmo)
        right_2d = self.a._get_theory(Theory = right_2d_cosmo)


        left_1d = self.a._get_theory(Theory = left_1d_cosmo)
        left_2d = self.a._get_theory(Theory = left_2d_cosmo)

        # Five stencil approximation for 1st derivative
        dcl_dparam = (left_2d - 8 * left_1d + 8 * right_1d - right_2d)/ (12 * step_size)
        return(dcl_dparam)


    def get_fisher(self, parameters,step_factor = 0.01):
        '''
        Function calculate the fisher matrix
        Parameters:
        ----
        parameters: list of string
            Fitted parameters counted in the fisher matrix.
        step_factor: float
            The factor that determine the step size of finite difference.
        verbose: bool
            For sanity check.
        '''
        # inverse of date covariance
        data_invc = self.data_invc
        
        F = np.empty([len(parameters),len(parameters)])
        dcl_dparm = {}
        for pars in parameters:
            dcl_dparm[pars] =  self.five_point_stencil_deriv(parms=self.parms,
                                                             par_name= pars,step_factor= step_factor)
        for i,par1 in enumerate(parameters):
            for j,par2 in enumerate(parameters):
                    de_par1 = dcl_dparm[par1]
                    de_par2 = dcl_dparm[par2]
                    F[i,j] =  multi_dot([de_par1,data_invc,de_par2])

        return(F)