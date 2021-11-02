import numpy as np
from numpy.linalg import multi_dot



class Fisher_first_deri():
    
    def __init__ (self,model,fitted_parms,method = 'five_stencil'):
        '''
        This Class implement the fisher matrix calculation based on first derivative expression
        Parameters:
        ----
        model: cobaya.model.Model object
           Cobaya model class
        fitted_parms: Dict
           Sampled parameters
        method: String
           Differentiation method. 
           Default is set to 'five_stencil': Five stencil approximation for 1st derivative
           (Note: The fisher sampler implemented in Cosmosis by Zuntz et al.(2015) uses five stencil approximation)
           The following alternatives are also avaliable:
           'three_stencil': Three stencil approximation for 1st derivative
           'seven_stencil': Seven stencil approximation for 1st derivative
           (see Fornberg, B. 1988, Mathematics of computation, 51, 699 for detailed description)
            
        '''
        self.method = method
        self.model = model
        self.ClLike = self.model.likelihood['cl_like.ClLike']
        
        # Sampled parameter name
        self.parms_name = [p for p in fitted_parms]
        
        # Fitting values of Sampled parameter 
        self.parms = fitted_parms

        # inverse of date covariance
        self.data_invc = self.ClLike.inv_cov
    
    def update (self,par_name,stencil,step_size):
        '''
        Calculating theory vector with varying values of a given sampled parameter
        based on stencil and step_size for differentiation
        Parameters:
        --------
        par_name: string
            The parameter the derivative corresponds to.
        step_factor: float
            The factor that determine the step size of finite difference.
        stencil: int
            The stencil at which the theory vector is evaluated for differentiation
            (See Fornberg, B. 1988, Mathematics of computation, 51, 699 for details)
        '''
        
        # update parameters based on step size
        params = self.parms.copy()
        params[par_name] = params[par_name]+ stencil * step_size
        
        self.model.loglikes(params)
        theory_v = self.ClLike._get_theory(**params)
        return(theory_v)
    
    def n_point_stencil_deriv(self,par_name = 'sigma8',step_factor = 0.01):
        '''
        Calculating first derivative of the theory data vector with respect to 
        a single parameter based on selected stencil approximation
        Parameters:
        ----
        parms: dict
            A dictionry that stores fitted value for each parameter.
        par_name: string
            The parameter the derivative corresponds to.
        step_factor: float
            The factor that determine the step size of finite difference.
        '''
        
        # typical variations of each parameter
        pref = self.ClLike.input_params_prefix
        
        # hard-coded typical variations of cosmology and bias parameters
        typ_var = {"sigma8": 0.1,"Omega_c": 0.5,"Omega_b": 0.2,"h": 0.5,"n_s": 0.2,"m_nu": 0.1,
               pref+ "_cl1_b1": 0.1, pref+"_cl2_b1": 0.1,pref+"_cl3_b1": 0.1,
               pref+"_cl4_b1": 0.1,pref+"_cl5_b1": 0.1,pref+"_cl6_b1": 0.1, 
               pref+"_cl1_b1p": 0.1,pref+"_cl2_b1p": 0.1,pref+"_cl3_b1p": 0.1,
               pref+"_cl4_b1p": 0.1,pref+"_cl5_b1p": 0.1,pref+"_cl6_b1p": 0.1, 
               pref+"_cl1_b2": 0.1,pref+"_cl2_b2": 0.1,pref+"_cl3_b2": 0.1,
               pref+"_cl4_b2": 0.1,pref+"_cl5_b2": 0.1,pref+"_cl6_b2": 0.1, 
               pref+"_cl1_bs": 0.1,pref+"_cl2_bs": 0.1,pref+"_cl3_bs": 0.1,
               pref+"_cl4_bs": 0.1,pref+"_cl5_bs": 0.1,pref+"_cl6_bs": 0.1} 
        
        # step_size for differentiation
        step_size = typ_var[par_name]*step_factor
        
        if self.method == 'five-stencil':
            # Five point stencils
            stencils = [-2,-1,1,2]
            # weight for each stencil
            stencil_wgt = [1,-8,8,-1]
            # stencil factor
            stencil_factor = 12
            
        elif self.method == 'three-stencil':
            stencils = [-1,1]
            stencil_wgt = [-1,1]
            stencil_factor = 2   
        
        elif self.method == 'seven-stencil':
            stencils = [-3,-2,-1,1,2,3]
            stencil_wgt = [-1,9,-45,45,-9,1]
            stencil_factor = 60  
        else:
            raise ValueError('Unknown differentiation method. Avaliable methods: three-stencil, five-stencil, seven-stencil')
            
        # first derivative of theory vector with respect to the sampled parameter
        dcl_dparam = np.zeros(shape = len(self.data_invc))
        
        for sten,wgt in zip(stencils,stencil_wgt):
            dcl_dparam += wgt * self.update(par_name=par_name, stencil=sten, step_size= step_size)
            
        dcl_dparam = dcl_dparam / (stencil_factor * step_size)
        
        return(dcl_dparam)
    
    def get_fisher(self,step_factor = 0.01):
        '''
        Function calculate the fisher matrix
        Parameters:
        ----
        parameters: list of string
            Fitted parameters counted in the fisher matrix.
        step_factor: float
            The factor that determine the step size of finite difference.
        '''
        
        F = np.empty([len(self.parms_name),len(self.parms_name)])
        dcl_dparm = {}
        for pars in self.parms_name:
            dcl_dparm[pars] =  self.n_point_stencil_deriv(par_name= pars,step_factor= step_factor)
        for i,par1 in enumerate(self.parms_name):
            for j,par2 in enumerate(self.parms_name):
                    de_par1 = dcl_dparm[par1]
                    de_par2 = dcl_dparm[par2]
                    F[i,j] =  multi_dot([de_par1.T,self.data_invc,de_par2])
        return(F)