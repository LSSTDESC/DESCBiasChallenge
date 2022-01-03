import numpy as np
from numpy.linalg import multi_dot



class Fisher_first_deri():
    
    def __init__ (self,model,parms,fp_name,method = 'five-stencil'):
        '''
        This Class implement first derivative expression fisher matrix calculation for error estimation
        Parameters:
        ----
        model: cobaya.model.Model object
           Cobaya model class
        parms: Dict
            Parameters needed to calculate theory vector
        fp_name: List
           Sampled parameter name
        method: String
           Differentiation method. 
           Default is set to 'five-stencil': Five stencil approximation for 1st derivative
           (Note: The fisher sampler implemented in Cosmosis by Zuntz et al.(2015) uses five stencil approximation)
           The following alternatives are also avaliable:
           'three-stencil': Three stencil approximation for 1st derivative
           'seven-stencil': Seven stencil approximation for 1st derivative
           (see Fornberg, B. 1988, Mathematics of computation, 51, 699 for detailed description)
            
        '''
        self.method = method
        self.model = model
        self.ClLike = self.model.likelihood['cl_like.ClLike']
        
        # Sampled parameter name
        self.parms_name = fp_name
        
        # Fitting values of Sampled parameter 
        self.parms = parms

        # inverse of date covariance
        self.data_invc = self.ClLike.inv_cov
        
        # inv_cov check
        if not np.allclose(self.data_invc, self.data_invc.T):
            print("WARNING: The inverse covariance matrix is not symmetric.")
    
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
        
        # hard-coded typical variations of cosmology
        typ_var = {"sigma8": 0.1,"Omega_c":0.5,"Omega_b": 0.2,"h": 0.5,"n_s": 0.2,"m_nu": 0.1}
        
        # step_size for differentiation
        if pref in par_name:
            # step_size for bias parameters with 0.1 typical variation 
            step_size = 0.1*step_factor
        else:
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
        step_factor: float
            The factor that determine the step size of finite difference. Default is set to 0.01
        '''
        dcl_dparm = []
        for pars in self.parms_name:
            dcl_dparm.append(self.n_point_stencil_deriv(par_name= pars,step_factor= step_factor))
        F = np.einsum("il,lk,jk->ij", dcl_dparm, self.data_invc, dcl_dparm)
        return(F)
    
    
class Fisher_second_deri():

    def __init__(self,model,pf,h):
        '''
        This class implement the 2nd derivative fisher matrix calculation written by Nathan Findlay.
        Parameters:
        ----
        model: cobaya.model.Model object
           Cobaya model class
        pf: Dict
           Sampled parameters 
        h: step size factor
        '''
        self.model = model
        self.pf = pf
        self.h_fact = h

    # Determine likelihood at new steps
    def fstep(self,param1,param2,h1,h2,signs):
        newp = self.pf.copy()
        newp[param1] = self.pf[param1] + signs[0]*h1
        newp[param2] = self.pf[param2] + signs[1]*h2

        newloglike = self.model.loglikes(newp)

        return -1*newloglike[0]

    # Fisher matrix elements
    def F_ij(self,param1,param2,h1,h2):
        # Diagonal elements
        if param1==param2:
            f1 = self.fstep(param1,param2,h1,h2,(0,+1))
            f2 = self.fstep(param1,param2,h1,h2,(0,0))
            f3 = self.fstep(param1,param2,h1,h2,(0,-1))
            F_ij = (f1-2*f2+f3)/(h2**2)
        # Off-diagonal elements
        else:
            f1 = self.fstep(param1,param2,h1,h2,(+1,+1))
            f2 = self.fstep(param1,param2,h1,h2,(-1,+1))
            f3 = self.fstep(param1,param2,h1,h2,(+1,-1))
            f4 = self.fstep(param1,param2,h1,h2,(-1,-1))
            F_ij = (f1-f2-f3+f4)/(4*h1*h2)

        return F_ij[0]

    # Calculate Fisher matrix
    def calc_Fisher(self):

        # typical variations of each parameter
        pref = self.model.likelihood['cl_like.ClLike'].input_params_prefix
        
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

        theta = list(self.pf.keys())  # array containing parameter names

        # Assign matrix elements
        F = np.empty([len(theta),len(theta)])
        for i in range(0,len(theta)):
            for j in range(0,len(theta)):
                param1 = theta[i]
                param2 = theta[j]
                h1 = self.h_fact*typ_var[param1]
                h2 = self.h_fact*typ_var[param2]
                F[i][j] = self.F_ij(param1,param2,h1,h2)

        return F

    # Get errors on parameters
    def get_err(self):
        covar = LA.inv(self.calc_Fisher())  # covariance matrix
        err = np.sqrt(np.diag(covar))  # estimated parameter errors
        return err