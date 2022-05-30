import numpy as np
import numpy.linalg as LA
from numpy.linalg import multi_dot



class Fisher_first_deri():
    
    def __init__ (self,model,parms,fp_name,step_factor = 0.01,method = 'five-stencil',full_expresssion = False):
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
        step_factor: float
            The factor that determine the step size of finite difference. Default is set to 0.01
        method: String
           Differentiation method. 
           Default is set to 'five-stencil': Five stencil approximation for 1st derivative
           The following alternatives are also avaliable: 'three-stencil', 'seven-stencil'
           (see Fornberg, B. 1988, Mathematics of computation, 51, 699 for detailed description)
        full_expression: bool 
            False -- Default fisher expression for gaussian likelihood: 
                F_ij = dcl/dpar_i.T * C^{-1} * dcl/dpar_j
            True -- With the extra term (usually expected to be averaged out) added to the default expression:
                F_ij = -(d-t).T * C^{-1} * d^2cl/dpar_i dpar_j + dcl/dpar_i.T * C^{-1} * dcl/dpar_j 
        '''
        self.method = method
        self.step_factor = step_factor
        self.model = model
        self.full_expresssion = full_expresssion
        self.ClLike = self.model.likelihood['cl_like.ClLike']
        
        # Sampled parameter name
        self.parms_name = fp_name
        
        # Fitting values of Sampled parameter 
        self.parms = parms

        # inverse of date covariance
        self.data_invc = np.linalg.inv(self.ClLike.cov)
        
        # inv_cov check
        if not np.allclose(self.data_invc, self.data_invc.T):
            print("WARNING: The inverse covariance matrix is not symmetric.")
            
    def get_r (self):
        '''
        Get data-theory vector
        '''
        params = self.parms.copy()
        #update model
        self.model.loglikes(params)
        theory_v = self.ClLike._get_theory(**params)
        r = self.ClLike.data_vec - theory_v
        return(r)
    
    def update (self,par_name,stencil,step_size):
        '''
        Calculating theory vector with varying values of a given sampled parameter
        based on stencil and step_size for 1st derivative differentiation
        Parameters:
        --------
        par_name: string
            The parameter the derivative corresponds to.
        step_size: float
            The step size of finite difference.
        stencil: int
            The stencil at which the theory vector is evaluated for differentiation
        '''
        
        # update parameters based on step size
        params = self.parms.copy()
        params[par_name] = params[par_name]+ stencil * step_size
        #update model
        self.model.loglikes(params)
        theory_v = self.ClLike._get_theory(**params)
        return(theory_v)

    def update_2ndd (self,par_name1,par_name2,stencil,step_size):
        '''
        Calculating theory vector with varying values of two sampled parameters
        based on corresponding stencil and step_size for 2nd derivative differentiation
        '''
        # update parameters based on step size
        params = self.parms.copy()
        params[par_name1] = params[par_name1]+ stencil[0] * step_size[0]
        params[par_name2] = params[par_name2]+ stencil[1] * step_size[1]
        #update model
        self.model.loglikes(params)
        theory_v = self.ClLike._get_theory(**params)
        return(theory_v)
    
    def n_point_stencil_deriv(self,par_name = 'sigma8',step_factor = 0.01):
        '''
        Calculating first derivative of the theory data vector with respect to 
        a single parameter based on selected stencil approximation method.
        Parameters:
        ----
        par_name: string
            The parameter the derivative corresponds to.
        step_factor: float
            The factor that determine the step size of finite difference.
        '''

        # step_size for differentiation
        if self.parms[par_name] != 0.:
            # step_size for nonzero parameters
            step_size = self.parms[par_name]*step_factor
        else:
            step_size = step_factor
        
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

    
    def second_deri(self,par_pair,step_factor = 0.01):
        '''
        Calculating 2nd derivative of the theory data vector with respect to two parameters 
        Equation used to calculate Diagonal elements: https://en.wikipedia.org/wiki/Five-point_stencil
        Equation used to calculate Off-diagonal elements: https://en.wikipedia.org/wiki/Finite_difference
        
        Parameters:
        ----
        par_pair: list of string
            The pair of parameters to calculate 2nd derivative
        step_factor: float
            The factor that determine the step size of finite difference.
        '''
        step_size = []
        
        pref = self.ClLike.input_params_prefix
        # hard-coded typical variations of cosmology
        typ_var = {"sigma8": 0.1,"Omega_c":0.5,"Omega_b": 0.2,"h": 0.5,"n_s": 0.2,"m_nu": 0.1}
        for k in range(2):
            # step_size for differentiation
            if pref in par_pair[k]:
                # step_size for bias parameters with 0.1 typical variation 
                step_size.append(0.1*step_factor)
            else:
                step_size.append(typ_var[par_pair[k]]*step_factor)
        dcl_dp1p2 = np.zeros(shape = len(self.data_invc))
        
        if par_pair[0] != par_pair[1]:
            
            stencils = [[1,1],[1,-1],[-1,1],[-1,-1]]
            stencil_wgt = [1,-1,-1,1]
            
            for sten,wgt in zip(stencils,stencil_wgt):
                dcl_dp1p2 += wgt * self.update_2ndd(par_name1 = par_pair[0], par_name2 = par_pair[1], stencil=sten, step_size= step_size)
                
            dcl_dp1p2 = dcl_dp1p2 / (4* step_size[0]*step_size[1])
            return(dcl_dp1p2)
        
        else:
            # Five point stencils
            stencils = [-2,-1,0,1,2]
            # weight for each stencil
            stencil_wgt = [-1,16,-30,16,-1]
            # stencil factor
            stencil_factor = 12
            
            for sten,wgt in zip(stencils,stencil_wgt):
                dcl_dp1p2 += wgt * self.update(par_name=par_pair[0], stencil=sten, step_size= step_size[0])
            
            dcl_dp1p2 = dcl_dp1p2 / (stencil_factor * step_size[0]**2)
            
            return(dcl_dp1p2)
    
    def get_fisher(self):
        '''
        Function calculate the fisher matrix
        '''
        if self.full_expresssion == False:
            
            dcl_dparm = []
            for pars in self.parms_name:
                dcl_dparm.append(self.n_point_stencil_deriv(par_name= pars,step_factor= self.step_factor))
            F = np.einsum("il,lk,jk->ij", dcl_dparm, self.data_invc, dcl_dparm)
            
            return(F)
        
        else:
            F = np.empty([len(self.parms_name),len(self.parms_name)])
            r = self.get_r()
            dcl_dparm = {}
            dcl_dp1p2 = {}
            for pars in self.parms_name:
                dcl_dparm[pars] =  self.n_point_stencil_deriv(par_name= pars,step_factor= self.step_factor)
            for i,par1 in enumerate(self.parms_name):
                for j,par2 in enumerate(self.parms_name):
                        de_par1 = dcl_dparm[par1]
                        de_par2 = dcl_dparm[par2]
                        if (j,i) in dcl_dp1p2:
                            dcl_dp1p2[i,j] = dcl_dp1p2[j,i]
                        else:
                            dcl_dp1p2[i,j] = self.second_deri(par_pair = [par1,par2],step_factor = self.step_factor)
                            
                        F[i,j] =  multi_dot([de_par1.T,self.data_invc,de_par2]) - multi_dot([r.T,self.data_invc,dcl_dp1p2[i,j]])               
            return(F)
        
    def get_err(self):
        '''
        Get errors on parameters
        '''
        covar = LA.inv(self.get_fisher())  # covariance matrix
        err = np.sqrt(np.diag(covar))  # estimated parameter errors
        return err

    def get_cov(self):
        '''
        Get covariance
        '''

        fisher = self.get_fisher()
        covar = LA.inv(fisher)  # covariance matrix

        return covar, fisher
    
    
    
class Fisher_second_deri():

    def __init__(self,model,pf,pf_name,h):
        '''
        This class implement the 2nd derivative fisher matrix calculation originally written by Nathan Findlay.
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
        self.pf_name = pf_name

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
               pref + "_cl1_b2p": 0.1, pref + "_cl2_b2p": 0.1, pref + "_cl3_b2p": 0.1,
               pref + "_cl4_b2p": 0.1, pref + "_cl5_b2p": 0.1, pref + "_cl6_b2p": 0.1,
               pref+"_cl1_bs": 0.1,pref+"_cl2_bs": 0.1,pref+"_cl3_bs": 0.1,
               pref+"_cl4_bs": 0.1,pref+"_cl5_bs": 0.1,pref+"_cl6_bs": 0.1,
               pref + "_cl1_bk2": 0.1, pref + "_cl2_bk2": 0.1, pref + "_cl3_bk2": 0.1,
               pref + "_cl4_bk2": 0.1, pref + "_cl5_bk2": 0.1, pref + "_cl6_bk2": 0.1,
               pref + "_cl1_bsn": 0.1, pref + "_cl2_bsn": 0.1, pref + "_cl3_bsn": 0.1,
               pref + "_cl4_bsn": 0.1, pref + "_cl5_bsn": 0.1, pref + "_cl6_bsn": 0.1,
               pref + "_cl1xcl2_bsnx": 0.1, pref + "_cl1xcl3_bsnx": 0.1, pref + "_cl1xcl4_bsnx": 0.1,
               pref + "_cl1xcl5_bsnx": 0.1, pref + "_cl1xcl6_bsnx": 0.1,
               pref + "_cl2xcl3_bsnx": 0.1, pref + "_cl2xcl4_bsnx": 0.1, pref + "_cl2xcl5_bsnx": 0.1,
               pref + "_cl2xcl6_bsnx": 0.1,
               pref + "_cl3xcl4_bsnx": 0.1, pref + "_cl3xcl5_bsnx": 0.1, pref + "_cl3xcl6_bsnx": 0.1,
               pref + "_cl4xcl5_bsnx": 0.1, pref + "_cl4xcl6_bsnx": 0.1,
               pref + "_cl5xcl6_bsnx": 0.1,
               pref + "_hod_lMmin_0": 0.1, pref + "_hod_lMmin_p": 0.1,
               pref + "_hod_siglM_0": 0.1, pref + "_hod_siglM_p": 0.1,
               pref + "_hod_lM0_0": 0.1, pref + "_hod_lM0_p": 0.1,
               pref + "_hod_lM1_0": 0.1, pref + "_hod_lM1_p": 0.1,
               pref + "_hod_alpha_0": 0.1, pref + "_hod_alpha_p": 0.1,
               pref + "_hod_alpha_HMCODE": 0.1, pref + "_hod_k_supress": 0.1
                   }

        for param in ['_lMmin_0', '_lMmin_p', '_siglM_0', '_siglM_p', '_lM0_0', '_lM0_p', '_lM1_0', '_lM1_p', \
                        '_alpha_0', '_alpha_p']:
            for key in ['_cl1', '_cl2', '_cl3', '_cl4', '_cl5', '_cl6']:
                typ_var[pref+key+param] = 0.1

        #theta = list(self.pf.keys())  # array containing parameter names
        theta = self.pf_name

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

    def get_cov(self):
        '''
        Get covariance
        '''

        fisher = self.calc_Fisher()
        covar = LA.inv(fisher)  # covariance matrix

        return covar, fisher