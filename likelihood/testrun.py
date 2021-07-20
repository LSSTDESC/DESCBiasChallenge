from cobaya.model import get_model
from cobaya.run import run
import yaml
import os

import numpy as np
import numpy.linalg as LA 

# Read in the yaml file
config_fn = 'test.yml'
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

# Get the mean proposed in the yaml file for each parameter
p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']
os.system('mkdir -p ' + info['output'])

print("params_dict = ", p0)

# Compute the likelihood
model = get_model(info)
loglikes, derived = model.loglikes(p0)
print("chi2 = ", -2 * loglikes[0])

# Run minimizer
updated_info, sampler = run(info)
bf = sampler.products()['minimum'].bestfit()
pf = {k: bf[k] for k in p0.keys()}
print("Final params: ")
print(pf)

#======================DETERMINE ERRORS ON PARAMETERS========================

# remove cobaya_out directory (just for now!) to make running code easier
os.system('rm -r cobaya_out')  

class Fisher:
    
    def __init__(self,pf):
        self.pf = pf
    

    def fstep(self,param1,param2,h1,h2,signs):  # Determine likelihood at new steps 
        newp = self.pf.copy()
        newp[param1] = self.pf[param1] + signs[0]*h1
        newp[param2] = self.pf[param2] + signs[1]*h2
    
        newloglike = model.loglikes(newp)
    
        return -1*newloglike[0]

    def F_ij(self,param1,param2,h1,h2):  # Hessian matrix elements
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

    def get_err(self):
        h_fact = 0.005  # stepsize factor

        # typical variations of each parameter
        typ_var = {"sigma8": 1,"Omega_c": 0.4,"Omega_b": 1,"h": 1,"n_s": 1,"m_nu": 1}  

        theta = list(self.pf.keys())  # array containing parameter names

        # Calculate Hessian matrix
        F = np.empty((len(theta),len(theta)))
        for i in range(0,len(theta)):
            for j in range(0,len(theta)):
                param1 = theta[i]
                param2 = theta[j]
                h1 = h_fact*typ_var[param1]
                h2 = h_fact*typ_var[param2]
                F[i][j] = self.F_ij(param1,param2,h1,h2)
                
        covar = LA.inv(F)  # covariance matrix
        err = np.sqrt(np.diag(covar))  # estimated parameter errors
        return err


final_params = Fisher(pf)
print('ERRORS: ',final_params.get_err())            

