from cobaya.model import get_model
from cobaya.run import run
import yaml
import os

import numpy as np
import numpy.linalg as LA 
import numdifftools as nd

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

h = 0.005  # stepsize

def fstep(params,signs):  # Determine likelihood at new steps 
    newp0 = pf.copy()
    newp0[params[0]] = pf[params[0]] + signs[0]*h
    newp0[params[1]] = pf[params[1]] + signs[1]*h
    
    newloglike = model.loglikes(newp0)
    
    return -1*newloglike[0]

def F_ij(params):  # Hessian matrix elements
    # Diagonal elements
    if params[0]==params[1]:  
        f1 = fstep(params,(0,+1))
        f2 = fstep(params,(0,0))
        f3 = fstep(params,(0,-1))
        F_ij = (f1-2*f2+f3)/(h**2)
        
    # Off-diagonal elements     
    else:  
        f1 = fstep(params,(+1,+1))
        f2 = fstep(params,(-1,+1))
        f3 = fstep(params,(+1,-1))
        f4 = fstep(params,(-1,-1))
        F_ij = (f1-f2-f3+f4)/(4*h**2)
          
    return F_ij[0]

theta = list(pf.keys())  # array containing parameter names

# Calculate Hessian matrix
F = np.zeros((len(theta),len(theta)))
for i in range(0,len(theta)):
    for j in range(0,len(theta)):
        F[i][j] = F_ij((theta[i],theta[j]))
       
covar = LA.inv(F)  # covariance matrix
err=np.sqrt(np.diag(covar))  # estimated parameter errors
print('ERRORS: ',err)            


#========ATTEMPT AT USING ND.HESSIAN INSTEAD (IGNORE!)====================
'''
par_name = list(pf.keys())

L = lambda theta : model.loglikes(dict(list(zip(par_name,theta))))[0][0]

Hess = nd.Hessian(L)
pf_arr = list(pf.values())
print('pfarr',pf_arr)
A = Hess(pf_arr)

print('F1',F)
print('F2',A)
'''

