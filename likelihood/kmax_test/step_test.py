#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:21:13 2021

@author: nathanfindlay
"""
from cobaya.model import get_model
from cobaya.run import run
import yaml
import os

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import csv

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
    def __init__(self,pf,h):
        self.pf = pf
        self.h_fact = h
    
    # Determine likelihood at new steps
    def fstep(self,param1,param2,h1,h2,signs):   
        newp = self.pf.copy()
        newp[param1] = self.pf[param1] + signs[0]*h1
        newp[param2] = self.pf[param2] + signs[1]*h2
    
        newloglike = model.loglikes(newp)
    
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
        #h_fact =  0.5 # stepsize factor
        #h_fact = array[indx]

        # typical variations of each parameter
        typ_var = {"sigma8": 0.1,"Omega_c": 0.5,"Omega_b": 0.2,"h": 0.5,"n_s": 0.2,"m_nu": 0.1}  

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
    
    # Determine condition number of Fisher matrix
    def get_cond_num(self):
        cond_num = LA.cond(self.calc_Fisher())
        return cond_num
        
    # Get errors on parameters
    def get_err(self):
        covar = LA.inv(self.calc_Fisher())  # covariance matrix
        err = np.sqrt(np.diag(covar))  # estimated parameter errors
        return err

step = np.geomspace(1e-4,0.3,20)
errs = np.zeros((len(step),len(pf)))
C = np.zeros(len(step))

for i in range(0,len(step)):
    h = step[i]
    final_params = Fisher(pf,h)
    errs[i] = final_params.get_err()
    C[i] = final_params.get_cond_num()

col=['b','r','g','c','m','y']
f1, axes = plt.subplots(2,1,sharex='all',figsize=(5,6))
f1.tight_layout()
f1.show()

axes[0].set_title('Parameter errors')
for i in range(0,len(pf)):  
    axes[0].plot(step,errs[:,i],color=col[i],label=list(pf.keys())[i])
axes[0].set_ylabel('Value')
axes[0].set_xscale('log')
#axes[0].set_ylim((3e-4,8e-4))
axes[0].legend()

axes[1].set_title('Fisher matrix condition number')
axes[1].plot(step,C)
axes[1].set_xscale('log')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Stepsize factor (h_fact)')

