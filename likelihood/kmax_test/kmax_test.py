from cobaya.model import get_model
from cobaya.run import run
from shutil import copyfile
import yaml
import os
import sys
sys.path.append('../likelihood')
import numpy as np
import numpy.linalg as LA
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run NL bias model..')

parser.add_argument('--path2defaultconfig', dest='path2defaultconfig', type=str, help='Path to default config file.', \
                    required=True)
parser.add_argument('--path2data', dest='path2data', type=str, help='Path to data sacc.', required=True)
parser.add_argument('--path2output', dest='path2output', type=str, help='Path to output directory.', required=True)
parser.add_argument('--bias_model', dest='bias_model', type=str, help='Name of bias model.', required=True)
parser.add_argument('--k_max', dest='k_max', type=float, help='Maximal k vector.', required=True)
parser.add_argument('--fit_params', dest='fit_params', nargs='+', help='Parameters to be fit.', required=True)
parser.add_argument('--bins', dest='bins', nargs='+', help='Redshift bins to be fit.', required=True)
parser.add_argument('--probes', dest='probes', nargs='+', help='Probes to be fit.', required=True)
parser.add_argument('--sigma8', dest='sigma8', type=float, help='Fixed parameter value.', required=False)
parser.add_argument('--Omega_c', dest='Omega_c', type=float, help='Fixed parameter value.', required=False)
parser.add_argument('--Omega_b', dest='Omega_b', type=float, help='Fixed parameter value.', required=False)
parser.add_argument('--h', dest='h', type=float, help='Fixed parameter value.', required=False)
parser.add_argument('--n_s', dest='n_s', type=float, help='Fixed parameter value.', required=False)

args = parser.parse_args()

path2output = args.path2output

bias_model = args.bias_model
k_max = args.k_max
fit_params = args.fit_params

# Set model
if bias_model == 'lin':
    model = 'Linear'
elif bias_model == 'EuPT' or bias_model == '3EuPT':
    model = 'EulerianPT'
elif bias_model == 'LPT':
    model = 'LagrangianPT'
else:
    raise ValueError("Unknown bias model")

logger.info('Running analysis for:')
logger.info('Bias model: {}.'.format(bias_model))
logger.info('k_max: {}.'.format(k_max))

# Read in the yaml file
config_fn = args.path2defaultconfig
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

# Determine true bias parameters depending on input
bias = [2., 2., 2., 2., 2., 2.]

if 'sigma8' in fit_params:
    info['params']['sigma8'] = {'prior': {'min': 0.1, 'max': 1.2},
                                                    'ref': {'dist': 'norm', 'loc': 0.8090212289405192, 'scale': 0.01},
                                                    'latex': '\sigma_8', 'proposal': 0.001}
elif args.sigma8 is not None:
    info['params']['sigma8'] = args.sigma8
else:
    info['params']['sigma8'] = 0.8090212289405192
if 'Omega_c' in fit_params:
    info['params']['Omega_c'] = {'prior': {'min': 0.05, 'max': 0.7},
                                                    'ref': {'dist': 'norm', 'loc': 0.26447041034523616, 'scale': 0.01},
                                                    'latex': '\Omega_c', 'proposal': 0.001}
elif args.Omega_c is not None:
    info['params']['Omega_c'] = args.Omega_c
else:
    info['params']['Omega_c'] = 0.26447041034523616
if 'Omega_b' in fit_params:
    info['params']['Omega_b'] = {'prior': {'min': 0.01, 'max': 0.2},
                                                    'ref': {'dist': 'norm', 'loc': 0.049301692328524445, 'scale': 0.01},
                                                    'latex': '\Omega_b', 'proposal': 0.001}
elif args.Omega_b is not None:
    info['params']['Omega_b'] = args.Omega_b
else:
    info['params']['Omega_b'] = 0.049301692328524445
if 'h' in fit_params:
    info['params']['h'] = {'prior': {'min': 0.1, 'max': 1.2},
                                                    'ref': {'dist': 'norm', 'loc': 0.6736, 'scale': 0.01},
                                                    'latex': 'h', 'proposal': 0.001}
elif args.h is not None:
    info['params']['h'] = args.h
else:
    info['params']['h'] = 0.6736
if 'n_s' in fit_params:
    info['params']['n_s'] = {'prior': {'min': 0.1, 'max': 1.2},
                                                    'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.01},
                                                    'latex': 'n_s', 'proposal': 0.001}
elif args.n_s is not None:
    info['params']['n_s'] = args.n_s
else:
    info['params']['n_s'] = 0.9649

probes = args.probes
info['likelihood']['cl_like.ClLike']['bins'] = [{'name': bin_name} for bin_name in args.bins]
info['likelihood']['cl_like.ClLike']['twopoints'] = [{'bins': [probes[2*i], probes[2*i+1]]} for i in range(len(probes)//2)]
n_bin = len(args.bins)
bin_nos = [int(bin_name[-1])-1 for bin_name in args.bins if 'cl' in bin_name]

# Template for bias parameters in yaml file
cl_param = {'prior': {'min': -100.0, 'max': 100.0}, 
        'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.01}, 
        'latex': 'blank', 'proposal': 0.001}

# Set bias parameter types used in each model
if bias_model in ['EuPT','LPT']:
    bpar = ['1','1p','2','s']
elif bias_model == '3EuPT':
    bpar = ['1', '1p', '2', 's', '3nl', 'k2']
else:
    bpar = ['1','1p']
    
# Write bias parameters into yaml file
for b in bpar:
    for i in bin_nos:
        param_name = 'cllike_cl'+str(i+1)+'_b'+b
        if param_name in fit_params:
            info['params'][param_name] = cl_param.copy()
            info['params'][param_name]['latex'] = 'b_'+b+'\\,\\text{for}\\,C_{l,'+str(i+1)+'}'
            if b == '0' or b == '1':
                info['params']['cllike_cl'+str(i+1)+'_b'+b]['ref'] = {'dist': 'norm', 'loc': bias[i], 'scale': 0.01}
        else:
            if b == '0' or b == '1':
                info['params']['cllike_cl'+str(i+1)+'_b'+b] = bias[i]
            else:
                info['params']['cllike_cl' + str(i + 1) + '_b' + b] = 0.

# Add model and input file
info['likelihood']['cl_like.ClLike']['bz_model'] = model
info['likelihood']['cl_like.ClLike']['input_file'] = args.path2data

# Add kmax and output file
info['likelihood']['cl_like.ClLike']['defaults']['kmax'] = float(k_max)
info['output'] = path2output

# Save yaml file
dir, _ = os.path.split(path2output)
os.makedirs(dir, exist_ok=True)
with open(path2output+'.yml', 'w') as yaml_file:
    yaml.dump(info, yaml_file, default_flow_style=False)

# Get the mean proposed in the yaml file for each parameter
p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']

print("params_dict = ", p0)

# Run minimizer
updated_info, sampler = run(info)
bf = sampler.products()['minimum']
np.save(info['output']+'.hessian.npy', sampler.products()['result_object'].hessian)
pf = {k: bf[k] for k in p0.keys()}
print("Final params: ")
print(pf)

# Compute the likelihoods
model = get_model(info)
loglikes, derived = model.loglikes(p0)
p0_chi2 = -2 * loglikes[0]
loglikes, derived = model.loglikes(pf)
pf_chi2 = -2 * loglikes[0]

# Determine errors on parameters
class Fisher:
    def __init__(self,pf):
        self.pf = pf
    
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
        h_fact = 0.01 # stepsize factor

        # typical variations of each parameter
        typ_var = {"sigma8": 0.1,"Omega_c": 0.5,"Omega_b": 0.2,"h": 0.5,"n_s": 0.2,"m_nu": 0.1,
                   "cllike_cl1_b1": 0.1,"cllike_cl2_b1": 0.1,"cllike_cl3_b1": 0.1,
                   "cllike_cl4_b1": 0.1,"cllike_cl5_b1": 0.1,"cllike_cl6_b1": 0.1, 
                   "cllike_cl1_b1p": 0.1,"cllike_cl2_b1p": 0.1,"cllike_cl3_b1p": 0.1,
                   "cllike_cl4_b1p": 0.1,"cllike_cl5_b1p": 0.1,"cllike_cl6_b1p": 0.1, 
                   "cllike_cl1_b2": 0.1,"cllike_cl2_b2": 0.1,"cllike_cl3_b2": 0.1,
                   "cllike_cl4_b2": 0.1,"cllike_cl5_b2": 0.1,"cllike_cl6_b2": 0.1, 
                   "cllike_cl1_bs": 0.1,"cllike_cl2_bs": 0.1,"cllike_cl3_bs": 0.1,
                   "cllike_cl4_bs": 0.1,"cllike_cl5_bs": 0.1,"cllike_cl6_bs": 0.1} 

        theta = list(self.pf.keys())  # array containing parameter names

        # Assign matrix elements
        F = np.zeros([len(theta),len(theta)])
        for i in range(0,len(theta)):
            for j in range(0,len(theta)):
                param1 = theta[i]
                param2 = theta[j]
                h1 = h_fact*typ_var[param1]
                h2 = h_fact*typ_var[param2]
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

    # Get errors on parameters
    def get_cov(self):
        covar = LA.inv(self.calc_Fisher())  # covariance matrix
        return covar

final_params = Fisher(pf)
# errs = list(final_params.get_err())
p0vals = list(p0.values())
pfvals = list(pf.values())
cov = list(final_params.get_cov())

# Save data to file
np.savez(info['output']+'.fisher.npz', bf=pfvals, truth=p0vals, chi2_bf=pf_chi2, chi2_truth=p0_chi2, cov=cov)