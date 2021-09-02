from cobaya.model import get_model
from cobaya.run import run
from shutil import copyfile
import yaml
import os
import sys
import numpy as np
import numpy.linalg as LA 

model_name = sys.argv[1]
input_name = sys.argv[2]
kmax = sys.argv[3]

if model_name == 'LIN':
    model = 'Linear'
elif model_name == 'EPT':
    model = 'EulerianPT'
elif model_name == 'LPT':
    model = 'LagrangianPT'
else:
    raise ValueError("Unknown bias model")

if input_name[-6:] == 'abacus':
    input_file = '../../data/abacus_' + str(input_name) + '.fits'
    config_fn = "kmax_abacus.yml"
else:
    input_file = '../../data/fid_' + str(input_name) + '.fits'
    config_fn = "kmax_test.yml"

filename = model + '-' + input_name
print('FILENAME: ',filename)
print('MODEL: '+model, '| INPUT: '+input_file, '| KMAX:',kmax)

# Read in the yaml file
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

# Add extra bins and two point correlations for red data
if input_name[:3] == 'red':
    cl_bins = 6
    info['likelihood']['cl_like.ClLike']['bins'].append({'name': 'cl6'})
    info['likelihood']['cl_like.ClLike']['twopoints'].extend(({'bins': ['cl6', 'cl6']},
        {'bins': ['cl6', 'sh1']},{'bins': ['cl6', 'sh2']},{'bins': ['cl6', 'sh3']},
        {'bins': ['cl6', 'sh4']},{'bins': ['cl6', 'sh5']}))
else:
    cl_bins = 5

# Determine true bias parameters depending on input
if input_name == 'red_linear':
    bias = [1.6998365377850582,1.8363120052596336,2.0014068547587547,
            2.194876151495485,2.4117020813162657,2.635943465441963]
elif input_name == 'red_HOD':
    bias = [1.5413800308327779,1.710079914623926,1.9648902747695882,
            2.3268344395068925,2.8092307577842197,3.3919171002407063]
elif input_name == 'red_abacus':
    bias = [1.5775086056861591,1.7976528872203488,2.1061869787273917,
            2.506907955836085,3.047912489134238,3.668692156124596]
elif input_name == 'HSC_linear':
    bias = [1.1216110790228042,1.3134406743963547,1.5150686625494223,
            1.788118723644769,2.3806048995796845]
elif input_name == 'HSC_HOD':
    bias = [1.1312074353454078,1.352198442467068,1.597874186348844,
            1.9672143112386806,2.952692757277202]
elif input_name == 'HSC_abacus':
    bias = [1.1588002687074057,1.3828823439943814,1.6308366946984219,
            1.9964258372379027,2.9145030702514796]
elif input_name == 'shear_const':
    bias = [1.,1.,1.,1.,1.]
else:
    bias = [2.,2.,2.,2.,2.,2.]

# Template for bias parameters in yaml file
cl_param = {'prior': {'min': -100.0, 'max': 100.0}, 
        'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.01}, 
        'latex': 'blank', 'proposal': 0.001}

# Set bias parameter types used in each model
if model_name in ['EPT','LPT']:
    bpar = ['1','1p','2','s']
else:
    bpar = ['0','p']
    
# Write bias parameters into yaml file
for b in bpar:
    for i in range(0,cl_bins):
        info['params']['cllike_cl'+str(i+1)+'_b'+b] = cl_param.copy()
        info['params']['cllike_cl'+str(i+1)+'_b'+b]['latex'] = 'b_'+b+'\\,\\text{for}\\,C_{l,'+str(i+1)+'}'
        if b == '0' or b == '1':
            info['params']['cllike_cl'+str(i+1)+'_b'+b]['ref'] = {'dist': 'norm', 'loc': bias[i], 'scale': 0.01}
        # Uncomment to remove bs parameter from minimisation
        # elif b == 's':
            # info['params']['cllike_cl'+str(i+1)+'_bs'] = 0.

# Add model and input file
info['likelihood']['cl_like.ClLike']['bz_model'] = model
info['likelihood']['cl_like.ClLike']['input_file'] = input_file

# Check if directory exists, if not, make directory
if not os.path.exists('results/'+filename):
    os.makedirs('results/'+filename)
    print(filename+' results directory created')

# Save yaml file 
with open('results/'+filename+'/'+filename+'.yml', 'w') as yaml_file:
    yaml.dump(info, yaml_file, default_flow_style=False)

# Add kmax and output file
info['likelihood']['cl_like.ClLike']['defaults']['kmax'] = float(kmax)
info['output'] = 'cobaya_out/' + filename + '_k' + kmax

# Get the mean proposed in the yaml file for each parameter
p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']
os.system('mkdir -p ' + info['output'])

print("params_dict = ", p0)

# Run minimizer
updated_info, sampler = run(info)
bf = sampler.products()['minimum'].bestfit()
pf = {k: bf[k] for k in p0.keys()}
print("Final params: ")
print(pf)

# Compute the likelihoods
model = get_model(info)
loglikes, derived = model.loglikes(p0)
p0_chi2 = -2 * loglikes[0]
loglikes, derived = model.loglikes(pf)
pf_chi2 = -2 * loglikes[0]
#======================DETERMINE ERRORS ON PARAMETERS========================

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
                   "cllike_cl1_b0": 0.1,"cllike_cl2_b0": 0.1,"cllike_cl3_b0": 0.1,
                   "cllike_cl4_b0": 0.1,"cllike_cl5_b0": 0.1,"cllike_cl6_b0": 0.1,  
                   "cllike_cl1_bp": 0.1,"cllike_cl2_bp": 0.1,"cllike_cl3_bp": 0.1,
                   "cllike_cl4_bp": 0.1,"cllike_cl5_bp": 0.1,"cllike_cl6_bp": 0.1,  
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

p0vals = list(p0.values())
pfvals = list(pf.values())
final_params = Fisher(pf)
errs = list(final_params.get_err())

# Save data to file
data = np.column_stack([float(kmax)] + p0vals + pfvals + errs + [p0_chi2] + [pf_chi2])
head = 'PARAMETERS: '+str(list(pf.keys()))+' ; MODEL: \''+ filename +'\'\nkmax   true_params('+str(len(p0vals))+')   calc_params('+str(len(pfvals))+')   errors('+str(len(errs))+')   p0_chi2   pf_chi2'
out = open('results/'+filename+'/results_k'+kmax+'.dat','w')
np.savetxt(out, data, header=head)
out.close()

