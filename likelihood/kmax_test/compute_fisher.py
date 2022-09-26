import sys
sys.path.append('../likelihood')
from cobaya.model import get_model
import yaml
from cl_like import fisher

import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run FM for already completed minimizer run...')

parser.add_argument('--path2output', dest='path2output', type=str, help='Path to output directory.', required=True)
args = parser.parse_args()

path2output = args.path2output

logger.info('Computing FM for:')
logger.info('{}.'.format(path2output))

info = yaml.load(path2output+'.yml', Loader=yaml.FullLoader)
model = get_model(info)
bestfit = np.loadtxt(path2output+'.bestfit', skiprows=2, usecols=1)
keys = np.loadtxt(path2output+'.bestfit', skiprows=2, usecols=2, dtype='U1012')

max_ind = np.where(keys=='m_nu')[0][0]
logger.info('Number of fitted parameters = {}.'.format(max_ind))
bestfit = bestfit[:max_ind]
keys = keys[:max_ind]

param_dict = dict(zip(keys, bestfit))

# Run error estimation fisher code
# Method: first derivative
F = fisher.Fisher_first_deri(model=model, parms=param_dict, fp_name=list(param_dict.keys()),
                                 step_factor=0.01, method='five-stencil', full_expresssion=False,
                                 cov_mode='spec_dec')
cov, FM = F.get_cov()

# Save data to file
np.savez(path2output+'.fisher_fd_mode=SpecDec.npz', cov=cov, fisher=FM)
