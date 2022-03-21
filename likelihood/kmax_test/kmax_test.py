import sys
sys.path.append('../likelihood')
from cobaya.model import get_model
from cobaya.run import run
from shutil import copyfile
import yaml
from cl_like import fisher
import os

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
parser.add_argument('--ref_bsn', dest='ref_bsn', nargs='+', help='bsn reference distribution (for initializtion).',
                    required=False)
parser.add_argument('--ref_b1', dest='ref_b1', nargs='+', help='b1 reference distribution (for initializtion).',
                    required=False)
parser.add_argument('--name_like', dest='name_like', type=str, help='Name of likelihood.', required=False,
                    default='cl_like.ClLike')
parser.add_argument('--sampler_type', dest='sampler_type', help='Type of sampler used.', default='minimizer',
                    required=False)

args = parser.parse_args()

path2output = args.path2output

bias_model = args.bias_model
k_max = args.k_max
fit_params = args.fit_params
name_like = args.name_like

# Set model
if bias_model == 'lin':
    model = 'Linear'
elif bias_model in ['EuPT', '3EuPT', '3EuPT_bk2', '3EuPT_b3nl']:
    model = 'EulerianPT'
elif bias_model in ['LPT', '3LPT', '3LPT_bk2', '3LPT_b3nl']:
    model = 'LagrangianPT'
elif bias_model in ['BACCO', '3BACCO_bk2', '3BACCO_bk2+bsn']:
    model = 'BACCO'
elif bias_model in ['anzu', '3anzu_bk2', '3anzu_bk2+bsn']:
    model = 'anzu'
elif bias_model == 'HOD':
    model = 'HOD'
else:
    raise ValueError("Unknown bias model")

logger.info('Running analysis for:')
logger.info('Likelihood: {}.'.format(name_like))
logger.info('Bias model: {}.'.format(bias_model))
logger.info('k_max: {}.'.format(k_max))

# Read in the yaml file
config_fn = args.path2defaultconfig
with open(config_fn, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

# Determine true bias parameters depending on input
DEFAULT_REF_B1 = 2.

# Default reference value for bsn
DEFAULT_REF_BSN = 1000.

# Note: we need to hard-code the BACCO parameter bounds:
# omega_matter: [0.23, 0.4 ]
# sigma8: [0.73, 0.9 ]
# omega_baryon: [0.04, 0.06]
# ns: [0.92, 1.01]
# hubble: [0.6, 0.8]
# neutrino_mass: [0. , 0.4]
# w0: [-1.15, -0.85]
# wa: [-0.3,  0.3]
# expfactor': [0.4, 1. ]

if 'sigma8' in fit_params:
    info['params']['sigma8'] = {'prior': {'min': 0.1, 'max': 1.2},
                                                    'ref': {'dist': 'norm', 'loc': 0.8090212289405192, 'scale': 0.01},
                                                    'latex': '\sigma_8', 'proposal': 0.001}
    if model == 'BACCO':
        info['params']['sigma8']['prior'] = {'min': 0.73, 'max': 0.9}
elif args.sigma8 is not None:
    info['params']['sigma8'] = args.sigma8
else:
    info['params']['sigma8'] = 0.8090212289405192
if 'Omega_c' in fit_params:
    info['params']['Omega_c'] = {'prior': {'min': 0.05, 'max': 0.7},
                                                    'ref': {'dist': 'norm', 'loc': 0.26447041034523616, 'scale': 0.01},
                                                    'latex': '\Omega_c', 'proposal': 0.001}
    if model == 'BACCO':
        info['params']['Omega_c']['prior'] = {'min': 0.19, 'max': 0.36}
elif args.Omega_c is not None:
    info['params']['Omega_c'] = args.Omega_c
else:
    info['params']['Omega_c'] = 0.26447041034523616
if 'Omega_b' in fit_params:
    info['params']['Omega_b'] = {'prior': {'min': 0.01, 'max': 0.2},
                                                    'ref': {'dist': 'norm', 'loc': 0.049301692328524445, 'scale': 0.01},
                                                    'latex': '\Omega_b', 'proposal': 0.001}
    if model == 'BACCO':
        info['params']['Omega_b']['prior'] = {'min': 0.04, 'max': 0.06}
elif args.Omega_b is not None:
    info['params']['Omega_b'] = args.Omega_b
else:
    info['params']['Omega_b'] = 0.049301692328524445
if 'h' in fit_params:
    info['params']['h'] = {'prior': {'min': 0.1, 'max': 1.2},
                                                    'ref': {'dist': 'norm', 'loc': 0.6736, 'scale': 0.01},
                                                    'latex': 'h', 'proposal': 0.001}
    if model == 'BACCO':
        info['params']['h']['prior'] = {'min': 0.6, 'max': 0.8}
elif args.h is not None:
    info['params']['h'] = args.h
else:
    info['params']['h'] = 0.6736
if 'n_s' in fit_params:
    info['params']['n_s'] = {'prior': {'min': 0.1, 'max': 1.2},
                                                    'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.01},
                                                    'latex': 'n_s', 'proposal': 0.001}
    if model == 'BACCO':
        info['params']['n_s']['prior'] = {'min': 0.92, 'max': 1.01}
elif args.n_s is not None:
    info['params']['n_s'] = args.n_s
else:
    info['params']['n_s'] = 0.9649

probes = args.probes
if probes != ['all']:
    logger.info('Only fitting a subset of bins and probes.')
    info['likelihood'][name_like]['bins'] = [{'name': bin_name} for bin_name in args.bins]
    info['likelihood'][name_like]['twopoints'] = [{'bins': [probes[2*i], probes[2*i+1]]} for i in range(len(probes)//2)]
    n_bin = len(args.bins)
    bin_nos = [int(bin_name[-1])-1 for bin_name in args.bins if 'cl' in bin_name]
else:
    logger.info('Fitting all bins and probes.')
    if 'red' in args.path2data:
        logger.info('Looking at red sample with 6 clustering bins.')
        info['likelihood'][name_like]['bins'] = [{'name': 'cl1'}, {'name': 'cl2'}, {'name': 'cl3'}, {'name': 'cl4'}, {'name': 'cl5'},\
                                                 {'name': 'cl6'}, {'name': 'sh1'}, {'name': 'sh2'}, {'name': 'sh3'}, {'name': 'sh4'}, \
                                                 {'name': 'sh5'}]
        info['likelihood'][name_like]['twopoints'] = [{'bins': ['cl1', 'cl1']}, {'bins': ['cl2', 'cl2']}, {'bins': ['cl3', 'cl3']}, \
                                                      {'bins': ['cl4', 'cl4']}, {'bins': ['cl5', 'cl5']}, {'bins': ['cl6', 'cl6']}, \
                                                      {'bins': ['cl1', 'sh1']}, {'bins': ['cl1', 'sh2']}, {'bins': ['cl1', 'sh3']}, \
                                                      {'bins': ['cl1', 'sh4']}, {'bins': ['cl1', 'sh5']}, {'bins': ['cl2', 'sh1']}, \
                                                      {'bins': ['cl2', 'sh2']}, {'bins': ['cl2', 'sh3']}, {'bins': ['cl2', 'sh4']}, \
                                                      {'bins': ['cl2', 'sh5']}, {'bins': ['cl3', 'sh1']}, {'bins': ['cl3', 'sh2']}, \
                                                      {'bins': ['cl3', 'sh3']}, {'bins': ['cl3', 'sh4']}, {'bins': ['cl3', 'sh5']}, \
                                                      {'bins': ['cl4', 'sh1']}, {'bins': ['cl4', 'sh2']}, {'bins': ['cl4', 'sh3']}, \
                                                      {'bins': ['cl4', 'sh4']}, {'bins': ['cl4', 'sh5']}, {'bins': ['cl5', 'sh1']}, \
                                                      {'bins': ['cl5', 'sh2']}, {'bins': ['cl5', 'sh3']}, {'bins': ['cl5', 'sh4']}, \
                                                      {'bins': ['cl5', 'sh5']}, {'bins': ['cl6', 'sh1']}, {'bins': ['cl6', 'sh2']}, \
                                                      {'bins': ['cl6', 'sh3']}, {'bins': ['cl6', 'sh4']}, {'bins': ['cl6', 'sh5']}, \
                                                      {'bins': ['sh1', 'sh1']}, {'bins': ['sh1', 'sh2']}, {'bins': ['sh1', 'sh3']}, \
                                                      {'bins': ['sh1', 'sh4']}, {'bins': ['sh1', 'sh5']}, {'bins': ['sh2', 'sh2']}, \
                                                      {'bins': ['sh2', 'sh3']}, {'bins': ['sh2', 'sh4']}, {'bins': ['sh2', 'sh5']}, \
                                                      {'bins': ['sh3', 'sh3']}, {'bins': ['sh3', 'sh4']}, {'bins': ['sh3', 'sh5']}, \
                                                      {'bins': ['sh4', 'sh4']}, {'bins': ['sh4', 'sh5']}, {'bins': ['sh5', 'sh5']}]
        n_bin = len(info['likelihood'][name_like]['bins'])
        bin_nos = [int(bin_dict['name'][-1]) - 1 for bin_dict in info['likelihood'][name_like]['bins'] if 'cl' in bin_dict['name']]
    elif 'HSC' in args.path2data:
        logger.info('Looking at HSC sample with 5 clustering bins.')
        info['likelihood'][name_like]['bins'] = [{'name': 'cl1'}, {'name': 'cl2'}, {'name': 'cl3'}, {'name': 'cl4'}, {'name': 'cl5'},\
                                                 {'name': 'sh1'}, {'name': 'sh2'}, {'name': 'sh3'}, {'name': 'sh4'}, {'name': 'sh5'}]
        info['likelihood'][name_like]['twopoints'] = [{'bins': ['cl1', 'cl1']}, {'bins': ['cl2', 'cl2']}, {'bins': ['cl3', 'cl3']}, \
                                                      {'bins': ['cl4', 'cl4']}, {'bins': ['cl5', 'cl5']}, {'bins': ['cl6', 'cl5']}, \
                                                      {'bins': ['cl1', 'sh1']}, {'bins': ['cl1', 'sh2']}, {'bins': ['cl1', 'sh3']}, \
                                                      {'bins': ['cl1', 'sh4']}, {'bins': ['cl1', 'sh5']}, {'bins': ['cl2', 'sh1']}, \
                                                      {'bins': ['cl2', 'sh2']}, {'bins': ['cl2', 'sh3']}, {'bins': ['cl2', 'sh4']}, \
                                                      {'bins': ['cl2', 'sh5']}, {'bins': ['cl3', 'sh1']}, {'bins': ['cl3', 'sh2']}, \
                                                      {'bins': ['cl3', 'sh3']}, {'bins': ['cl3', 'sh4']}, {'bins': ['cl3', 'sh5']}, \
                                                      {'bins': ['cl4', 'sh1']}, {'bins': ['cl4', 'sh2']}, {'bins': ['cl4', 'sh3']}, \
                                                      {'bins': ['cl4', 'sh4']}, {'bins': ['cl4', 'sh5']}, {'bins': ['cl5', 'sh1']}, \
                                                      {'bins': ['cl5', 'sh2']}, {'bins': ['cl5', 'sh3']}, {'bins': ['cl5', 'sh4']}, \
                                                      {'bins': ['cl5', 'sh5']}, {'bins': ['sh1', 'sh1']}, {'bins': ['sh1', 'sh2']}, \
                                                      {'bins': ['sh1', 'sh3']}, {'bins': ['sh1', 'sh4']}, {'bins': ['sh1', 'sh5']}, \
                                                      {'bins': ['sh2', 'sh2']}, {'bins': ['sh2', 'sh3']}, {'bins': ['sh2', 'sh4']}, \
                                                      {'bins': ['sh2', 'sh5']}, {'bins': ['sh3', 'sh3']}, {'bins': ['sh3', 'sh4']}, \
                                                      {'bins': ['sh3', 'sh5']}, {'bins': ['sh4', 'sh4']}, {'bins': ['sh4', 'sh5']}, \
                                                      {'bins': ['sh5', 'sh5']}]
        n_bin = len(info['likelihood'][name_like]['bins'])
        bin_nos = [int(bin_dict['name'][-1]) - 1 for bin_dict in info['likelihood'][name_like]['bins'] if 'cl' in bin_dict['name']]

# Set bias parameter types used in each model
if bias_model in ['EuPT', '3EuPT', '3EuPT_bk2', '3EuPT_b3nl',
                  'LPT', '3LPT', '3LPT_bk2', '3LPT_b3nl',
                  'BACCO', '3BACCO_bk2', '3BACCO_bk2+bsn',
                  'anzu', '3anzu_bk2', '3anzu_bk2+bsn']:
    bpar = ['1', '1p', '2', 's', '3nl', 'k2', 'sn']
elif bias_model == 'Linear':
    bpar = ['1','1p']
elif bias_model == 'HOD':
    bpar = ['lMmin_0', 'lMmin_p',
            'siglM_0', 'siglM_p',
            'lM0_0', 'lM0_p',
            'lM1_0', 'lM1_p',
            'alpha_0', 'alpha_p']

ref_bsn = args.ref_bsn
if args.ref_bsn is not None:
    ref_bsn = [0 for i in range(len(args.ref_bsn))]
    for i, ref in enumerate(args.ref_bsn):
        if ref != 'None':
            ref_bsn[i] = float(ref)
        else:
            ref_bsn[i] = None
else:
    ref_bsn = [None for i in range(n_bin)]
ref_b1 = args.ref_b1
if args.ref_b1 is not None:
    ref_b1 = [0 for i in range(len(args.ref_b1))]
    for i, ref in enumerate(args.ref_b1):
        if ref != 'None':
            ref_b1[i] = float(ref)
        else:
            ref_b1[i] = None
else:
    ref_b1 = [None for i in range(n_bin)]

if bias_model != 'HOD':
    # Template for bias parameters in yaml file
    cl_param = {'prior': {'min': -100.0, 'max': 100.0},
            'ref': {'dist': 'norm', 'loc': 0., 'scale': 0.01},
            'latex': 'blank', 'proposal': 0.001}
else:
    HOD_means = [12.95, -2.0, 0.25, 0., 12.3, 0., 14.0, -1.5, 1.32, 0.]
    cl_param = {'prior': {'min': -100.0, 'max': 100.0},
            'ref': {'dist': 'norm', 'loc': 'blank', 'scale': 0.1},
            'latex': 'blank', 'proposal': 0.001}

# Add model and input file
info['likelihood'][name_like]['bz_model'] = model
info['likelihood'][name_like]['input_file'] = args.path2data

# Write bias parameters into yaml file
input_params_prefix = info['likelihood'][name_like]['input_params_prefix']
if bias_model != 'HOD':
    for b in bpar:
        for i in bin_nos:
            param_name = input_params_prefix+'_cl'+str(i+1)+'_b'+b
            if param_name in fit_params:
                info['params'][param_name] = cl_param.copy()
                info['params'][param_name]['latex'] = 'b_'+b+'\\,\\text{for}\\,C_{l,'+str(i+1)+'}'
                if b == '0' or b == '1':
                    if ref_b1[i] is not None:
                        mean = ref_b1[i]
                    else:
                        mean = DEFAULT_REF_B1
                    info['params'][input_params_prefix+'_cl'+str(i+1)+'_b'+b]['ref'] = {'dist': 'norm', 'loc': mean, 'scale': 0.1}
                elif b == 'sn':
                    if ref_bsn[i] is not None:
                        mean = ref_bsn[i]
                    else:
                        mean = DEFAULT_REF_BSN

                    info['params'][input_params_prefix + '_cl' + str(i + 1) + '_b' + b]['ref'] = {'dist': 'norm',
                                                                                                  'loc': mean,
                                                                                                  'scale': 0.1*np.abs(mean)}
                    info['params'][input_params_prefix + '_cl' + str(i + 1) + '_b' + b]['prior'] = {'min': -2.*np.abs(mean),
                                                                                                    'max': 2.*np.abs(mean)}
            else:
                if b == '0' or b == '1':
                    info['params'][input_params_prefix+'_cl'+str(i+1)+'_b'+b] = DEFAULT_REF_B1
                else:
                    info['params'][input_params_prefix+'_cl' + str(i + 1) + '_b' + b] = 0.
else:
    for i, b in enumerate(bpar):
        param_name = input_params_prefix + '_hod_' + b
        if param_name in fit_params:
            info['params'][param_name] = cl_param.copy()
            info['params'][param_name]['latex'] = b + '\\,\\text{for HOD}'
            info['params'][param_name]['ref'] = {'dist': 'norm', 'loc': HOD_means[i], 'scale': 0.1}
        else:
            info['params'][param_name] = HOD_means[i]

# Add kmax and output file
info['likelihood'][name_like]['defaults']['kmax'] = float(k_max)
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
if args.sampler_type == 'minimizer':
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

    p_all = {}
    for p in info['params']:
        if isinstance(info['params'][p], dict):
            if 'ref' in info['params'][p]:
                p_all[p] = bf[p]
        else:
            p_all[p] = info['params'][p]

    # Run error estimation fisher code
    F = fisher.Fisher_first_deri(model = model, parms = p_all, fp_name = list(pf.keys()),
                                 step_factor = 0.01, method = 'five-stencil', full_expresssion = False)
    cov = F.get_cov()

    p0vals = list(p0.values())
    pfvals = list(pf.values())

    # Save data to file
    np.savez(info['output']+'.fisher.npz', bf=pfvals, truth=p0vals, chi2_bf=pf_chi2, chi2_truth=p0_chi2, cov=cov)
elif args.sampler_type == 'mcmc':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from cobaya.run import run
    from cobaya.log import LoggedError

    success = False
    try:
        upd_info, mcmc = run(info)
        success = True
    except LoggedError as err:
        pass

    # Did it work? (e.g. did not get stuck)
    success = all(comm.allgather(success))

    if not success and rank == 0:
        print("Sampling failed!")