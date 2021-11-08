#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

MODIFIED BS abacus_hod.py and GRAND_HOD.py in anaconda3
/global/homes/b/boryanah/anaconda3/envs/desc/lib/python3.7/site-packages/abacusnbody/hod/GRAND_HOD.py

Usage
-----
$ python ./run_hod.py --help
'''

import os
import glob
import time

import yaml
import numpy as np
import argparse

from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/red.yaml'

def get_logMmin(z, z_pivot, logMmin_0, logMmin_p):
    #log10(M_min) = lMmin_0 + lMmin_p * (1/(1+z) - 1/(1+z_pivot))
    return logMmin_0 + logMmin_p * (1/(1+z) - 1/(1+z_pivot))

def get_logM0(z, z_pivot, logM0_0, logM0_p):
    #log10(M_0) = lM0_0 + lM0_p * (1/(1+z) - 1/(1+z_pivot))
    return logM0_0 + logM0_p * (1/(1+z) - 1/(1+z_pivot))

def get_logM1(z, z_pivot, logM1_0, logM1_p):
    #log10(M_1) = lM1_0 + lM1_p * (1/(1+z) - 1/(1+z_pivot))
    return logM1_0 + logM1_p * (1/(1+z) - 1/(1+z_pivot))

def convert_david_to_sandy(logMmin, logM0, logM1, sigma_d, alpha):
    logM_cut = logMmin
    kappa = 10.**logM0/10.**logM_cut
    sigma = sigma_d/np.sqrt(2)
    # alpha and logM1 are the same
    return logM_cut, kappa, logM1, sigma, alpha

def david_to_sandy(z, z_pivot, logMmin_0, logMmin_p, logM0_0, logM0_p, logM1_0, logM1_p, alpha, sigma_lnM):
    logMmin = get_logMmin(z, z_pivot, logMmin_0, logMmin_p)
    logM0 = get_logM0(z, z_pivot, logM0_0, logM0_p)
    logM1 = get_logM1(z, z_pivot, logM1_0, logM1_p)
    logM_cut, kappa, logM1, sigma, alpha = convert_david_to_sandy(logMmin, logM0, logM1, sigma_lnM, alpha)
    sandy_dic = {}
    sandy_dic['logM_cut'] = logM_cut
    sandy_dic['logM1'] = logM1
    sandy_dic['kappa'] = kappa
    sandy_dic['sigma'] = sigma
    sandy_dic['alpha'] = alpha
    return sandy_dic


def main(path2config):
    
    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']

    # parameters that vary
    sample_params = yaml.load(open('config/sample_params.yaml'))
    all_sample_dict = sample_params['all_sample']
    red_sample_dict = sample_params['red_sample']

    zs = [0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0]

    if 'red_AB' in path2config:
        gal_type = 'red_AB'
    elif 'red' in path2config:
        gal_type = 'red'
    elif 'all' in path2config:
        gal_type = 'all'
        
    # run the HODs
    for i in range(len(zs)):
        print(i)
        
        # create a new abacushod object
        sim_params['z_mock'] = zs[i]
        newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

        if gal_type == 'all':
            # dictionary with parameters
            all_sample_dict['z'] = zs[i]
            sandy_dic = david_to_sandy(**all_sample_dict)        
            for key in sandy_dic.keys():
                newBall.tracers['LRG'][key] = sandy_dic[key]
            start = time.time()
            mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread = 64)
            print("Done hod, took time ", time.time() - start)
        
        if 'red' in gal_type:
            # needs to have a different name
            # dictionary with parameters
            red_sample_dict['z'] = zs[i]
            sandy_dic = david_to_sandy(**red_sample_dict)        
            for key in sandy_dic.keys():
                newBall.tracers['LRG'][key] = sandy_dic[key]
            start = time.time()
            mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread = 64)
            print("Done hod, took time ", time.time() - start)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
