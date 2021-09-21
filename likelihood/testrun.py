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

# Compute the likelihood at that point
model = get_model(info)
loglikes, derived = model.loglikes(p0)
print("chi2 = ", -2 * loglikes[0])

# Run the sampler
updated_info, sampler = run(info)
bf = sampler.products()['minimum'].bestfit()
pf = {k: bf[k] for k in p0.keys()}
print("Final params: ")
print(pf)
