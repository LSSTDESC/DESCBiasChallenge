import numpy as np
import getdist.mcsamples as gmc

s = gmc.loadMCSamples('cobaya_out/MCMC')  # This reads the chain

print(s.getParamNames())

print('PARAMETERS: ',s.getMeans()[:-2])

print('ERRORS: ',np.sqrt(np.diag(s.getCov()))[:-2])

