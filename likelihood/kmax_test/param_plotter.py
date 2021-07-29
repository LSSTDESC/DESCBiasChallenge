import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt( 'kmax_test.dat' )

no_of_params = int((len(data[0,:])-1)/3)
kmax = data[:,0]
true_params = data[0,1:no_of_params+1]
calc_params = data[:,no_of_params+1:2*no_of_params+1]
errs = data[:,2*no_of_params+1:3*no_of_params+1]

plt.errorbar(kmax,calc_params[:,0],yerr=errs[:,0])
plt.hlines(kmax, min(kmax), max(kmax), linestyles='dashed')
plt.savefig('param_plot.jpg')
