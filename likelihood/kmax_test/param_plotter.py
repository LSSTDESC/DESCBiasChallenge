import numpy as np
import matplotlib.pyplot as plt

f = open('kmax_test.dat')
header = f.readline()

data = np.loadtxt( 'kmax_test.dat' )

no_of_params = int((len(data[0,:])-1)/3)
kmax = data[:,0]
true_params = data[0,1:no_of_params+1]
calc_params = data[:,no_of_params+1:2*no_of_params+1]
errs = data[:,2*no_of_params+1:3*no_of_params+1]

def param_plt(param_no):
    param_name = header.split('\'')[2*param_no+1]
    axes[param_no].errorbar(kmax,calc_params[:,param_no],yerr=errs[:,param_no],marker='o')
    axes[param_no].hlines(true_params[param_no], min(kmax), max(kmax), linestyles='dashed',color='r',label='true value')
    axes[param_no].set_ylabel(param_name,fontsize=20)
    axes[param_no].legend()    

f1, axes = plt.subplots(no_of_params,1,sharex='all',figsize=(5,3*no_of_params))
f1.tight_layout(pad=2)
f1.show()
f1.savefig('param_plot.jpg')

for i in range(0,no_of_params):
    param_plt(i)
axes[no_of_params-1].set_xlabel('kmax (1/Mpc)',fontsize=20)
