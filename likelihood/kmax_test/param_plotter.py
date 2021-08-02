import numpy as np
import matplotlib.pyplot as plt
import os

path = os.getcwd()  # get current working directory

# create combined results file of all kmax
res = open('results.dat','w') 
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and 'result_k' in i:
        f = open(i, "r")
        fdata = f.read()
        res.write(fdata+'\n')  
        f.close()
res.close()

f = open('results.dat')
header = f.readline()

data = np.loadtxt('results.dat')
# sort results by kmax
data = data[np.argsort(data[:, 0])]

# extract result data
no_of_params = int((len(data[0,:])-1)/3)
kmax = data[:,0]
true_params = data[0,1:no_of_params+1]
calc_params = data[:,no_of_params+1:2*no_of_params+1]
errs = data[:,2*no_of_params+1:3*no_of_params+1]

# plot parameter variation with kmax
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

# create plots for all parameters 
for i in range(0,no_of_params):
    param_plt(i)
axes[no_of_params-1].set_xlabel('kmax (1/Mpc)',fontsize=20)
