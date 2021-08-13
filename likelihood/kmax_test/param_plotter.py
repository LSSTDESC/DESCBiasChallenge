import numpy as np
import matplotlib.pyplot as plt
import os
import sys

folder_name = sys.argv[1]
path = os.getcwd()+'/'+folder_name  # get current working directory

# create combined results file of all kmax
res = open(folder_name+'results.dat','w') 
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and 'results_k' in i:
        f = open(folder_name+i, "r")
        fdata = f.read()
        res.write(fdata+'\n')
        f.close()
res.close()

f = open(folder_name+'results.dat')
header = f.readline()
subheader = f.readline()

data = np.loadtxt(folder_name+'results.dat')
# sort results by kmax
data = data[np.argsort(data[:, 0])]

# extract result data
no_of_params = int((len(data[0,:])-1)/3)
model_name = header.split('\'')[-2]
kmax = data[:,0]
true_params = data[0,1:no_of_params+1]
calc_params = data[:,no_of_params+1:2*no_of_params+1]
errs = data[:,2*no_of_params+1:3*no_of_params+1]
p0_chi2 = data[:,-2]
pf_chi2 = data[:,-1]

# save results to file
np.savetxt(folder_name+'/'+str(model_name)+'.dat',data,header=header[2:]+subheader[2:-1])
os.system('rm '+folder_name+'results.dat')

# plot parameter variation with kmax
def param_plt(param_no):
    param_name = header.split('\'')[2*param_no+1]
    axes[param_no].errorbar(kmax,calc_params[:,param_no],yerr=errs[:,param_no],marker='o')
    axes[param_no].hlines(true_params[param_no], min(kmax), max(kmax), linestyles='dashed',color='r',label='true value')
    axes[param_no].set_ylabel(param_name,fontsize=20)
    axes[param_no].legend()    

f1, axes = plt.subplots(no_of_params+1,1,sharex='all',figsize=(5,3*(no_of_params+1)))
f1.show()

# create plots for all parameters 
for i in range(0,no_of_params):
    param_plt(i)
axes[0].set_title(str(model_name),fontsize=25,y=1.1)
axes[no_of_params].scatter(kmax,pf_chi2,marker='x',label='calculated')
# axes[no_of_params].scatter(kmax,p0_chi2,marker='x',color='r',label='true')
axes[no_of_params].set_xlabel('kmax (1/Mpc)',fontsize=20)
axes[no_of_params].set_ylabel(r'$\chi^2$',fontsize=20)
axes[no_of_params].legend()

plt.tight_layout()
plt.savefig(folder_name+str(model_name)+'.png')

