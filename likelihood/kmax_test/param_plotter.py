import numpy as np
import matplotlib.pyplot as plt
import os
import sys

input_file = sys.argv[1]

# insert models to plot
models = ['Linear','EulerianPT','LagrangianPT']

# initialize arrays
no_of_params = list(models)
header = list(models)
subheader = list(models)
kmax = list(models)
true_params = list(models)
calc_params = list(models)
errs = list(models)
p0_chi2 = list(models)
pf_chi2 = list(models)
line_colour = []

# extract data for each model
for m in range(0,len(models)):
    folder_name = 'results/'+ models[m]+'-'+input_file+'/'
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
    header[m] = f.readline()
    subheader[m] = f.readline()

    data = np.loadtxt(folder_name+'results.dat')
    # sort results by kmax
    data = data[np.argsort(data[:, 0])]

    # save results to file
    model_name = header[m].split('\'')[-2]
    np.savetxt(folder_name+str(model_name)+'.dat',data,header=header[m][2:]+subheader[m][2:-1])
    print('results combined to '+folder_name+str(model_name)+'.dat')
    os.system('rm '+folder_name+'results.dat')

    # extract result data
    no_of_params[m] = int((len(data[0,:])-1)/3)
    kmax[m] = data[:,0]
    true_params[m] = data[0,1:no_of_params[m]+1]
    calc_params[m] = data[:,no_of_params[m]+1:2*no_of_params[m]+1]
    errs[m] = data[:,2*no_of_params[m]+1:3*no_of_params[m]+1]
    p0_chi2[m] = data[:,-2]
    pf_chi2[m] = data[:,-1]

    line_colour.append('C'+str(m))

# determine cosmological parameters used
cosmo_params = header[0].split(", \'cllike")[0].split("[")[1]
cosmo_params = cosmo_params.split(", ")
cosmo_params = [par[1:-1] for par in cosmo_params]
no_of_cosmo = len(cosmo_params)

# number of bias parameters for each model
no_of_bias = [x - no_of_cosmo for x in no_of_params]

# total number of parameters to be included in plots
total_params = sum(no_of_params) - (len(models)-1)*no_of_cosmo

label_true = ['true_value'] + ['_nolabel_']*(len(models)-1)

# plot parameter variation with kmax
def param_plt(i,param_no,m):
    param_name = header[m].split('\'')[2*param_no+1]
    axes[i].errorbar(kmax[m],calc_params[m][:,param_no],yerr=errs[m][:,param_no],
            marker='o',color = line_colour[m],label=models[m])
    # add only one true value line for multiple model plots
    if i<no_of_cosmo:
        axes[i].hlines(true_params[m][param_no], min(kmax[m]), max(kmax[m]),
                linestyles='dashed',color='r',
                label=label_true[m])
    else:
        axes[i].hlines(true_params[m][param_no], min(kmax[m]), max(kmax[m]),
                linestyles='dashed',color='r',label='true value')
    axes[i].set_ylabel(param_name,fontsize=20)
    axes[i].legend()

f1, axes = plt.subplots(total_params+1,1,sharex='all',figsize=(5,3*(total_params+1)))
f1.show()

# create plots for all parameters
for m in range(0,len(models)):
    for i in range(0,total_params):
        # bias parameters for each model
        if ((i>=no_of_cosmo+sum(no_of_bias[:m])) 
                and (i<no_of_cosmo+sum(no_of_bias[:m+1]))):
            param_plt(i,i-sum(no_of_bias[:m]),m)
        # cosmo parameters with all models
        elif (i<no_of_cosmo):
            param_plt(i,i,m)
      
axes[0].set_title(input_file,fontsize=25,y=1.1)
# create chi^2 plot at end
for m in range(0,len(models)):
    axes[total_params].scatter(kmax[m],pf_chi2[m],marker='x',color=line_colour[m],label=models[m]+' calculated')
axes[total_params].set_xlabel('kmax (1/Mpc)',fontsize=20)
axes[total_params].set_ylabel(r'$\chi^2$',fontsize=20)
axes[total_params].legend()

plt.tight_layout()
plt.savefig('results/'+input_file+'.png')
print('figure saved as results/'+input_file+'.png')
