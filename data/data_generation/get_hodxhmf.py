import numpy as np
import asdf
from astropy.io import ascii

# redshifts
zs = [0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.7]
#zs = [1.1, 1.4, 1.7]

# sample directories
sim_name = "AbacusSummit_base_c000_ph006"
red_sample_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks_red/"+sim_name
all_sample_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks/"+sim_name
Lbox = 2000. # Mpc/h

# mass bins
mass_bins = np.logspace(10, 15, 31)
mass_binc = 0.5 * (mass_bins[1:] + mass_bins[:-1])


dtype = []
for i in range(len(zs)):
    dtype.append((f'z{zs[i]:.3f}', 'f8'))
dtype.append(('mass_binc', 'f8'))
data_all = np.empty(len(mass_binc), dtype=dtype)
data_red = np.empty(len(mass_binc), dtype=dtype)
data_red['mass_binc'] = mass_binc
data_all['mass_binc'] = mass_binc
print(dtype)

# for each redshift
for i in range(len(zs)):
    # this redshift
    z = zs[i]
    print("z = ", z)


    # HOD for galaxies
    fn_gal = red_sample_dir+f'/z{z:.3f}/galaxies/LRGs.dat'
    
    # load the galaxies
    gals_arr = ascii.read(fn_gal)
    mass_halo = gals_arr['mass']
    print("loaded red")
    hist_red, edges = np.histogram(mass_halo, bins=mass_bins)
    data_red[f'z{z:.3f}'] = hist_red
    

    # filename
    fn_gal = all_sample_dir+f'/z{z:.3f}/galaxies/LRGs.dat'

    # load the galaxies
    gals_arr = ascii.read(fn_gal)
    mass_halo = gals_arr['mass']
    print("loaded all")
    hist_all, edges = np.histogram(mass_halo, bins=mass_bins)
    data_all[f'z{z:.3f}'] = hist_all
 

np.save("HODxHMF_all.npy", data_all)
np.save("HODxHMF_red.npy", data_red)
