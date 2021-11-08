import numpy as np
import asdf
from astropy.io import ascii

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

# redshifts
zs = [0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.7]
#zs = [1.1, 1.4, 1.7]

# sample directories
sim_name = "AbacusSummit_base_c000_ph006"
Lbox = 2000. # Mpc/h
n_chunks = 34

# mass bins
mass_bins = np.logspace(10, 15, 31)
mass_binc = 0.5 * (mass_bins[1:] + mass_bins[:-1])

dtype = []
for i in range(len(zs)):
    dtype.append((f'z{zs[i]:.3f}', 'f8'))
dtype.append(('mass_binc', 'f8'))
data_hmf = np.empty(len(mass_binc), dtype=dtype)
data_hmf['mass_binc'] = mass_binc
print(dtype)

# for each redshift
for i in range(len(zs)):
    # this redshift
    z = zs[i]
    print("z = ", z)

    # filename
    sim_dir = f"/global/project/projectdirs/desi/cosmosim/Abacus/{sim_name:s}/halos/z{z:.3f}/halo_info/halo_info_000.asdf"
    cat = CompaSOHaloCatalog(sim_dir, load_subsamples='A_halo_rv', fields = ['N'])
    part_mass = cat.header['ParticleMassHMsun']
    mass_halo = cat.halos['N'].astype(np.float64) * part_mass
    
    # load the galaxies
    print("loaded halos")
    hist, edges = np.histogram(mass_halo, bins=mass_bins)
    data_hmf[f'z{z:.3f}'] = hist*n_chunks
 
np.save("HMF.npy", data_hmf)
