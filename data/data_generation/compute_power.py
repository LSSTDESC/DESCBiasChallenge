import glob
from tools import get_single_Pk, get_all_red_Pk, get_mat_Pk, write_asdf

# parameters
Lbox = 2000. # Mpc/h
N_dim = 2304
interlaced = True
redshifts = [0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0]
sim_name = "AbacusSummit_base_c000_ph006"

header = {}
header['Lbox'] = Lbox
header['N_dim'] = N_dim
header['sim_name'] = sim_name
header['interlaced'] = interlaced
header['units'] = 'Mpc/h'

for redshift in redshifts:
    header['redshift'] = redshift

    # filenames
    pos_gal_fns = f"/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks_red_AB/{sim_name:s}/z{redshift:.3f}/galaxies/pos_red_AB_galaxy.fits"
    pos_gal_red_fns = f"/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks_red/{sim_name:s}/z{redshift:.3f}/galaxies/pos_red_galaxy.fits"
    pos_gal_all_fns = f"/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks/{sim_name:s}/z{redshift:.3f}/galaxies/pos_all_galaxy.fits"
    pos_mat_fns = sorted(glob.glob(f"/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks_red/{sim_name:s}/z{redshift:.3f}/matter/pos_matter_*.fits"))
    dens_dir = f"/global/cscratch1/sd/boryanah/data_hybrid/abacus/{sim_name:s}/"
    data_dir = "/global/homes/b/boryanah/lsst_bias/data/"
    
    # save all power spectra
    # matter
    power_mat_dic = get_mat_Pk(pos_mat_fns, dens_dir, data_dir, N_dim, Lbox, interlaced, dk=None)
    write_asdf(power_mat_dic, (f"power_mat_z{redshift:.3f}.asdf"), data_dir, header=header)

    # galaxies and matter
    power_all_dic, power_red_dic = get_all_red_Pk(pos_gal_all_fns, pos_gal_red_fns, pos_mat_fns, dens_dir, data_dir, N_dim, Lbox, interlaced, dk=None)

    header['gal_sample'] = 'all'
    write_asdf(power_all_dic, (f"power_all_z{redshift:.3f}.asdf"), data_dir, header=header)

    header['gal_sample'] = 'red'
    write_asdf(power_red_dic, (f"power_red_z{redshift:.3f}.asdf"), data_dir, header=header)

    # additional assembly bias sample (added later)
    power_dic = get_single_Pk(pos_gal_fns, pos_mat_fns, dens_dir, data_dir, N_dim, Lbox, interlaced, dk=None)
    header['gal_sample'] = 'red_AB'
    write_asdf(power_dic, (f"power_red_AB_z{redshift:.3f}.asdf"), data_dir, header=header)
