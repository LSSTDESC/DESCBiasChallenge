import gc
import os

import numpy as np
import asdf
from astropy.io import ascii

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from bitpacked import unpack_rvint
from tools import write_fits

# redshifts
zs = [0.1, 0.3, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0]

# sample directories
sim_name = "AbacusSummit_base_c000_ph006"
red_sample_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks_red/"+sim_name
red_AB_sample_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks_red_AB/"+sim_name
all_sample_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks/"+sim_name
halo_sample_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_David_scratch/mocks_halo/"+sim_name

# simulation directory
Lbox = 2000. # Mpc/h
sim_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/"+sim_name
n_chunks = 34

# for each redshift
for i in range(len(zs)):
    # this redshift
    z = zs[i]
    print("redshift = ", z)
    
    save_halo = True
    if save_halo:
        os.makedirs((halo_sample_dir+f'/z{z:.3f}/halos/'), exist_ok=True)
        # HALO SAMPLE
        
        # filename
        sim_dir = f"/global/project/projectdirs/desi/cosmosim/Abacus/{sim_name:s}/halos/z{z:.3f}/halo_info/"
        cat = CompaSOHaloCatalog(sim_dir, load_subsamples=False, fields = ['N', 'x_L2com'])
        part_mass = cat.header['ParticleMassHMsun']
        mass_halo = cat.halos['N'].astype(np.float64) * part_mass
        poss_halo = cat.halos['x_L2com']

        Mbins = np.array([12.0, 12.5])#, 13.0, 13.5])
        for i in range(len(Mbins)-1):
            # load the halos
            mchoice = (10.**Mbins[i] < mass_halo) & (10.**Mbins[i+1] >= mass_halo) 
            pos_halo = poss_halo[mchoice]
            pos_halo += Lbox/2.
            print("halo positions = ", pos_halo[:5])

            # write out the halo
            try:
                os.unlink(halo_sample_dir+f'/z{z:.3f}/halos/pos_{i+1:d}_halo.fits')
            except OSError:
                pass
            write_fits(pos_halo, f"halo_{i+1:d}", (halo_sample_dir+f'/z{z:.3f}/halos/'))
            del pos_halo
            gc.collect()

        del mass_halo, poss_halo, cat
        gc.collect()
    
    save_galaxy = False
    if save_galaxy:
        # RED SAMPLE AB
        
        # filename
        fn_gal = red_AB_sample_dir+f'/z{z:.3f}/galaxies/LRGs.dat'

        # load the galaxies
        gals_arr = ascii.read(fn_gal)
        pos_gal = np.vstack((gals_arr['x'], gals_arr['y'], gals_arr['z'])).T
        pos_gal += Lbox/2.
        print("galaxy positions = ", pos_gal[:5])
        
        # write out the galaxies
        try:
            os.unlink(red_AB_sample_dir+f'/z{z:.3f}/galaxies/pos_red_AB_galaxy.fits')
        except OSError:
            pass
        write_fits(pos_gal, "red_AB_galaxy", (red_AB_sample_dir+f'/z{z:.3f}/galaxies/'))
        del pos_gal, gals_arr
        gc.collect()

        # RED SAMPLE
        """
        # filename
        fn_gal = red_sample_dir+f'/z{z:.3f}/galaxies/LRGs.dat'

        # load the galaxies
        gals_arr = ascii.read(fn_gal)
        pos_gal = np.vstack((gals_arr['x'], gals_arr['y'], gals_arr['z'])).T
        pos_gal += Lbox/2.
        print("galaxy positions = ", pos_gal[:5])
        
        # write out the galaxies
        try:
            os.unlink(red_sample_dir+f'/z{z:.3f}/galaxies/pos_red_galaxy.fits')
        except OSError:
            pass
        write_fits(pos_gal, "red_galaxy", (red_sample_dir+f'/z{z:.3f}/galaxies/'))
        del pos_gal, gals_arr
        gc.collect()

        # ALL SAMPLE
        # filename
        fn_gal = all_sample_dir+f'/z{z:.3f}/galaxies/LRGs.dat'

        # load the galaxies
        gals_arr = ascii.read(fn_gal)
        pos_gal = np.vstack((gals_arr['x'], gals_arr['y'], gals_arr['z'])).T
        pos_gal += Lbox/2.
        print("galaxy positions = ", pos_gal[:5])
    
        # write out the galaxies
        try:
            os.unlink(all_sample_dir+f'/z{z:.3f}/galaxies/pos_all_galaxy.fits')
        except OSError:
            pass
        write_fits(pos_gal, "all_galaxy", (all_sample_dir+f'/z{z:.3f}/galaxies/'))
        del pos_gal, gals_arr
        gc.collect()
        """

    save_matter = False
    if save_matter:
        # create new directories
        os.makedirs((red_sample_dir+f'/z{z:.3f}/matter/'), exist_ok=True)
    
        # load the matter particles
        for i_chunk in range(n_chunks):
            # halo and field particles
            fn_halo = sim_dir+f'/halos/z{z:.3f}/halo_rv_A/halo_rv_A_{i_chunk:03d}.asdf'
            fn_field = sim_dir+f'/halos/z{z:.3f}/field_rv_A/field_rv_A_{i_chunk:03d}.asdf'

            # write out the halo (L0+L1) matter particles
            halo_data = (asdf.open(fn_halo)['data'])['rvint']
            pos_halo, _ = unpack_rvint(halo_data, Lbox, float_dtype=np.float32, velout=False)
            try:
                os.unlink(red_sample_dir+f'/z{z:.3f}/matter/pos_matter_halo_{i_chunk:03d}.fits')
            except OSError:
                pass
            pos_halo += Lbox/2.
            print("pos_halo = ", pos_halo[:5])
            write_fits(pos_halo, (f"matter_halo_{i_chunk:03d}"), (red_sample_dir+f'/z{z:.3f}/matter/'))
            del halo_data, pos_halo
            gc.collect()
            
            # write out the field matter particles
            field_data = (asdf.open(fn_field)['data'])['rvint']
            pos_field, _ = unpack_rvint(field_data, Lbox, float_dtype=np.float32, velout=False)
            try:
                os.unlink(red_sample_dir+f'/z{z:.3f}/matter/pos_matter_field_{i_chunk:03d}.fits')
            except OSError:
                pass
            pos_field += Lbox/2.
            print("pos_field = ", pos_field[:5])
            write_fits(pos_field, (f"matter_field_{i_chunk:03d}"), (red_sample_dir+f'/z{z:.3f}/matter/'))
            del field_data, pos_field
            gc.collect()
            

