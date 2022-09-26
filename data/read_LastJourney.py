import numpy as np
import h5py

pks_hh = {}  # halo-halo auto-spectrum     key=mass-bin: 0, 1, 2, 3
pks_hm = {}  # halo-matter cross-spectrum  key=mass-bin: 0, 1, 2, 3

step = 331  # Available: 331 338 347 355 365 373 382 392 401
with h5py.File(f"LastJourneyData/pk_{step}.hdf5", "r") as f:
    for mb in range(4):
        # All pk files have the following format:
        # column 0: ks [h Mpc^-1]
        # column 1: pk [(h^-1 Mpc)^3]
        # column 2: number of k modes in bin
        ds_hh = f[f'pk_hh_massbin_{mb}']  # auto-spectra dataset
        ds_hm = f[f'pk_hm_massbin_{mb}']  # cross-spectra dataset

        # mass range and number of halos/matter particles for print statement
        mass_lo = np.log10(ds_hh.attrs['delta1_mass_low'])  # log10(M200c)
        mass_hi = np.log10(ds_hh.attrs['delta1_mass_high'])  # log10(M200c)
        nparts_h = ds_hh.attrs['delta1_nparts']  # number of halos
        nparts_m = ds_hm.attrs['delta2_nparts']  # number of particles for matter cross
        print(f"mass bin {mb} ({mass_lo}, {mass_hi}): nhalos={nparts_h:8d} nparticles={nparts_m:12d}")

        pks_hh[mb] = ds_hh[:]
        pks_hm[mb] = ds_hm[:]

    # matter-matter auto-spectrum
    pks_mm = f['pk_mm'][:]