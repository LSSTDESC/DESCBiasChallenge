# Script to combine raw LastJourney power spectrum
# Since dealing with raw data, using interpolation instead of smoothing and fitting.
import asdf
import numpy as np
import h5py
import pyccl as ccl
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def get_LJ_cosmo():

    h = 0.6766
    omega_b = 0.049 
    omega_cdm = 0.26067 
    A_s = np.e**3.047/1e10 #2.083e-09
    n_s = 0.9665
    sigma8 = 0.8102 

    cosmo = ccl.Cosmology(Omega_c=omega_cdm, Omega_b=omega_b,
                            h=h, A_s=A_s, n_s=n_s, m_nu=0.0)

    cosmo.compute_linear_power()
    cosmo.compute_nonlin_power()
    return cosmo

class LastJourney_Interpolator(object):
    def __init__(self,cosmo,step,mb):
        self.step = step
        self.mb = mb
        self.pks = {}
        # read data
        with h5py.File(f"LastJourneyData/pk_{self.step}.hdf5", "r") as f:
            ds_hh = f[f'pk_hh_massbin_{self.mb}']  # auto-spectra dataset
            ds_hm = f[f'pk_hm_massbin_{self.mb}']  # cross-spectra dataset
            pks_hh = ds_hh[:]
            pks_hm = ds_hm[:]
            pks_mm = f['pk_mm'][:]
            mass_lo = np.log10(ds_hh.attrs['delta1_mass_low'])  # log10(M200c)
            mass_hi = np.log10(ds_hh.attrs['delta1_mass_high'])  #log10(M200c)
        print(f"mass bin {mb} ({mass_lo}, {mass_hi})")
  
        ks = pks_mm[:,0]
        k_mask = (ks < 1.5)
        self.pks['k_s'] = ks[k_mask]
        self.pks['Pk_mm']= pks_mm[:,1][k_mask]
        #pkhh
        self.pks['Pk_hh'] = pks_hh[:,1][k_mask]
        #pkhm
        self.pks['Pk_hm']= pks_hm[:,1][k_mask]
        
    def get_pk_interp(self,tr1,tr2):
        pk_interp = interp1d(self.pks['k_s'], self.pks[f'Pk_{tr1}{tr2}'], 
                      fill_value='extrapolate', bounds_error=False)
        return(pk_interp)
    
    
cosmo = get_LJ_cosmo()
steps = [331,338,347,355,365,373,382,392,401]
Interpolators = [LastJourney_Interpolator(cosmo, s,mb = 0) for s in steps]


def get_pk2d_LJ():
    h = cosmo['h']
    a_s = (1/201. + (1 - 1/120)/500 * (np.array(steps) + 1))
    k_s = np.geomspace(1E-3, 1.2, 512)
    tr_combs = [('m','m'),
                ('h','h'),
                ('h','m')]
    pks = {}
    for t1,t2 in tr_combs:
        pks[f'{t1}_{t2}'] = np.array([f.get_pk_interp(t1,t2)(k_s/h)/h**3
                                    for f in Interpolators])
    pks_ccl = np.array([ccl.nonlin_matter_power(cosmo,k_s,a)
                      for a in a_s])
    ratio = pks_ccl/pks['m_m']
    return (a_s,k_s,pks,pks_ccl,ratio)

a_s, k_s, pks, pks_ccl, ratio = get_pk2d_LJ()
np.savez('LastJourneyData/pk2d_halo_Mmin=12-Max=12p5_LJ_raw.npz',
        a_s = a_s, k_s = k_s, pk_ccl = pks_ccl, **pks)