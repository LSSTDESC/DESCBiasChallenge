#! /usr/bin/env python

import numpy as np
import pyccl as ccl
from scipy import interpolate

class HaloModCorrection(object):

    def __init__(self, cosmo, hmc, pM, k_range=[1E-1, 5], nlk=20, z_range=[0., 1.], nz=16):

        lkarr = np.linspace(np.log10(k_range[0]), np.log10(k_range[1]), nlk)
        karr = 10.**lkarr
        zarr = np.linspace(z_range[0], z_range[1], nz)
        aarr = 1./(1. + zarr)

        pk_hm = ccl.halos.halomod_power_spectrum(cosmo, hmc, karr, aarr, pM, normprof1=True)
        pk_hf = np.array([ccl.nonlin_matter_power(cosmo, karr, a) for a in aarr])
        ratio = pk_hf / pk_hm

        self.rk_func = interpolate.interp2d(lkarr, zarr, ratio, bounds_error=False, fill_value=1)

    def rk_interp(self, k, a):

        z = 1./a - 1.

        return self.rk_func(np.log10(k), z)
