import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import sys
sys.path.append('../likelihood/cl_like')
from bacco import BACCOCalculator, get_bacco_pk2d
from scipy.integrate import simps
import os


def get_abacus_cosmo():
    """ Returns the Abacus cosmology as a CCL object
    """

    cosmology = {'omega_b': 0.02237,
                 'omega_cdm': 0.12,
                 'h': 0.6736,
                 'A_s': 2.083e-09,
                 'n_s': 0.9649,
                 'alpha_s': 0.0,
                 'N_ur': 2.0328,
                 'N_ncdm': 1.0,
                 'omega_ncdm': 0.0006442,
                 'w0_fld': -1.0,
                 'wa_fld': 0.0
                 }
    cosmo = ccl.Cosmology(Omega_c=cosmology['omega_cdm']/cosmology['h']**2,
                          Omega_b=cosmology['omega_b']/cosmology['h']**2,
                          h=cosmology['h'], A_s=cosmology['A_s'], n_s=cosmology['n_s'],
                          m_nu=0.06)
    sigma8 = ccl.sigma8(cosmo)
    cosmo = ccl.Cosmology(Omega_c=cosmology['omega_cdm']/cosmology['h']**2,
                          Omega_b=cosmology['omega_b']/cosmology['h']**2,
                          h=cosmology['h'], sigma8=sigma8, n_s=cosmology['n_s'],
                          m_nu=0.06)
    cosmo.compute_linear_power()
    cosmo.compute_nonlin_power()
    return cosmo, cosmology


class DataGenerator(object):
    lmax = 5000

    def __init__(self, config):
        self.c = config
        print(self.c['sacc_name'])
        # Read redshift distributions and compute number densities
        # for all redshift bins
        d = np.load(self.c['dNdz_file'])
        self.z_cl = d['z_cl']
        self.nz_cl = d['dNdz_cl'].T
        norms_cl = simps(self.nz_cl, x=self.z_cl)
        self.ndens_cl = self.c['ndens_cl']*norms_cl/np.sum(norms_cl)
        self.ndens_cl *= (180*60/np.pi)**2
        self.z_sh = d['z_sh']
        self.nz_sh = d['dNdz_sh'].T
        norms_sh = simps(self.nz_sh, x=self.z_sh)
        self.ndens_sh = self.c['ndens_sh']*norms_sh/np.sum(norms_sh)
        self.ndens_sh *= (180*60/np.pi)**2
        self.n_cl = len(self.ndens_cl)
        self.n_sh = len(self.ndens_sh)

        # Cosmological model
        if 'cosmology' in self.c:
            if isinstance(self.c['cosmology'], dict):
                self.cosmo = ccl.Cosmology(**(self.c['cosmology']))
            else:
                if self.c['cosmology'] == 'Abacus':
                    self.cosmo, cosmo_dict = get_abacus_cosmo()
                    self.c['cosmology'] = cosmo_dict
                else:
                    raise ValueError('Unknown cosmology')
        else:
            self.cosmo = ccl.CosmologyVanillaLCDM()
        ccl.sigma8(self.cosmo)

        # Bias model
        self.bias_model = self.c['bias']['model']
        self.ll = None

    def _get_covariance(self, cls, unwrap=True):
        """ Generates Gaussian covariance given power spectrum matrix
        """
        nt, _, nl = cls.shape
        ll = self._get_ell_sampling()
        ls = ll['mean']
        dl = ll['d_ell']
        fsky = self.c.get('fsky', 0.4)
        ncl = (nt*(nt+1)) // 2
        cov = np.zeros([ncl, nl, ncl, nl])
        nmodes = fsky*(2*ls+1)*dl
        for i1, i2, icl, ni1, ni2, clit in self._get_indices(nt):
            for j1, j2, jcl, nj1, nj2, cljt in self._get_indices(nt):
                cli1j1 = cls[i1, j1]
                cli1j2 = cls[i1, j2]
                cli2j1 = cls[i2, j1]
                cli2j2 = cls[i2, j2]
                cov[icl, :, jcl, :] = np.diag((cli1j1*cli2j2 +
                                               cli1j2*cli2j1)/nmodes)
        if unwrap:
            cov = cov.reshape([ncl*nl, ncl*nl])
        return cov

    def _get_tracer_name(self, i):
        """ Returns tracer name given its index
        """
        # Clustering first, then shear
        if i < self.n_cl:
            return f'cl{i+1}'
        else:
            j = i - self.n_cl
            return f'sh{j+1}'

    def _get_cl_type(self, i, j):
        """ Returns power spectrum type given
        tracer indices.
        """
        if i < self.n_cl:
            if j < self.n_cl:
                return 'cl_00'
            else:
                return 'cl_0e'
        else:
            if j < self.n_cl:
                return 'cl_0e'
            else:
                return 'cl_ee'

    def _get_indices(self, nt):
        """ Iterator through all bin pairs
        """
        icl = 0
        for i1 in range(nt):
            for i2 in range(i1, nt):
                yield (i1, i2, icl,
                       self._get_tracer_name(i1),
                       self._get_tracer_name(i2),
                       self._get_cl_type(i1, i2))
                icl += 1

    def _get_nls(self):
        """ Computes matrix of noise power spectra
        """
        ll = self._get_ell_sampling()
        n_tot = self.n_cl + self.n_sh
        nls = np.zeros([n_tot, n_tot, ll['n_bpw']])
        sgamma = self.c.get('e_rms', 0.28)
        # Clustering first
        for i in range(self.n_cl):
            nls[i, i, :] = 1./self.ndens_cl[i]
        # Then shear
        for i in range(self.n_sh):
            nls[i+self.n_cl, i+self.n_cl, :] = sgamma**2/self.ndens_sh[i]
        return nls

    def _get_shear_tracers(self):
        """ Generates all shear tracers
        """
        wl_tracers = [ccl.WeakLensingTracer(self.cosmo, (self.z_sh, n))
                for n in self.nz_sh]
        if self.bias_model != 'BACCO':
            return wl_tracers
        else:
            pt_tracers = [pt.PTMatterTracer() for i in range(self.n_sh)]
            return wl_tracers, pt_tracers

    def _get_clustering_tracers(self):
        """ Generates all clustering tracers
        """
        # If linear bias (constant or otherwise), include linear
        # bias as part of the tracer
        if self.bias_model == 'constant':
            bc = self.c['bias'].get('constant_bias', 1.)
            bz = np.ones_like(self.z_cl)*bc
        elif self.bias_model == 'HSC_linear':
            bc = self.c['bias'].get('constant_bias', 0.95)
            bz = bc/ccl.growth_factor(self.cosmo, 1./(1+self.z_cl))
        # For BACCO we need to create PT tracers with bias information for each tracer
        elif self.bias_model == 'BACCO':
            pt_tracers = []
            for i in range(self.n_cl):
                zmean = np.average(self.z_cl, weights=self.nz_cl[i, :])
                b1 = self.c['bias']['bias_params']['cl{}_b1'.format(i+1)]
                b1p = self.c['bias']['bias_params']['cl{}_b1p'.format(i+1)]
                bz = b1 + b1p * (self.z_cl - zmean)
                b2 = self.c['bias']['bias_params']['cl{}_b2'.format(i+1)]
                bs = self.c['bias']['bias_params']['cl{}_bs'.format(i+1)]
                bk2 = self.c['bias']['bias_params'].get('cl{}_bk2'.format(i+1), None)
                b3nl = self.c['bias']['bias_params'].get('cl{}_b3nl'.format(i+1), None)
                pt_tracers.append(pt.PTNumberCountsTracer(b1=(self.z_cl, bz), b2=b2,
                                                          bs=bs, bk2=bk2, b3nl=b3nl))
        else:
            # Otherwise, just set b=1 (bias will be in P(k)s)
            bz = np.ones_like(self.z_cl)
        nc_tracers = [ccl.NumberCountsTracer(self.cosmo, False, (self.z_cl, n),
                                (self.z_cl, bz)) for n in self.nz_cl]
        if self.bias_model != 'BACCO':
            return nc_tracers
        else:
            return nc_tracers, pt_tracers

    def get_b_effective(self, z):
        """ Returns the effective bias at a given redshift
        """
        if self.bias_model == 'constant':
            return self.c['bias'].get('constant_bias', 1.)
        elif self.bias_model == 'HSC_linear':
            return self.c['bias'].get('constant_bias', 0.95)/ccl.growth_factor(self.cosmo, 1./(1+z))
        elif self.bias_model == 'HOD':
            md = ccl.halos.MassDef200m()
            cm = ccl.halos.ConcentrationDuffy08(mdef=md)
            mf = ccl.halos.MassFuncTinker08(self.cosmo, mass_def=md)
            bm = ccl.halos.HaloBiasTinker10(self.cosmo, mass_def=md)
            pg = ccl.halos.HaloProfileHOD(cm, **(self.c['bias']['HOD_params']))
            hmc = ccl.halos.HMCalculator(self.cosmo, mf, bm, md)
            b = ccl.halos.halomod_bias_1pt(self.cosmo, hmc, 1E-4, 1/(1+z), pg, normprof=True)
            return b
        elif self.bias_model == 'Abacus':
            d = np.load('AbacusData/pk2d_abacus.npz')
            gtype = self.c['bias']['galtype']
            ids = d['k_s'] < 0.1
            pkgg = d[f'{gtype}_{gtype}'][:, ids]
            pkmm = d['m_m'][:, ids]
            bz = np.mean(np.sqrt(pkgg/pkmm), axis=1)
            zz = 1./d['a_s']-1
            from scipy.interpolate import interp1d
            bf = interp1d(zz[::-1], bz[::-1], fill_value='extrapolate', bounds_error=False)
            return bf(z)

    def _get_pks(self):
        """ Computes P_gg(k, z) and P_gm(k, z)
        """
        if ((self.bias_model == 'constant') or
                (self.bias_model == 'HSC_linear')):
            # If linear bias, all Pks are just the matter power spectrum
            # (by default in CCL)
            pk_gg = None
            pk_gm = None
        elif self.bias_model == 'HOD':
            # Halo model calculation
            print("Getting HOD Pks")
            md = ccl.halos.MassDef200m()
            cm = ccl.halos.ConcentrationDuffy08(mdef=md)
            mf = ccl.halos.MassFuncTinker08(self.cosmo, mass_def=md)
            bm = ccl.halos.HaloBiasTinker10(self.cosmo, mass_def=md)
            pg = ccl.halos.HaloProfileHOD(cm, **(self.c['bias']['HOD_params']))
            pgg = ccl.halos.Profile2ptHOD()
            pm = ccl.halos.HaloProfileNFW(cm)
            hmc = ccl.halos.HMCalculator(self.cosmo, mf, bm, md)
            k_s = np.geomspace(1E-4, 1E2, 512)
            lk_s = np.log(k_s)
            a_s = 1./(1+np.linspace(0., 2., 30)[::-1])

            def alpha_HMCODE(a):
                return 0.7

            def k_supress(a):
                return 0.001

            pk_gg = ccl.halos.halomod_Pk2D(self.cosmo, hmc, pg, prof_2pt=pgg,
                                           prof2=pg,
                                           normprof1=True, normprof2=True,
                                           lk_arr=lk_s, a_arr=a_s,
                                           smooth_transition=alpha_HMCODE,
                                           supress_1h=k_supress)
            pk_gm = ccl.halos.halomod_Pk2D(self.cosmo, hmc, pg,
                                           prof2=pm,
                                           normprof1=True, normprof2=True,
                                           lk_arr=lk_s, a_arr=a_s,
                                           smooth_transition=alpha_HMCODE,
                                           supress_1h=k_supress)
        elif self.bias_model == 'Abacus' or self.bias_model == 'Abacus_unnorm' or self.bias_model == 'Abacus_ggl+wl=norm':
            # If using Abacus, read all the smooth power spectra
            # (generated in AbacusData.ipynb), and interpolate
            # in k and a.
            print("Getting Abacus Pks")
            d = np.load('AbacusData/pk2d_abacus.npz')
            gtype = self.c['bias']['galtype']

            # The red-red Pk is super noisy at z>1.7, so we remove that
            if gtype in ['red', 'red_AB']:
                imin = 3
            else:
                imin = 0

            # Interpolation done internally in the Pk2D objects
            pk_gg = ccl.Pk2D(a_arr=d['a_s'][imin:], lk_arr=np.log(d['k_s']),
                             pk_arr=np.log(d[f'{gtype}_{gtype}'][imin:, :]),
                             is_logp=True)
            pgm = d[f'{gtype}_m']
            if self.bias_model == 'Abacus_ggl+wl=norm':
                phf = np.array([ccl.nonlin_matter_power(self.cosmo, d['k_s'], a)
                          for a in d['a_s']])
                pmm = d[f'm_m']
                pgm *= np.sqrt(phf/pmm)
            pk_gm = ccl.Pk2D(a_arr=d['a_s'], lk_arr=np.log(d['k_s']),
                             pk_arr=np.log(pgm),
                             is_logp=True)
            pk_mm = ccl.Pk2D(a_arr=d['a_s'], lk_arr=np.log(d['k_s']),
                             pk_arr=np.log(d[f'm_m']),
                             is_logp=True)
        else:
            raise NotImplementedError("Bias model " + self.bias_model +
                                      " not implemented.")
        if 'Abacus' not in self.bias_model:
            return {'gg': pk_gg,
                    'gm': pk_gm}
        else:
            return {'gg': pk_gg,
                    'gm': pk_gm,
                    'mm': pk_mm}

    def _get_cls(self):
        """ Computes all angular power spectra
        """
        if self.bias_model != 'BACCO':
            # Get P(k)s
            pks = self._get_pks()
            # Get clustering tracers
            t_cl = self._get_clustering_tracers()
            # Get shear tracers
            t_sh = self._get_shear_tracers()
        else:
            # Get clustering and PT tracers
            t_cl, pt_cl = self._get_clustering_tracers()
            # Get shear tracers
            t_sh, pt_sh = self._get_shear_tracers()
            ptc = BACCOCalculator(log10k_min=np.log10(1e-2 * self.c['cosmology']['h']),
                                  log10k_max=np.log10(0.75 * self.c['cosmology']['h']),
                                  nk_per_decade=20, h=self.c['cosmology']['h'], k_filter=self.c['bias']['k_filter'])
            ptc.update_pk(self.cosmo)
            pts = pt_cl + pt_sh

        # Ell sampling
        ll = self._get_ell_sampling()
        ts = t_cl + t_sh
        n_tot = self.n_cl + self.n_sh

        # Loop over all tracer pairs
        cls = np.zeros([n_tot, n_tot, ll['n_bpw']])
        for i1, t1 in enumerate(ts):
            for i2, t2 in enumerate(ts):
                if i2 < i1:
                    continue
                if self.bias_model != 'BACCO':
                    if self.bias_model != 'Abacus_unnorm':
                        pk = None
                    else:
                        pk = pks['mm']
                    if i1 < self.n_cl:
                        if i2 < self.n_cl:
                            pk = pks['gg']  # gg case
                        else:
                            pk = pks['gm']  # gm case
                else:
                    if i1 >= self.n_cl and i2 >= self.n_cl:
                        pk = None
                    else:
                        pk = get_bacco_pk2d(self.cosmo, pts[i1], tracer2=pts[i2], ptc=ptc)
                # Limber integral
                cl = ccl.angular_cl(self.cosmo, t1, t2, ll['ls'],
                                    p_of_k_a=pk)
                # Bandpower window convolution
                clb = np.dot(ll['bpws'], cl)
                cls[i1, i2, :] = clb
                if i1 != i2:
                    cls[i2, i1, :] = clb
        return cls

    def _get_ell_sampling(self):
        """ Defines the ell sampling of the data vector.
        We use linear sampling with separation
        `dl_linear` = 10 up to a given `ell_linear`, and
        then switch to log spacing with `nl_per_decade`=10
        ells per dex. The value of `ell_linear` is such that
        the separation between adjacent ells after
        `ell_linear` using log sampling is larger or equal
        to `d_ell_linear`. We start at ell=2 and stop at
        ell=5000.
        """
        if self.ll is None:
            # First work out the ell edges
            dl_linear = 10
            nl_per_decade = 10
            dlogl = 1./nl_per_decade
            l_edges = [2]
            l_last = l_edges[0]
            while l_last < self.lmax:
                dl_log = l_last*(10**dlogl-1)
                if dl_log < dl_linear:
                    l_last += dl_linear
                else:
                    l_last += dl_log
                l_edges.append(int(l_last))
            l_edges = np.array(l_edges)

            # Compute bandpower window functions.
            # Assumed top-hat weighted by 2*l+1.
            n_bpw = len(l_edges)-1
            l_all = np.arange(l_edges[-1])
            bpw_windows = np.zeros([n_bpw, l_edges[-1]])
            l_mean = np.zeros(n_bpw)
            for i in range(n_bpw):
                nells = l_edges[i+1] - l_edges[i]
                msk = (l_all < l_edges[i+1]) & (l_all >= l_edges[i])
                bpw_windows[i, msk] = 1./nells
                l_mean[i] = np.average(l_all[msk],
                                       weights=2*l_all[msk]+1.)
            self.ll = {'ls': l_all,
                       'n_bpw': n_bpw,
                       'edges': l_edges,
                       'd_ell': np.diff(l_edges),
                       'mean': l_mean,
                       'bpws': bpw_windows}
        return self.ll

    def get_sacc_file(self):
        """ Generates sacc file containing full
        data vector, N(z)s, and covariance matrix.
        """
        import sacc
        s = sacc.Sacc()

        # Tracers
        print("Tracers")
        for i, n in enumerate(self.nz_cl):
            s.add_tracer('NZ', f'cl{i+1}',
                         quantity='galaxy_density',
                         spin=0, z=self.z_cl, nz=n)
            z_eff = np.sum(n*self.z_cl)/np.sum(n)
            print(self.get_b_effective(z_eff))
        for i, n in enumerate(self.nz_sh):
            s.add_tracer('NZ', f'sh{i+1}',
                         quantity='galaxy_shear',
                         spin=2, z=self.z_sh, nz=n)

        # Bandpower windows
        print("Windows")
        ll = self._get_ell_sampling()
        wins = sacc.BandpowerWindow(ll['ls'], ll['bpws'].T)

        # Cls
        print("Cls")
        sl = self._get_cls()
        for i1, i2, icl, n1, n2, clt in self._get_indices(self.n_cl+self.n_sh):
            s.add_ell_cl(clt, n1, n2, ll['mean'], sl[i1, i2], window=wins)

        # Covariance
        print("Cov")
        nl = self._get_nls()
        cov = self._get_covariance(sl+nl, unwrap=True)
        s.add_covariance(cov)

        if self.c.get('add_noise', False):
            s.mean = np.random.multivariate_normal(s.mean, cov)

        # Save
        print("Write")
        s.save_fits(self.c['sacc_name'], overwrite=True)
        return s

    def save_config(self):
        """ Saves yaml file used to generate these data.
        """
        import yaml

        with open(self.c['sacc_name']+'.yml', 'w') as outfile:
            yaml.dump(self.c, outfile, default_flow_style=False)


# Cosmological parameters for non-Abacus datasets
cospar = {'Omega_c': 0.25,
          'Omega_b': 0.05,
          'h': 0.7,
          'sigma8': 0.81,
          'n_s': 0.96,
          'w0': -1,
          'transfer_function': 'eisenstein_hu'}

# ###
# # 1. Constant linear bias
# # Same clustering and shear bins
# config = {'ndens_sh': 27.,
#           'ndens_cl': 27.,
#           'dNdz_file': 'data/dNdz_shear_shear.npz',
#           'e_rms': 0.28,
#           'cosmology': cospar,
#           'bias': {'model': 'constant',
#                    'constant_bias': 1.},
#           'sacc_name': 'fid_shear_const.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# # Red clustering
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': cospar,
#           'bias': {'model': 'constant',
#                    'constant_bias': 2.},
#           'sacc_name': 'fid_red_const.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# ###
#
#
# ###
# # 2. Evolving linear bias (a la HSC)
# # Same clustering and shear bins
# config = {'ndens_sh': 27.,
#           'ndens_cl': 27.,
#           'dNdz_file': 'data/dNdz_shear_shear.npz',
#           'e_rms': 0.28,
#           'cosmology': cospar,
#           'bias': {'model': 'HSC_linear',
#                    'constant_bias': 0.95},
#           'sacc_name': 'fid_HSC_linear.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# # Red clustering (2001.06018)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': cospar,
#           'bias': {'model': 'HSC_linear',
#                    'constant_bias': 1.5},
#           'sacc_name': 'fid_red_linear.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# ###
#
#
# ###
# # 3. HOD
# # HSC HOD parameters (Nicola et al.)
# # Same clustering and shear bins
# config = {'ndens_sh': 27.,
#           'ndens_cl': 27.,
#           'dNdz_file': 'data/dNdz_shear_shear.npz',
#           'e_rms': 0.28,
#           'cosmology': cospar,
#           'bias': {'model': 'HOD',
#                    'HOD_params': {'lMmin_0': 11.88,
#                                   'lMmin_p': -0.5,
#                                   'siglM_0': 0.4,
#                                   'siglM_p': 0.,
#                                   'lM0_0': 11.88,
#                                   'lM0_p': -0.5,
#                                   'lM1_0': 13.08,
#                                   'lM1_p': 0.9,
#                                   'a_pivot': 1./(1+0.65)}},
#           'sacc_name': 'fid_HSC_HOD.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# # LRGs (from 2001.06018)
# # Red clustering
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': cospar,
#           'bias': {'model': 'HOD',
#                    'HOD_params': {'lMmin_0': 12.95,
#                                   'lMmin_p': -2.0,
#                                   'siglM_0': 0.25,
#                                   'siglM_p': 0.,
#                                   'lM0_0': 12.3,
#                                   'lM0_p': 0.,
#                                   'lM1_0': 14.0,
#                                   'lM1_p': -1.5,
#                                   'alpha_0': 1.32,
#                                   'alpha_p': 0.,
#                                   'a_pivot': 1./(1+0.65)}},
#           'sacc_name': 'fid_red_HOD.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# ###
#
#
# ###
# # 4. From Abacus
# # HSC (same HOD params)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 27.,
#           'dNdz_file': 'data/dNdz_shear_shear.npz',
#           'e_rms': 0.28,
#           'cosmology': 'Abacus',
#           'bias': {'model': 'Abacus',
#                    'galtype': 'all'},
#           'sacc_name': 'abacus_HSC_abacus.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# # HSC, red nzs (same HOD params)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': 'Abacus',
#           'bias': {'model': 'Abacus',
#                    'galtype': 'all'},
#           'sacc_name': 'abacus_HSC_abacus_bins=red_nd=red.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# # Red (same HOD params)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': 'Abacus',
#           'bias': {'model': 'Abacus',
#                    'galtype': 'red'},
#           'sacc_name': 'abacus_red_abacus.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# Red Y1 errors (same HOD params)
config = {'ndens_sh': 10.,
          'ndens_cl': 1.5,
          'dNdz_file': 'data/dNdz_shear_red.npz',
          'e_rms': 0.28,
          'cosmology': 'Abacus',
          'bias': {'model': 'Abacus_unnorm',
                   'galtype': 'red'},
          'sacc_name': 'abacus_red_unnorm_err=Y1_abacus.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
# # Red unnorm (same HOD params)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': 'Abacus',
#           'bias': {'model': 'Abacus_unnorm',
#                    'galtype': 'red'},
#           'sacc_name': 'abacus_red_unnorm_abacus.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# Red all norm (same HOD params)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': 'Abacus',
#           'bias': {'model': 'Abacus_ggl+wl=norm',
#                    'galtype': 'red'},
#           'sacc_name': 'abacus_red_ggl+wl=norm_abacus.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# Red spectro (same HOD params)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_spectro_red.npz',
#           'e_rms': 0.28,
#           'cosmology': 'Abacus',
#           'bias': {'model': 'Abacus_unnorm',
#                    'galtype': 'red'},
#           'sacc_name': 'abacus_red_spectro_unnorm_abacus.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# Red (with assembly bias)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': 'Abacus',
#           'bias': {'model': 'Abacus',
#                    'galtype': 'red_AB'},
#           'sacc_name': 'abacus_red_AB_abacus.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
# ###
# 5. From BACCO
# Red spectro (same HOD params)
# config = {'ndens_sh': 27.,
#           'ndens_cl': 4.,
#           'dNdz_file': 'data/dNdz_shear_red.npz',
#           'e_rms': 0.28,
#           'cosmology': 'Abacus',
#           'bias': {'model': 'BACCO',
#                    'k_filter': None,
#                    'bias_params': {'cl1_b1': 2.,
#                                    'cl1_b1p': 1.,
#                                    'cl1_b2': 1.,
#                                    'cl1_bs': 1.,
#                                    'cl1_bk2': 1.,
#                                    'cl2_b1': 2.,
#                                    'cl2_b1p': 1.,
#                                    'cl2_b2': 1.,
#                                    'cl2_bs': 1.,
#                                    'cl2_bk2': 1.,
#                                    'cl3_b1': 2.,
#                                    'cl3_b1p': 1.,
#                                    'cl3_b2': 1.,
#                                    'cl3_bs': 1.,
#                                    'cl3_bk2': 1.,
#                                    'cl4_b1': 2.,
#                                    'cl4_b1p': 1.,
#                                    'cl4_b2': 1.,
#                                    'cl4_bs': 1.,
#                                    'cl4_bk2': 1.,
#                                    'cl5_b1': 2.,
#                                    'cl5_b1p': 1.,
#                                    'cl5_b2': 1.,
#                                    'cl5_bs': 1.,
#                                    'cl5_bk2': 1.,
#                                    'cl6_b1': 2.,
#                                    'cl6_b1p': 1.,
#                                    'cl6_b2': 1.,
#                                    'cl6_bs': 1.,
#                                    'cl6_bk2': 1.
#                    }},
#           'sacc_name': 'fid_red_BACCO.fits'}
# if not os.path.isfile(config['sacc_name']):
#     d = DataGenerator(config)
#     s = d.get_sacc_file()
#     d.save_config()
#     print(" ")
#
# # Tarball
# os.system('tar -cpzf data_DESCBiasChallenge.tar.gz *.fits *.fits.yml README_data.md')
