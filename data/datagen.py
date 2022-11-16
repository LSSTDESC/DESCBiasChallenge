import numpy as np
import pyccl as ccl
from scipy.integrate import simps
from scipy.interpolate import interp1d
import os


def get_abacus_cosmo():
    """ Returns the Abacus cosmology as a CCL object
    """
    omega_b = 0.02237
    omega_cdm = 0.12
    h = 0.6736
    A_s = 2.083e-09
    n_s = 0.9649
    alpha_s = 0.0
    N_ur = 2.0328
    N_ncdm = 1.0
    omega_ncdm = 0.0006442
    w0_fld = -1.0
    wa_fld = 0.0
    cosmo = ccl.Cosmology(Omega_c=omega_cdm/h**2,
                          Omega_b=omega_b/h**2,
                          h=h, A_s=A_s, n_s=n_s,
                          m_nu=0.06)
    cosmo.compute_linear_power()
    cosmo.compute_nonlin_power()
    return cosmo


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
                    self.cosmo = get_abacus_cosmo()
                else:
                    raise ValueError('Unknown cosmology')
        else:
            self.cosmo = ccl.CosmologyVanillaLCDM()
        ccl.sigma8(self.cosmo)

        # Bias model
        self.bias_model = self.c['bias']['model']
        self.ll = None
        # n_tot = n_cl + n_sh + cmbk
        self.n_tot = self.n_cl + self.n_sh + 1
        self.cmbk_nl, self.cmbk_nlb = None, None

    def _get_cmbk_nl(self,):
        if self.cmbk_nl is None:
            fname = self.c['CMBk_noise']
            ell, nl = np.loadtxt(fname, unpack=True)

            ll = self._get_ell_sampling()
            nl = interp1d(ell, nl, bounds_error=False,
                          fill_value=(nl[0], nl[-1]))(ll['ls'])

            self.cmbk_nl = nl
            self.cmbk_nlb = np.dot(ll['bpws'], nl)

        return self.cmbk_nl, self.cmbk_nlb

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
        elif i < self.n_cl + self.n_sh:
            j = i - self.n_cl
            return f'sh{j+1}'
        else:
            return 'cmbk'

    def _get_cl_type(self, i, j):
        """ Returns power spectrum type given
        tracer indices.
        """
        if i < self.n_cl:
            if (j < self.n_cl) or (j == self.n_tot - 1):
                return 'cl_00'
            else:
                return 'cl_0e'
        elif i < self.n_cl + self.n_sh:
            if (j < self.n_cl) or (j == self.n_tot - 1):
                return 'cl_0e'
            else:
                return 'cl_ee'
        else:
            # Else = cmbk
            return 'cl_00'

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
        nls = np.zeros([self.n_tot, self.n_tot, ll['n_bpw']])
        sgamma = self.c.get('e_rms', 0.28)
        # Clustering first
        for i in range(self.n_cl):
            nls[i, i, :] = 1./self.ndens_cl[i]
        # Then shear
        for i in range(self.n_sh):
            nls[i+self.n_cl, i+self.n_cl, :] = sgamma**2/self.ndens_sh[i]
        # Then CMBk
        nls[-1, -1, :] = self._get_cmbk_nl()[1]
        return nls

    def _get_shear_tracers(self):
        """ Generates all shear tracers
        """
        return [ccl.WeakLensingTracer(self.cosmo, (self.z_sh, n))
                for n in self.nz_sh]

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
        else:
            # Otherwise, just set b=1 (bias will be in P(k)s)
            bz = np.ones_like(self.z_cl)
        return [ccl.NumberCountsTracer(self.cosmo, False, (self.z_cl, n),
                                       (self.z_cl, bz))
                for n in self.nz_cl]

    def _get_cmbk_tracer(self):
        """ Generates the CMBk tracer
        """
        return [ccl.CMBLensingTracer(self.cosmo, 1100)]

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
        elif self.bias_model == 'Abacus':
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
            pk_gm = ccl.Pk2D(a_arr=d['a_s'], lk_arr=np.log(d['k_s']),
                             pk_arr=np.log(d[f'{gtype}_m']),
                             is_logp=True)
        else:
            raise NotImplementedError("Bias model " + self.bias_model +
                                      " not implemented.")
        return {'gg': pk_gg,
                'gm': pk_gm}

    def _get_cls(self):
        """ Computes all angular power spectra
        """
        # Get P(k)s
        pks = self._get_pks()
        # Get clustering tracers
        t_cl = self._get_clustering_tracers()
        # Get shear tracers
        t_sh = self._get_shear_tracers()
        # Get CMBk tracer
        t_cmbk = self._get_cmbk_tracer()
        # Ell sampling
        ll = self._get_ell_sampling()
        ts = t_cl + t_sh + t_cmbk

        # Loop over all tracer pairs
        cls = np.zeros([self.n_tot, self.n_tot, ll['n_bpw']])
        for i1, t1 in enumerate(ts):
            for i2, t2 in enumerate(ts):
                if i2 < i1:
                    continue
                pk = None
                if i1 < self.n_cl:
                    if i2 < self.n_cl:
                        pk = pks['gg']  # gg case
                    else:
                        pk = pks['gm']  # gm case
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

        ll = self._get_ell_sampling()
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

        nl = self._get_cmbk_nl()[0]
        s.add_tracer('Map', 'cmbk', quantity='cmb_convergence', spin=0,
                     ell=ll['ls'], beam=np.ones_like(nl),
                     beam_extra={'nl': nl})


        # Bandpower windows
        print("Windows")
        wins = sacc.BandpowerWindow(ll['ls'], ll['bpws'].T)

        # Cls
        print("Cls")
        sl = self._get_cls()
        for i1, i2, icl, n1, n2, clt in self._get_indices(self.n_tot):
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

###
# 1. Constant linear bias
# Same clustering and shear bins
config = {'ndens_sh': 27.,
          'ndens_cl': 27.,
          'dNdz_file': 'data/dNdz_shear_shear.npz',
          'e_rms': 0.28,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'cosmology': cospar,
          'bias': {'model': 'constant',
                   'constant_bias': 1.},
          'sacc_name': 'fid_shear_const.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
# Red clustering
config = {'ndens_sh': 27.,
          'ndens_cl': 4.,
          'dNdz_file': 'data/dNdz_shear_red.npz',
          'e_rms': 0.28,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'cosmology': cospar,
          'bias': {'model': 'constant',
                   'constant_bias': 2.},
          'sacc_name': 'fid_red_const.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
###


###
# 2. Evolving linear bias (a la HSC)
# Same clustering and shear bins
config = {'ndens_sh': 27.,
          'ndens_cl': 27.,
          'dNdz_file': 'data/dNdz_shear_shear.npz',
          'e_rms': 0.28,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'cosmology': cospar,
          'bias': {'model': 'HSC_linear',
                   'constant_bias': 0.95},
          'sacc_name': 'fid_HSC_linear.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
# Red clustering (2001.06018)
config = {'ndens_sh': 27.,
          'ndens_cl': 4.,
          'dNdz_file': 'data/dNdz_shear_red.npz',
          'e_rms': 0.28,
          'cosmology': cospar,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'bias': {'model': 'HSC_linear',
                   'constant_bias': 1.5},
          'sacc_name': 'fid_red_linear.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
###


###
# 3. HOD
# HSC HOD parameters (Nicola et al.)
# Same clustering and shear bins
config = {'ndens_sh': 27.,
          'ndens_cl': 27.,
          'dNdz_file': 'data/dNdz_shear_shear.npz',
          'e_rms': 0.28,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'cosmology': cospar,
          'bias': {'model': 'HOD',
                   'HOD_params': {'lMmin_0': 11.88,
                                  'lMmin_p': -0.5,
                                  'siglM_0': 0.4,
                                  'siglM_p': 0.,
                                  'lM0_0': 11.88,
                                  'lM0_p': -0.5,
                                  'lM1_0': 13.08,
                                  'lM1_p': 0.9,
                                  'a_pivot': 1./(1+0.65)}},
          'sacc_name': 'fid_HSC_HOD.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
# LRGs (from 2001.06018)
# Red clustering
config = {'ndens_sh': 27.,
          'ndens_cl': 4.,
          'dNdz_file': 'data/dNdz_shear_red.npz',
          'e_rms': 0.28,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'cosmology': cospar,
          'bias': {'model': 'HOD',
                   'HOD_params': {'lMmin_0': 12.95,
                                  'lMmin_p': -2.0,
                                  'siglM_0': 0.25,
                                  'siglM_p': 0.,
                                  'lM0_0': 12.3,
                                  'lM0_p': 0.,
                                  'lM1_0': 14.0,
                                  'lM1_p': -1.5,
                                  'alpha_0': 1.32,
                                  'alpha_p': 0.,
                                  'a_pivot': 1./(1+0.65)}},
          'sacc_name': 'fid_red_HOD.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
###


###
# 4. From Abacus
# HSC (same HOD params)
config = {'ndens_sh': 27.,
          'ndens_cl': 27.,
          'dNdz_file': 'data/dNdz_shear_shear.npz',
          'e_rms': 0.28,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'cosmology': 'Abacus',
          'bias': {'model': 'Abacus',
                   'galtype': 'all'},
          'sacc_name': 'abacus_HSC_abacus.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
# Red (same HOD params)
config = {'ndens_sh': 27.,
          'ndens_cl': 4.,
          'dNdz_file': 'data/dNdz_shear_red.npz',
          'e_rms': 0.28,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'cosmology': 'Abacus',
          'bias': {'model': 'Abacus',
                   'galtype': 'red'},
          'sacc_name': 'abacus_red_abacus.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
# Red (with assembly bias)
config = {'ndens_sh': 27.,
          'ndens_cl': 4.,
          'dNdz_file': 'data/dNdz_shear_red.npz',
          'e_rms': 0.28,
          'CMBk_noise': 'data/kappa_noise_cmbs4_deproj0.txt',
          'cosmology': 'Abacus',
          'bias': {'model': 'Abacus',
                   'galtype': 'red_AB'},
          'sacc_name': 'abacus_red_AB_abacus.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")
###

# Tarball
os.system('tar -cpzf data_DESCBiasChallenge.tar.gz *.fits *.fits.yml README_data.md')
