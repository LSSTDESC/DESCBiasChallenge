import numpy as np
import pyccl as ccl
from scipy.integrate import simps
import os


class DataGenerator(object):
    lmax = 5000

    def __init__(self, config):
        self.c = config
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
        if 'cosmology' in self.c:
            self.cosmo = ccl.Cosmology(**(self.c['cosmology']))
        else:
            self.cosmo = ccl.CosmologyVanillaLCDM()
        ccl.sigma8(self.cosmo)
        self.bias_model = self.c['bias']['model']
        self.ll = None

    def get_covariance(self, cls, unwrap=True):
        nt, _, nl = cls.shape
        ll = self.get_ell_sampling()
        ls = ll['mean']
        dl = ll['d_ell']
        fsky = self.c.get('fsky', 0.4)
        ncl = (nt*(nt+1)) // 2
        cov = np.zeros([ncl, nl, ncl, nl])
        nmodes = fsky*(2*ls+1)*dl
        for i1, i2, icl, ni1, ni2, clit in self.get_indices(nt):
            for j1, j2, jcl, nj1, nj2, cljt in self.get_indices(nt):
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
        if i < self.n_cl:
            return f'cl{i+1}'
        else:
            j = i - self.n_cl
            return f'sh{j+1}'

    def _get_cl_type(self, i, j):
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

    def get_indices(self, nt):
        icl = 0
        for i1 in range(nt):
            for i2 in range(i1, nt):
                yield (i1, i2, icl,
                       self._get_tracer_name(i1),
                       self._get_tracer_name(i2),
                       self._get_cl_type(i1, i2))
                icl += 1

    def get_nls(self):
        ll = self.get_ell_sampling()
        n_tot = self.n_cl + self.n_sh
        nls = np.zeros([n_tot, n_tot, ll['n_bpw']])
        sgamma = self.c.get('e_rms', 0.28)
        for i in range(self.n_cl):
            nls[i, i, :] = 1./self.ndens_cl[i]
        for i in range(self.n_sh):
            nls[i+self.n_cl, i+self.n_cl, :] = sgamma**2/self.ndens_sh[i]
        return nls

    def get_shear_tracers(self):
        return [ccl.WeakLensingTracer(self.cosmo, (self.z_sh, n))
                for n in self.nz_sh]

    def get_clustering_tracers(self):
        if self.bias_model == 'constant':
            bc = self.c['bias'].get('constant_bias', 1.)
            bz = np.ones_like(self.z_cl)*bc
        elif self.bias_model == 'HSC_linear':
            bc = self.c['bias'].get('constant_bias', 0.95)
            bz = bc/ccl.growth_factor(self.cosmo, 1./(1+self.z_cl))
        else:
            bz = np.ones_like(self.z_cl)
        return [ccl.NumberCountsTracer(self.cosmo, False, (self.z_cl, n),
                                       (self.z_cl, bz))
                for n in self.nz_cl]

    def get_pks(self):
        if ((self.bias_model == 'constant') or
                (self.bias_model == 'HSC_linear')):
            pk_gg = None
            pk_gm = None
        elif self.bias_model == 'HOD':
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
        else:
            raise NotImplementedError("Bias model " + self.bias_model +
                                      " not implemented.")
        return {'gg': pk_gg,
                'gm': pk_gm}

    def get_cls(self):
        pks = self.get_pks()
        t_cl = self.get_clustering_tracers()
        t_sh = self.get_shear_tracers()
        ll = self.get_ell_sampling()
        ts = t_cl + t_sh
        n_tot = self.n_cl + self.n_sh

        cls = np.zeros([n_tot, n_tot, ll['n_bpw']])
        for i1, t1 in enumerate(ts):
            for i2, t2 in enumerate(ts):
                if i2 < i1:
                    continue
                pk = None
                if i1 < self.n_cl:
                    if i2 < self.n_cl:
                        pk = pks['gg']
                    else:
                        pk = pks['gm']
                cl = ccl.angular_cl(self.cosmo, t1, t2, ll['ls'], p_of_k_a=pk)
                clb = np.dot(ll['bpws'], cl)
                cls[i1, i2, :] = clb
                if i1 != i2:
                    cls[i2, i1, :] = clb
        return cls

    def get_ell_sampling(self):
        if self.ll is None:
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
        import sacc
        s = sacc.Sacc()

        # Tracers
        print("Tracers")
        for i, n in enumerate(self.nz_cl):
            s.add_tracer('NZ', f'cl{i+1}',
                         quantity='galaxy_density',
                         spin=0, z=self.z_cl, nz=n)
        for i, n in enumerate(self.nz_sh):
            s.add_tracer('NZ', f'sh{i+1}',
                         quantity='galaxy_shear',
                         spin=2, z=self.z_sh, nz=n)

        # Bandpower windows
        print("Windows")
        ll = self.get_ell_sampling()
        wins = sacc.BandpowerWindow(ll['ls'], ll['bpws'].T)

        # Cls
        print("Cls")
        sl = self.get_cls()
        for i1, i2, icl, n1, n2, clt in self.get_indices(self.n_cl+self.n_sh):
            s.add_ell_cl(clt, n1, n2, ll['mean'], sl[i1, i2], window=wins)

        # Covariance
        print("Cov")
        nl = self.get_nls()
        cov = self.get_covariance(sl+nl, unwrap=True)
        s.add_covariance(cov)

        if self.c.get('add_noise', False):
            s.mean = np.random.multivariate_normal(s.mean, cov)

        # Save
        print("Write")
        s.save_fits(self.c['sacc_name'], overwrite=True)
        return s

    def save_config(self):
        import yaml

        with open(self.c['sacc_name']+'.yml', 'w') as outfile:
            yaml.dump(self.c, outfile, default_flow_style=False)


cospar = {'Omega_c': 0.25,
          'Omega_b': 0.05,
          'h': 0.7,
          'sigma8': 0.81,
          'n_s': 0.96,
          'w0': -1,
          'transfer_function': 'eisenstein_hu'}

# Constant linear bias
# Same clustering and shear bins
config = {'ndens_sh': 27.,
          'ndens_cl': 27.,
          'dNdz_file': 'data/dNdz_shear_shear.npz',
          'e_rms': 0.28,
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
          'cosmology': cospar,
          'bias': {'model': 'constant',
                   'constant_bias': 2.},
          'sacc_name': 'fid_red_const.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")


# Evolving linear bias (a la HSC)
# Same clustering and shear bins
config = {'ndens_sh': 27.,
          'ndens_cl': 27.,
          'dNdz_file': 'data/dNdz_shear_shear.npz',
          'e_rms': 0.28,
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
          'bias': {'model': 'HSC_linear',
                   'constant_bias': 1.5},
          'sacc_name': 'fid_red_linear.fits'}
if not os.path.isfile(config['sacc_name']):
    d = DataGenerator(config)
    s = d.get_sacc_file()
    d.save_config()
    print(" ")


# HOD (a la HSC)
# Same clustering and shear bins
config = {'ndens_sh': 27.,
          'ndens_cl': 27.,
          'dNdz_file': 'data/dNdz_shear_shear.npz',
          'e_rms': 0.28,
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

# HOD (LRGs from 2001.06018)
# Red clustering
config = {'ndens_sh': 27.,
          'ndens_cl': 4.,
          'dNdz_file': 'data/dNdz_shear_red.npz',
          'e_rms': 0.28,
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
