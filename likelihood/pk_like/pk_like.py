import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
# import pyccl.nl_pt as pt
import sys
sys.path.append('../likelihood/cl_like')
import tracers as pt
from lpt import LPTCalculator, get_lpt_pk2d
from ept import EPTCalculator, get_ept_pk2d
from bacco import BACCOCalculator, get_bacco_pk2d
from heft import HEFTCalculator, get_anzu_pk2d
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from anzu.emu_funcs import LPTEmulator
import baccoemu_beta as baccoemu


class PkLike(Likelihood):
    # All parameters starting with this will be
    # identified as belonging to this stage.
    input_params_prefix: str = ""
    # Input sacc file
    input_file: str = ""
    # IA model name. Currently all of these are
    # just flags, but we could turn them into
    # homogeneous systematic classes.
    ia_model: str = "IANone"
    # N(z) model name
    nz_model: str = "NzNone"
    # b(z) model name
    bz_model: str = "BzNone"
    # Shape systamatics
    shape_model: str = "ShapeNone"
    # List of bin names
    bins: list = []
    # List of default settings (currently only scale cuts)
    defaults: dict = {}
    # List of two-point functions that make up the data vector
    twopoints: list = []
    # Low-pass filter for PT
    k_pt_filter: float = 0.

    def initialize(self):
        # Read SACC file
        self._read_data()
        # Initialize emu to train it once
        if self.bz_model == 'anzu':
            self.emu = LPTEmulator(kecleft=True, extrap=False)
        if self.bz_model == 'BACCO':
            self.emu = baccoemu.Lbias_expansion()

    def _read_data(self):
        """
        Reads sacc file
        Selects relevant data.
        Applies scale cuts
        Reads tracer metadata (N(z))
        Reads covariance
        """
        import sacc

        def get_suffix_for_tr(tr):
            q = tr.quantity
            if (q == 'galaxy_density') or (q == 'cmb_convergence'):
                return '0'
            elif q == 'galaxy_shear':
                return 'e'
            else:
                raise ValueError(f'dtype not found for quantity {q}')

        s = sacc.Sacc.load_fits(self.input_file)
        self.bin_properties = {}
        kmax_default = self.defaults.get('kmax', 0.1)
        for b in self.bins:
            if b['name'] not in s.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b['name'])
            t = s.tracers[b['name']]
            zmean = np.average(t.z, weights=t.nz)
            self.bin_properties[b['name']] = {'z_fid': t.z,
                                              'nz_fid': t.nz,
                                              'zmean_fid': zmean}
            # Ensure all tracers have kmin
            if b['name'] not in self.defaults:
                self.defaults[b['name']] = {}
                self.defaults[b['name']]['kmin'] = self.defaults['kmin']

            # Give galaxy clustering an kmax
            if t.quantity == 'galaxy_density':
                # Get lmax from kmax for galaxy clustering
                if 'kmax' not in self.defaults[b['name']]:
                    self.defaults[b['name']]['kmax'] = kmax_default

        # First check which parts of the data vector to keep
        indices = []
        for pk in self.twopoints:
            tn1, tn2 = pk['bins']
            kmin = np.max([self.defaults[tn1].get('kmin', 2),
                           self.defaults[tn2].get('kmin', 2)])
            kmax = np.min([self.defaults[tn1].get('kmax', 1E30),
                           self.defaults[tn2].get('kmax', 1E30)])
            # Get the suffix for both tracers
            pk_name1 = get_suffix_for_tr(s.tracers[tn1])
            pk_name2 = get_suffix_for_tr(s.tracers[tn2])
            ind = s.indices('cl_%s%s' % (pk_name1, pk_name2), (tn1, tn2),
                            ell__gt=kmin, ell__lt=kmax)
            indices += list(ind)
        s.keep_indices(np.array(indices))

        # Now collect information about those
        # and put the C_ells in the right order
        indices = []
        self.pk_meta = []
        id_sofar = 0
        self.used_tracers = {}
        self.l_min_sample = 1E30
        self.l_max_sample = -1E30
        for pk in self.twopoints:
            # Get the suffix for both tracers
            tn1, tn2 = pk['bins']
            pk_name1 = get_suffix_for_tr(s.tracers[tn1])
            pk_name2 = get_suffix_for_tr(s.tracers[tn2])
            k, pk, cov, ind = s.get_ell_cl('cl_%s%s' % (pk_name1, pk_name2),
                                              tn1,
                                              tn2,
                                              return_cov=True,
                                              return_ind=True)
            if pk.size > 0:
                if tn1 not in self.used_tracers:
                    self.used_tracers[tn1] = s.tracers[tn1].quantity
                if tn2 not in self.used_tracers:
                    self.used_tracers[tn2] = s.tracers[tn2].quantity

            self.pk_meta.append({'bin_1': tn1,
                                 'bin_2': tn2,
                                 'k': k,
                                 'pk': pk,
                                 'cov': cov,
                                 'inds': (id_sofar +
                                          np.arange(pk.size,
                                                    dtype=int))})
            indices += list(ind)
            id_sofar += pk.size
        indices = np.array(indices)
        # Reorder data vector and covariance
        self.data_vec = s.mean[indices]
        self.cov = s.covariance.covmat[indices][:, indices]
        # Spectral decomposition to avoid negative eigenvalues
        self.inv_cov = np.linalg.pinv(self.cov, rcond=1E-15, hermitian=True)
        self.ic_w, self.ic_v = np.linalg.eigh(self.inv_cov)
        self.ic_v = self.ic_v.T
        self.ic_w[self.ic_w < 0] = 0
        self.ndata = len(self.data_vec)

    def _get_nz(self, cosmo, name, **pars):
        """ Get redshift distribution for a given tracer."""
        z = self.bin_properties[name]['z_fid']
        nz = self.bin_properties[name]['nz_fid']
        if self.nz_model == 'NzShift':
            z = z + pars[self.input_params_prefix + '_' + name + '_dz']
            msk = z >= 0
            z = z[msk]
            nz = nz[msk]
        elif self.nz_model != 'NzNone':
            raise LoggedError(self.log, "Unknown Nz model %s" % self.nz_model)
        return (z, nz)

    def _get_bz(self, cosmo, name, **pars):
        """ Get linear galaxy bias. Unless we're using a linear bias,
        this should be just 1."""
        z = self.bin_properties[name]['z_fid']
        zmean = self.bin_properties[name]['zmean_fid']
        bz = np.ones_like(z)
        if self.bz_model == 'Linear':
            b1 = pars[self.input_params_prefix + '_' + name + '_b1']
            b1p = pars[self.input_params_prefix + '_' + name + '_b1p']
            bz = b1 + b1p * (z - zmean)
        return (z, bz)

    def _get_ia_bias(self, cosmo, name, **pars):
        """ Intrinsic alignment amplitude.
        """
        if self.ia_model == 'IANone':
            return None
        else:
            z = self.bin_properties[name]['z_fid']
            if self.ia_model == 'IAPerBin':
                A = pars[self.input_params_prefix + '_' + name + '_A_IA']
                A_IA = np.ones_like(z) * A
            elif self.ia_model == 'IADESY1':
                A0 = pars[self.input_params_prefix + '_A_IA']
                eta = pars[self.input_params_prefix + '_eta_IA']
                A_IA = A0 * ((1+z)/1.62)**eta
            else:
                raise LoggedError(self.log, "Unknown IA model %s" %
                                  self.ia_model)
            return (z, A_IA)

    def _get_tracers(self, cosmo, **pars):
        """ Transforms all used tracers into CCL tracers for the
        current set of parameters."""
        trs = {}
        is_PT_bias = self.bz_model in ['LagrangianPT', 'EulerianPT', 'BACCO', 'anzu']
        for name, q in self.used_tracers.items():
            if q == 'galaxy_density':
                nz = self._get_nz(cosmo, name, **pars)
                bz = self._get_bz(cosmo, name, **pars)
                t = ccl.NumberCountsTracer(cosmo, dndz=nz, bias=bz,
                                           has_rsd=False)
                if is_PT_bias:
                    z = self.bin_properties[name]['z_fid']
                    zmean = self.bin_properties[name]['zmean_fid']
                    pref = self.input_params_prefix + '_' + name
                    b1 = pars[pref + '_b1']
                    b1p = pars[pref + '_b1p']
                    bz = b1 + b1p * (z - zmean)
                    b2 = pars[pref + '_b2']
                    bs = pars[pref + '_bs']
                    bk2 = pars.get(pref + '_bk2', None)
                    b3nl = pars.get(pref + '_b3nl', None)
                    bsn = pars.get(pref + '_bsn', None)
                    if bk2 is not None or b3nl is not None:
                        ptt = pt.PTNumberCountsTracer(b1=(z, bz), b2=b2,
                                                      bs=bs, bk2=bk2, b3nl=b3nl, sn=bsn)
                    else:
                        ptt = pt.PTNumberCountsTracer(b1=(z, bz), b2=b2,
                                                      bs=bs)
            elif q == 'galaxy_shear':
                nz = self._get_nz(cosmo, name, **pars)
                ia = self._get_ia_bias(cosmo, name, **pars)
                t = ccl.WeakLensingTracer(cosmo, nz, ia_bias=ia)
                if is_PT_bias:
                    ptt = pt.PTMatterTracer()
            elif q == 'cmb_convergence':
                # B.H. TODO: pass z_source as parameter to the YAML file
                t = ccl.CMBLensingTracer(cosmo, z_source=1100)
                if is_PT_bias:
                    ptt = pt.PTMatterTracer()
            trs[name] = {}
            trs[name]['ccl_tracer'] = t
            if is_PT_bias:
                trs[name]['PT_tracer'] = ptt
        return trs

    def _get_pk_data(self, cosmo):
        """ Get all cosmology-dependent ingredients to create the
        different P(k)s needed for the C_ell calculation.
        For linear bias, this is just the matter power spectrum.
        """
        # Get P(k)s from CCL
        if self.bz_model == 'Linear':
            cosmo.compute_nonlin_power()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            return {'pk_mm': pkmm}
        elif self.bz_model in ['EulerianPT', 'LagrangianPT']:
            a_s = 1./(1+np.linspace(0., 4., 30)[::-1])
            if self.k_pt_filter > 0:
                k_filter = self.k_pt_filter
            else:
                k_filter = None
            if self.bz_model == 'EulerianPT':
                ptc = EPTCalculator(with_NC=True, with_IA=False,
                                    log10k_min=-4, log10k_max=2,
                                    nk_per_decade=20,
                                    a_arr=a_s, k_filter=k_filter)
            else:
                ptc = LPTCalculator(log10k_min=-4, log10k_max=2,
                                    nk_per_decade=20, h=cosmo['h'],
                                    a_arr=a_s, k_filter=k_filter)
            cosmo.compute_nonlin_power()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            pk_lin_z0 = ccl.linear_matter_power(cosmo, ptc.ks, 1.)
            Dz = ccl.growth_factor(cosmo, ptc.a_s)
            ptc.update_pk(pk_lin_z0, Dz)
            return {'ptc': ptc, 'pk_mm': pkmm}
        elif self.bz_model == 'BACCO':
            if self.k_pt_filter > 0:
                k_filter = self.k_pt_filter
            else:
                k_filter = None
            ptc = BACCOCalculator(bacco_emu=self.emu, log10k_min=np.log10(1e-2*cosmo['h']), log10k_max=np.log10(0.75*cosmo['h']),
                                  nk_per_decade=20, h=cosmo['h'], k_filter=k_filter)
            cosmo.compute_nonlin_power()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            ptc.update_pk(cosmo)
            return {'ptc': ptc, 'pk_mm': pkmm}
        elif self.bz_model == 'anzu':
            a_s = 1. / (1 + np.linspace(0., 4., 30)[::-1])
            # TODO: Implement k_filter in anzu
            if self.k_pt_filter > 0:
                k_filter = self.k_pt_filter
            else:
                k_filter = None
            ptc = HEFTCalculator(self.emu, cosmo, a_arr=a_s)
            cosmo.compute_nonlin_power()
            pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
            ptc.update_pk(cosmo)
            return {'ptc': ptc, 'pk_mm': pkmm}
        else:
            raise LoggedError(self.log,
                              "Unknown bias model %s" % self.bz_model)

    def _get_pkxy(self, cosmo, pkm, pkd, trs, **pars):
        """ Get the P(k) between two tracers. """
        q1 = self.used_tracers[pkm['bin_1']]
        q2 = self.used_tracers[pkm['bin_2']]

        if (self.bz_model == 'Linear') or (self.bz_model == 'BzNone'):
            if (q1 == 'galaxy_density') and (q2 == 'galaxy_density'):
                pk_pt = pkd['pk_mm']  # galaxy-galaxy
            elif ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                pk_pt = pkd['pk_mm']  # matter-matter
            else:
                pk_pt = pkd['pk_mm']  # galaxy-matter
        elif (self.bz_model == 'EulerianPT'):
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                pk_pt = pkd['pk_mm']  # matter-matter
            else:
                ptt1 = trs[pkm['bin_1']]['PT_tracer']
                ptt2 = trs[pkm['bin_2']]['PT_tracer']
                pk_pt = get_ept_pk2d(cosmo, ptt1, tracer2=ptt2,
                                     ptc=pkd['ptc'], sub_lowk=False)
        elif (self.bz_model == 'LagrangianPT'):
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                pk_pt = pkd['pk_mm']  # matter-matter
            else:
                ptt1 = trs[pkm['bin_1']]['PT_tracer']
                ptt2 = trs[pkm['bin_2']]['PT_tracer']
                pk_pt = get_lpt_pk2d(cosmo, ptt1, tracer2=ptt2,
                                     ptc=pkd['ptc'])
        elif (self.bz_model == 'BACCO'):
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                pk_pt = pkd['pk_mm']  # matter-matter
            else:
                ptt1 = trs[pkm['bin_1']]['PT_tracer']
                ptt2 = trs[pkm['bin_2']]['PT_tracer']
                pk_pt = get_bacco_pk2d(cosmo, ptt1, tracer2=ptt2,
                                     ptc=pkd['ptc'])
        elif (self.bz_model == 'anzu'):
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                pk_pt = pkd['pk_mm']  # matter-matter
            else:
                ptt1 = trs[pkm['bin_1']]['PT_tracer']
                ptt2 = trs[pkm['bin_2']]['PT_tracer']
                pk_pt = get_anzu_pk2d(cosmo, ptt1, tracer2=ptt2,
                                       ptc=pkd['ptc'])
        else:
            raise LoggedError(self.log,
                              "Unknown bias model %s" % self.bz_model)

        if self.bin_properties[pkm['bin_1']]['zmean_fid'] == self.bin_properties[pkm['bin_2']]['zmean_fid']:
            pk = pk_pt.eval(pkm['k'], 1./(1. + self.bin_properties[pkm['bin_1']]['zmean_fid']), cosmo)
            # print(q1, q2)
            # print(pkm['k'])
            # print(self.bin_properties[pkm['bin_1']]['zmean_fid'], self.bin_properties[pkm['bin_2']]['zmean_fid'])
            # print(pk)
            # if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
            #     ptt1 = trs[pkm['bin_1']]['PT_tracer']
            #     ptt2 = trs[pkm['bin_2']]['PT_tracer']
            #
            #     cosmology = {'omega_b': 0.02237,
            #                  'omega_cdm': 0.12,
            #                  'h': 0.6736,
            #                  'A_s': 2.083e-09,
            #                  'n_s': 0.9649,
            #                  'alpha_s': 0.0,
            #                  'N_ur': 2.0328,
            #                  'N_ncdm': 1.0,
            #                  'omega_ncdm': 0.0006442,
            #                  'w0_fld': -1.0,
            #                  'wa_fld': 0.0
            #                  }
            #     cosmo = ccl.Cosmology(Omega_c=cosmology['omega_cdm'] / cosmology['h'] ** 2,
            #                           Omega_b=cosmology['omega_b'] / cosmology['h'] ** 2,
            #                           h=cosmology['h'], A_s=cosmology['A_s'], n_s=cosmology['n_s'],
            #                           m_nu=0.06)
            #     sigma8 = ccl.sigma8(cosmo)
            #     cosmo = ccl.Cosmology(Omega_c=cosmology['omega_cdm'] / cosmology['h'] ** 2,
            #                           Omega_b=cosmology['omega_b'] / cosmology['h'] ** 2,
            #                           h=cosmology['h'], sigma8=sigma8, n_s=cosmology['n_s'],
            #                           m_nu=0.06)
            #     cosmo.compute_linear_power()
            #     cosmo.compute_nonlin_power()
            #
            #     pkd['ptc'].update_pk(cosmo)
            #
            #     pk_pt = get_bacco_pk2d(cosmo, ptt1, tracer2=ptt2,
            #                           ptc=pkd['ptc'])
            #     pk = pk_pt.eval(pkm['k'], 1., cosmo)
            #     print(pk)
            #     pk = pkd['ptc'].bacco_table
            #     print(pk[-1, 0, :])
            #     pk = pk_pt.eval(pkm['k'], 1./(1. + self.bin_properties[pkm['bin_1']]['zmean_fid']), cosmo)
            #     print(pk)
        else:
            pk = np.zeros_like(pkm['k'])

        return pk

    def _get_pk_all(self, cosmo, pk, **pars):
        """ Compute all C_ells."""
        # Gather all tracers
        trs = self._get_tracers(cosmo, **pars)

        # Correlate all needed pairs of tracers
        pks = []
        for pkm in self.pk_meta:
            pkxy = self._get_pkxy(cosmo, pkm, pk, trs, **pars)
            # clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
            pks.append(pkxy)
        return pks

    def _apply_shape_systematics(self, cls, **pars):
        if self.shape_model == 'ShapeMultiplicative':
            # Multiplicative shear bias
            for i, clm in enumerate(self.cl_meta):
                q1 = self.used_tracers[clm['bin_1']]
                q2 = self.used_tracers[clm['bin_2']]
                if q1 == 'galaxy_shear':
                    m1 = pars[self.input_params_prefix + '_' +
                              clm['bin_1'] + '_m']
                else:
                    m1 = 0.
                if q2 == 'galaxy_shear':
                    m2 = pars[self.input_params_prefix + '_' +
                              clm['bin_2'] + '_m']
                else:
                    m2 = 0.
                prefac = (1+m1) * (1+m2)
                cls[i] *= prefac

    def get_pks_theory(self, **pars):
        # Get cosmological model
        res = self.provider.get_CCL()
        cosmo = res['cosmo']

        # First, gather all the necessary ingredients for the different P(k)
        pkd = res['pk_data']

        # Then pass them on to convert them into C_ells
        pks = self._get_pk_all(cosmo, pkd, **pars)

        # Multiplicative bias if needed
        # self._apply_shape_systematics(cls, **pars)
        return pks

    def _get_spin_component(self, tr):
        return 'e' if self.used_tracers[tr] == 'galaxy_shear' else '0'

    def get_sacc_file(self, **pars):
        import sacc

        # Create empty file
        s = sacc.Sacc()

        # Add tracers
        for n, p in self.bin_properties.items():
            if n not in self.used_tracers:
                continue
            q = self.used_tracers[n]
            spin = 2 if q == 'galaxy_shear' else 0
            if q != 'cmb_convergence':
                s.add_tracer('NZ', n, quantity=q, spin=spin,
                             z=p['z_fid'], nz=p['nz_fid'])
            else:
                s.add_tracer('Map', n, quantity=q, spin=spin,
                             ell=np.arange(10), beam=np.ones(10))

        # Calculate power spectra
        pks = self.get_pks_theory(**pars)
        for pkm, pk in zip(self.pk_meta, pks):
            p1 = self._get_spin_component(pkm['bin_1'])
            p2 = self._get_spin_component(pkm['bin_2'])
            s.add_ell_cl(f'pk_{p1}{p2}', pkm['bin_1'], pkm['bin_2'],
                         pkm['l_eff'], pk)

        s.add_covariance(self.cov)
        return s

    def _get_theory(self, **pars):
        """ Computes theory vector."""
        pks = self.get_pks_theory(**pars)

        # Flattening into a 1D array
        pk_out = np.zeros(self.ndata)
        for pkm, pk in zip(self.pk_meta, pks):
            pk_out[pkm['inds']] = pk

        return pk_out

    def get_requirements(self):
        # By selecting `self._get_pk_data` as a `method` of CCL here,
        # we make sure that this function is only run when the
        # cosmological parameters vary.
        return {'CCL': {'methods': {'pk_data': self._get_pk_data}}}

    def logp(self, **pars):
        """
        Simple Gaussian likelihood.
        """
        t = self._get_theory(**pars)
        r = t - self.data_vec
        # t = np.random.multivariate_normal(t, self.cov)
        # chi2 = np.dot(r, self.inv_cov.dot(r))
        re = np.dot(self.ic_v, r)
        chi2 = np.sum(re**2*self.ic_w)
        return -0.5*chi2
