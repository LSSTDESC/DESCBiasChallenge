import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import pyccl.nl_pt as pt
from cl_like.lpt import LPTCalculator, get_lpt_pk2d
from cl_like.ept import EPTCalculator, get_ept_pk2d
import yaml

class posterior():
    
    def __init__(self, input_yaml):
        self.input_yaml = input_yaml
        
        with open(input_yaml, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        likelihood_dir = data_loaded['likelihood']['cl_like.ClLike']
        self.theory = data_loaded['theory']['cl_like.CCL']
        self.pars = data_loaded['params']
        self.input_params_prefix: str = likelihood_dir['input_params_prefix']
        # Input sacc file
        self.input_file: str = likelihood_dir['input_file']
        # IA model name. Currently all of these are
        # just flags, but we could turn them into
        # homogeneous systematic classes.
        self.ia_model: str = likelihood_dir['ia_model']
        # N(z) model name
        self.nz_model: str = likelihood_dir['nz_model']
        # b(z) model name
        self.bz_model: str = likelihood_dir['bz_model']
        # Shape systamatics
        self.shape_model: str = likelihood_dir['shape_model']
        # List of bin names
        self.bins: list = likelihood_dir['bins']
        # List of default settings (currently only scale cuts)
        self.defaults: dict = likelihood_dir['defaults']
        # List of two-point functions that make up the data vector
        self.twopoints: list = likelihood_dir['twopoints']
        # Low-pass filter for PT
        self.k_pt_filter: float = 0.01
        self.pars = data_loaded['params']
        
        self._read_data()
        self._get_ell_sampling()
        
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

        def get_lmax_from_kmax(cosmo, kmax, z, nz):
            zmid = np.sum(z*nz)/np.sum(nz)
            chi = ccl.comoving_radial_distance(cosmo, 1./(1+zmid))
            lmax = np.max([10., kmax * chi - 0.5])
            return lmax

        s = sacc.Sacc.load_fits(self.input_file)
        self.bin_properties = {}
        cosmo_lcdm = ccl.CosmologyVanillaLCDM()
        kmax_default = self.defaults.get('kmax', 0.1)
        for b in self.bins:
            if b['name'] not in s.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b['name'])
            t = s.tracers[b['name']]
            zmean = np.average(t.z, weights=t.nz)
            self.bin_properties[b['name']] = {'z_fid': t.z,
                                              'nz_fid': t.nz,
                                              'zmean_fid': zmean}
            # Ensure all tracers have ell_min
            if b['name'] not in self.defaults:
                self.defaults[b['name']] = {}
                self.defaults[b['name']]['lmin'] = self.defaults['lmin']

            # Give galaxy clustering an ell_max
            if t.quantity == 'galaxy_density':
                # Get lmax from kmax for galaxy clustering
                if 'kmax' in self.defaults[b['name']]:
                    kmax = self.defaults[b['name']]['kmax']
                else:
                    kmax = kmax_default
                lmax = get_lmax_from_kmax(cosmo_lcdm,
                                          kmax,
                                          t.z, t.nz)
                self.defaults[b['name']]['lmax'] = lmax

            # Make sure everything else has an ell_max
            if 'lmax' not in self.defaults[b['name']]:
                self.defaults[b['name']]['lmax'] = self.defaults['lmax']

        # First check which parts of the data vector to keep
        indices = []
        for cl in self.twopoints:
            tn1, tn2 = cl['bins']
            lmin = np.max([self.defaults[tn1].get('lmin', 2),
                           self.defaults[tn2].get('lmin', 2)])
            lmax = np.min([self.defaults[tn1].get('lmax', 1E30),
                           self.defaults[tn2].get('lmax', 1E30)])
            # Get the suffix for both tracers
            cl_name1 = get_suffix_for_tr(s.tracers[tn1])
            cl_name2 = get_suffix_for_tr(s.tracers[tn2])
            ind = s.indices('cl_%s%s' % (cl_name1, cl_name2), (tn1, tn2),
                            ell__gt=lmin, ell__lt=lmax)
            indices += list(ind)
        s.keep_indices(np.array(indices))

        # Now collect information about those
        # and put the C_ells in the right order
        indices = []
        self.cl_meta = []
        id_sofar = 0
        self.used_tracers = {}
        self.l_min_sample = 1E30
        self.l_max_sample = -1E30
        for cl in self.twopoints:
            # Get the suffix for both tracers
            tn1, tn2 = cl['bins']
            cl_name1 = get_suffix_for_tr(s.tracers[tn1])
            cl_name2 = get_suffix_for_tr(s.tracers[tn2])
            l, c_ell, cov, ind = s.get_ell_cl('cl_%s%s' % (cl_name1, cl_name2),
                                              tn1,
                                              tn2,
                                              return_cov=True,
                                              return_ind=True)
            if c_ell.size > 0:
                if tn1 not in self.used_tracers:
                    self.used_tracers[tn1] = s.tracers[tn1].quantity
                if tn2 not in self.used_tracers:
                    self.used_tracers[tn2] = s.tracers[tn2].quantity

            bpw = s.get_bandpower_windows(ind)
            if np.amin(bpw.values) < self.l_min_sample:
                self.l_min_sample = np.amin(bpw.values)
            if np.amax(bpw.values) > self.l_max_sample:
                self.l_max_sample = np.amax(bpw.values)

            self.cl_meta.append({'bin_1': tn1,
                                 'bin_2': tn2,
                                 'l_eff': l,
                                 'cl': c_ell,
                                 'cov': cov,
                                 'inds': (id_sofar +
                                          np.arange(c_ell.size,
                                                    dtype=int)),
                                 'l_bpw': bpw.values,
                                 'w_bpw': bpw.weight.T})
            indices += list(ind)
            id_sofar += c_ell.size
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

    def _get_ell_sampling(self, nl_per_decade=30):
        # Selects ell sampling.
        # Ell max/min are set by the bandpower window ells.
        # It currently uses simple log-spacing.
        # nl_per_decade is currently fixed at 30
        if self.l_min_sample == 0:
            l_min_sample_here = 2
        else:
            l_min_sample_here = self.l_min_sample
        nl_sample = int(np.log10(self.l_max_sample / l_min_sample_here) *
                        nl_per_decade)
        l_sample = np.unique(np.geomspace(l_min_sample_here,
                                          self.l_max_sample+1,
                                          nl_sample).astype(int)).astype(float)

        if self.l_min_sample == 0:
            self.l_sample = np.concatenate((np.array([0.]), l_sample))
        else:
            self.l_sample = l_sample
            
            
    def _eval_interp_cl(self, cl_in, l_bpw, w_bpw):
        """ Interpolates C_ell, evaluates it at bandpower window
        ell values and convolves with window."""
        f = interp1d(self.l_sample, cl_in)
        cl_unbinned = f(l_bpw)
        cl_binned = np.dot(w_bpw, cl_unbinned)
        return cl_binned

    def _get_nz(self, cosmo, name):
        """ Get redshift distribution for a given tracer."""
        z = self.bin_properties[name]['z_fid']
        nz = self.bin_properties[name]['nz_fid']
        if self.nz_model == 'NzShift':
            z = z + self.pars[self.input_params_prefix + '_' + name + '_dz']
            msk = z >= 0
            z = z[msk]
            nz = nz[msk]
        elif self.nz_model != 'NzNone':
            raise LoggedError(self.log, "Unknown Nz model %s" % self.nz_model)
        return (z, nz)    

    def _get_bz(self, cosmo, name):
        """ Get linear galaxy bias. Unless we're using a linear bias,
        this should be just 1."""
        z = self.bin_properties[name]['z_fid']
        zmean = self.bin_properties[name]['zmean_fid']
        bz = np.ones_like(z)
        if self.bz_model == 'Linear':
            b1 = self.pars[self.input_params_prefix + '_' + name + '_b1']
            b1p = self.pars[self.input_params_prefix + '_' + name + '_b1p']
            bz = b1 + b1p * (z - zmean)
        return (z, bz)    
    
    def _get_ia_bias(self, cosmo, name):
        """ Intrinsic alignment amplitude.
        """
        if self.ia_model == 'IANone':
            return None
        else:
            z = self.bin_properties[name]['z_fid']
            if self.ia_model == 'IAPerBin':
                A = self.pars[self.input_params_prefix + '_' + name + '_A_IA']
                A_IA = np.ones_like(z) * A
            elif self.ia_model == 'IADESY1':
                A0 = self.pars[self.input_params_prefix + '_A_IA']
                eta = self.pars[self.input_params_prefix + '_eta_IA']
                A_IA = A0 * ((1+z)/1.62)**eta
            else:
                raise LoggedError(self.log, "Unknown IA model %s" %
                                  self.ia_model)
            return (z, A_IA)
        
    def _get_tracers(self, cosmo, bias_pars = None):
        """ Transforms all used tracers into CCL tracers for the
        current set of parameters."""
        trs = {}
        is_PT_bias = self.bz_model in ['LagrangianPT', 'EulerianPT']
        for name, q in self.used_tracers.items():
            if q == 'galaxy_density':
                nz = self._get_nz(cosmo, name)
                bz = self._get_bz(cosmo, name)
                t = ccl.NumberCountsTracer(cosmo, dndz=nz, bias=bz,
                                           has_rsd=False)
                if is_PT_bias:
                    z = self.bin_properties[name]['z_fid']
                    zmean = self.bin_properties[name]['zmean_fid']                    
                    if bias_pars is None:  

                        b1 = self.pars[self.input_params_prefix + '_' + name + '_b1']
                        b1p = self.pars[self.input_params_prefix + '_' + name + '_b1p']
                        bz = b1 + b1p * (z - zmean)
                        b2 = self.pars[self.input_params_prefix + '_' + name + '_b2']
                        bs = self.pars[self.input_params_prefix + '_' + name + '_bs']
                        ptt = pt.PTNumberCountsTracer(b1=(z, bz), b2=b2, bs=bs)
                        
                    else:
                        b1 = bias_pars[self.input_params_prefix + '_' + name + '_b1']
                        b1p = bias_pars[self.input_params_prefix + '_' + name + '_b1p']
                        bz = b1 + b1p * (z - zmean)
                        b2 = bias_pars[self.input_params_prefix + '_' + name + '_b2']
                        bs = bias_pars[self.input_params_prefix + '_' + name + '_bs']
                        ptt = pt.PTNumberCountsTracer(b1=(z, bz), b2=b2, bs=bs)
            elif q == 'galaxy_shear':
                nz = self._get_nz(cosmo, name)
                ia = self._get_ia_bias(cosmo, name)
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
        else:
            raise LoggedError(self.log,
                              "Unknown bias model %s" % self.bz_model)
            
    def _get_pkxy(self, cosmo, clm, pkd, trs):
        """ Get the P(k) between two tracers. """
        q1 = self.used_tracers[clm['bin_1']]
        q2 = self.used_tracers[clm['bin_2']]

        if (self.bz_model == 'Linear') or (self.bz_model == 'BzNone'):
            if (q1 == 'galaxy_density') and (q2 == 'galaxy_density'):
                return pkd['pk_mm']  # galaxy-galaxy
            elif ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                return pkd['pk_mm']  # matter-matter
            else:
                return pkd['pk_mm']  # galaxy-matter
        elif (self.bz_model == 'EulerianPT'):
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                return pkd['pk_mm']  # matter-matter
            else:
                ptt1 = trs[clm['bin_1']]['PT_tracer']
                ptt2 = trs[clm['bin_2']]['PT_tracer']
                pk_pt = get_ept_pk2d(cosmo, ptt1, tracer2=ptt2,
                                     ptc=pkd['ptc'], sub_lowk=False)
                return pk_pt
        elif (self.bz_model == 'LagrangianPT'):
            if ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
                return pkd['pk_mm']  # matter-matter
            else:
                ptt1 = trs[clm['bin_1']]['PT_tracer']
                ptt2 = trs[clm['bin_2']]['PT_tracer']
                pk_pt = get_lpt_pk2d(cosmo, ptt1, tracer2=ptt2,
                                     ptc=pkd['ptc'])
                return pk_pt
        else:
            raise LoggedError(self.log,
                              "Unknown bias model %s" % self.bz_model)
            
    def _get_cl_all(self, cosmo, pk,bias_pars):
        """ Compute all C_ells."""
        # Gather all tracers
        trs = self._get_tracers(cosmo,bias_pars)

        # Correlate all needed pairs of tracers
        cls = []
        for clm in self.cl_meta:
            pkxy = self._get_pkxy(cosmo, clm, pk, trs)
            cl = ccl.angular_cl(cosmo,
                                trs[clm['bin_1']]['ccl_tracer'],
                                trs[clm['bin_2']]['ccl_tracer'],
                                self.l_sample, p_of_k_a=pkxy)
            clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
            cls.append(clb)
        return cls
    
    def _apply_shape_systematics(self, cls):
        if self.shape_model == 'ShapeMultiplicative':
            # Multiplicative shear bias
            for i, clm in enumerate(self.cl_meta):
                q1 = self.used_tracers[clm['bin_1']]
                q2 = self.used_tracers[clm['bin_2']]
                if q1 == 'galaxy_shear':
                    m1 = self.pars[self.input_params_prefix + '_' +
                              clm['bin_1'] + '_m']
                else:
                    m1 = 0.
                if q2 == 'galaxy_shear':
                    m2 = self.pars[self.input_params_prefix + '_' +
                              clm['bin_2'] + '_m']
                else:
                    m2 = 0.
                prefac = (1+m1) * (1+m2)
                cls[i] *= prefac
                
    def get_cls_theory(self,CCL_cosmo,bias_pars):
        
        if CCL_cosmo is None:
            cosmo = ccl.Cosmology(
                Omega_c= self.pars['Omega_c'], 
                Omega_b=self.pars['Omega_b'], 
                h=self.pars['h'], 
                n_s=self.pars['n_s'],
                sigma8= self.pars['sigma8'],
                T_CMB=2.7255,m_nu=self.pars['m_nu'],
                transfer_function=self.theory['transfer_function'],
                matter_power_spectrum=self.theory['matter_pk'],
                baryons_power_spectrum=self.theory['baryons_pk'])
            print('No CCL cosmology provided! Using default value from input file.')
        else:
            cosmo = CCL_cosmo
        pkd = self._get_pk_data(cosmo)
        # Then pass them on to convert them into C_ells
        cls = self._get_cl_all(cosmo, pkd,bias_pars)
        # Multiplicative bias if needed
        self._apply_shape_systematics(cls)
        return cls

    def _get_theory(self,Theory):
        """ Computes theory vector."""
        CCL_cosmo = Theory['Cosmo']
        bias_pars = Theory['bias_parms']
        cls = self.get_cls_theory(CCL_cosmo,bias_pars)

        # Flattening into a 1D array
        cl_out = np.zeros(self.ndata)
        for clm, cl in zip(self.cl_meta, cls):
            cl_out[clm['inds']] = cl

        return cl_out
    
    def logp(self,Theory):
        """
        Simple Gaussian likelihood.
        """
        t = self._get_theory(Theory)
        r = t - self.data_vec
        # t = np.random.multivariate_normal(t, self.cov)
        # chi2 = np.dot(r, self.inv_cov.dot(r))
        re = np.dot(self.ic_v, r)
        chi2 = np.sum(re**2*self.ic_w)
        return -0.5*chi2
        