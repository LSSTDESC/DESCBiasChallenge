import numpy as np
from anzu.emu_funcs import LPTEmulator
import pyccl as ccl
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
from scipy.interpolate import interp1d


class HEFTCalculator(object):
    """ Class implements observables in 'Hybrid EFT'
    using anzu ( github.com/kokron/anzu ). Stephen Chen's
    velocileptors is used to compute underlying LPT spectra.
    Structure will be very similar to LPTCalculator().

    Args:
        ...
    """
    def __init__(self, emu=None, cosmo=None, a_arr=None, *args):
        self.emu = emu
        if emu is None:
            #No emu found, train and store emu object
            self.emu = LPTEmulator(kecleft=True, extrap=False)
        self.cosmo = cosmo
        if a_arr is None: 
            self.a_arr = 1./(1+np.linspace(0., 4., 30)[::-1])
        else:
            self.a_arr = a_arr
        self.nas = len(self.a_arr)
        self.lpt_table = None
        self.ks = None
        self.cleftobj = None

    def update_pk(self, cosmo):
        '''
        Computes the HEFT predictions using Anzu for the cosmology being probed by the likelihood.
        
        CLEFT predictions are made assuming kecleft as the LPT prediction. 

        Returns:
            self.pk_table : attribute for the HEFTCalculator class that will be passed onto the rest of the code
        Todo:
        1. Probably good to figure out k's used. LPTCalculator uses log10k = [-4, 2] which we can't access with anzu.
        '''
    
    
        k = np.logspace(-4, 0, 1000)
        k_emu = np.logspace(-2, 0, 100) 
        # If using kecleft, check that we're only varying the redshift
        
        if self.cleftobj is None:
            # Do the full calculation again, as the cosmology changed.
            pk = ccl.linear_matter_power(
                    self.cosmo, k * self.cosmo['h'], 1) * (self.cosmo['h'])**3
            
            # Function to obtain the no-wiggle spectrum.
            # Not implemented yet, maybe Wallisch maybe B-Splines?
            # pnw = p_nwify(pk)
            # For now just use Stephen's standard savgol implementation.
            self.cleftobj = RKECLEFT(k, pk)
        
        # Adjust growth factors
         
        Dz = ccl.background.growth_factor(cosmo, self.a_arr)
        lpt_spec = np.zeros((self.nas, 10, len(k)))
        for i,D in enumerate(Dz):

            self.cleftobj.make_ptable(D=D, kmin=k[0], kmax=k[-1], nk=1000)
            cleftpk = self.cleftobj.pktable.T
    
            # Adjust normalizations to match anzu measurements 
            cleftpk[3:, :] = self.cleftobj.pktable.T[3:, :]
            cleftpk[2, :] /= 2
            cleftpk[6, :] /= 0.25
            cleftpk[7, :] /= 2
            cleftpk[8, :] /= 2
            #Do we have to spline every time? nevertheless 
            cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')
            lpt_spec[i] = cleftspline(k)[1:11, :]
        #Computed the relevant lpt predictions, plug into emu
        #1. Set up cosmovec for anzu from CCL cosmo object
        #Final array should be [Nz, Nparam]
        #Parameter order is as specified in anzu documentation.
        #2. Update ptc for units that are h-ful for anzu 
        #3. Compute emulator basis functions
        #NOTE: Check ptc.lpt_table has the proper normalization for Anzu LPT spectra.
        h=cosmo['h']
        Oc = cosmo['Omega_c']
        Ob = cosmo['Omega_b']
        s8 = cosmo['sigma8']
        w  = cosmo['w0']
        ns = cosmo['n_s']
        neff = cosmo['Neff']
        anzu_temp = np.vstack([Ob*h**2, Oc*h**2, w, ns, s8, 100*h, neff] for n in range(self.nas)).T
        anzu_cosmo = np.vstack([anzu_temp, self.a_arr]).T  
        
        
        emu_spec = self.emu.predict(k_emu, anzu_cosmo, spec_lpt=lpt_spec,k_lpt=k)  
        
        
        self.lpt_table = emu_spec                
        #Convert units back to 1/Mpc and Mpc^3
        self.lpt_table /= cosmo['h']**3
        self.ks = k_emu*cosmo['h']



    def delta_b(self, cosmo, a, k):
        """
        inputs:
            cosmo: Cosmological parameters.
            a: scale factor
            k: wave vector
            
        calculates correction parameter that needs to be added to b1 under nongaussian conditions
        
        """
        
        growth_factor = ccl.background.growth_factor(cosmo, a)
        h = cosmo['h'] * 100
        Om = cosmo['Omega_m']
        constant = (h/(2.998 * (10 ** 8))) ** 2
        delta_c = 1.686
        t_k = ccl.power.linear_matter_power(cosmo, k, 1) / k**(cosmo['ns'])
        t_kl = ccl.power.linear_matter_power(cosmo, 10**-4, 1) / 10**-4
        t_k /= t_kl
        
        
        b = 3 * delta_c * Om * constant / (k ** 2 * t_k * growth_factor)
        
        return b





    def get_pgg(self, b11, b21, bs1, b12, b22, bs2,
                bk21=None, bk22=None, bsn1=None, bsn2=None, bsnx=None, fnl=None):
        """ 
        Get P_gg between two tracer samples with sets of bias params starting from the heft component spectra
    
        Adapted from the anzu method 'basis_to_full' from the LPTEmulator class.
        
        Inputs: 
        -btheta: vector of bias + shot noise. See notes below for structure of terms
    
        TODO:
        -Generalize biases for two populations of tracers. For now we're doing auto-correlations but it should be an exercise in book-keeping to adjust.
    
        Notes from Anzu documentation:
            Bias parameters can either be
            btheta = [b1, b2, bs2, SN]
            or
            btheta = [b1, b2, bs2, bnabla2, SN]
            Where SN is a constant term, and the bnabla2 terms follow the approximation
            <X, nabla^2 delta> ~ -k^2 <X, 1>. 
            Note the term <nabla^2, nabla^2> isn't included in the prediction since it's degenerate with even higher deriv
            terms such as <nabla^4, 1> which in principle have different parameters. 
        """
        # Clarification:
        # anzu uses the following expansion for the galaxy overdensity:
        #   d_g = b1 d + b2/2 d2^2 + bs s^2 + bnabla2d nabla^2 d
        # (see Eq. 1 of https://arxiv.org/abs/2101.12187).
        #
        # The BACCO table below contains the following power spectra
        # in order:
        # <1,1>
        # <1,d>
        # <d,d>
        # <1,d^2>
        # <d,d^2>
        # <d^2,d^2> (!)
        # <1,s^2>
        # <d,s^2>
        # <d^2,s^2> (!)
        # <s^2,s^2> (!)
        #
        # EPT uses:
        #   d_g = b1 d + b2 d2^2/2 + bs s^2/2 + b3 psi/2 + bnabla nablad/2
        # So:
        #   a) The spectra involving b2 are for d^2 - convert to d^2/2
        #   b) The spectra involving bs are for s^2 - convert to s^2/2
        #   c) The spectra involving bnabla are for nablad - convert to nablad/2
        # Also, the spectra marked with (!) tend to a constant
        # as k-> 0, which we can suppress with a low-pass filter.
        #
        # Importantly, we have corrected the spectra involving d^2 and s2 to
        # make the definitions of b2, bs equivalent to what we have adopted for
        # the EPT and LPT expansions.

        b1_list = [b11, b21, bs1, bk21, bsn1]
        b2_list = [b12, b22, bs2, bk22, bsn2]

        cross = True
        if np.all([np.all(b1_list[i] == b2_list[i]) for i in range(len(b1_list))]):
            cross = False

        bL11 = b11 - 1 + self.delta_b(self.cosmo, self.ks, self.a_arr)
        bL12 = b12 - 1

        if bk21 is None:
            bk21 = np.zeros_like(self.a_arr)
        if bk22 is None:
            bk22 = np.zeros_like(self.a_arr)
        if bsn1 is None:
            bsn1 = np.zeros_like(self.a_arr)
        if bsn2 is None:
            bsn2 = np.zeros_like(self.a_arr)
        if bsnx is None:
            bsnx = np.zeros_like(self.a_arr)

        # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
        # bterms_hh = [np.ones(self.nas),
        #              2*bL11, bL11**2,
        #              b21, b21*bL11, 0.25*b21**2,
        #              bs1, bs1*bL11, 0.5*bs1*b21, 0.25*bs1**2,
        #              bk21, bk21*bL11, 0.5*bk21*b21, 0.5*bk21*bs1]
        # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
        bterms_hh = [np.ones(self.nas),
                     bL11+bL12, bL11*bL12,
                     0.5*(b21+b22), 0.5*(b21*bL12+b22*bL11), 0.25*b21*b22,
                     0.5*(bs1+bs2), 0.5*(bs1*bL12+bs2*bL11), 0.25*(bs1*b22+bs2*b21), 0.25*bs1*bs2,
                     0.5*(bk21+bk22), 0.5*(bk21*bL12+bk22+bL11), 0.25*(bk21*b22+bk22*b21), 0.25*(bk21*bs2+bk22*bs1)]

        pkvec = np.zeros(shape=(self.nas, 14, len(self.ks)))
        pkvec[:,:10] = self.lpt_table
        # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
        nabla_idx = [0, 1, 3, 6]
        # Higher derivative terms
        pkvec[:,10:] = -self.ks**2 * pkvec[:,nabla_idx]

        bterms_hh = np.array(bterms_hh)
        if not cross:
            p_hh = np.einsum('bz, zbk->zk', bterms_hh, pkvec) + np.einsum('z, zk->zk', bsn1,
                                                                          np.ones(shape=(self.nas, len(self.ks))))
        else:
            p_hh = np.einsum('bz, zbk->zk', bterms_hh, pkvec) + np.einsum('z, zk->zk', bsnx,
                                                                          np.ones(shape=(self.nas, len(self.ks))))

        return p_hh

    def get_pgg_debug(self, b11, b21, bs1, b12, b22, bs2,
                bk21=None, bk22=None, bsn1=None, bsn2=None, fnl=None):
        """
        Get P_gg between two tracer samples with sets of bias params starting from the heft component spectra

        Adapted from the anzu method 'basis_to_full' from the LPTEmulator class.

        Inputs:
        -btheta: vector of bias + shot noise. See notes below fos tructure of terms

        TODO:
        -Generalize biases for two populations of tracers. For now we're doing auto-correlations but it should be an exercise in book-keeping to adjust.

        Notes from Anzu documentation:
            Bias parameters can either be
            btheta = [b1, b2, bs2, SN]
            or
            btheta = [b1, b2, bs2, bnabla2, SN]
            Where SN is a constant term, and the bnabla2 terms follow the approximation
            <X, nabla^2 delta> ~ -k^2 <X, 1>.
            Note the term <nabla^2, nabla^2> isn't included in the prediction since it's degenerate with even higher deriv
            terms such as <nabla^4, 1> which in principle have different parameters.
        """
        # Clarification:
        # anzu uses the following expansion for the galaxy overdensity:
        #   d_g = b1 d + b2/2 d2^2 + bs s^2 + bnabla2d nabla^2 d
        # (see Eq. 1 of https://arxiv.org/abs/2101.12187).
        #
        # The BACCO table below contains the following power spectra
        # in order:
        # <1,1>
        # <1,d>
        # <d,d>
        # <1,d^2>
        # <d,d^2>
        # <d^2,d^2> (!)
        # <1,s^2>
        # <d,s^2>
        # <d^2,s^2> (!)
        # <s^2,s^2> (!)
        #
        # EPT uses:
        #   d_g = b1 d + b2 d2^2/2 + bs s^2/2 + b3 psi/2 + bnabla nablad/2
        # So:
        #   a) The spectra involving b2 are for d^2 - convert to d^2/2
        #   b) The spectra involving bs are for s^2 - convert to s^2/2
        #   c) The spectra involving bnabla are for nablad - convert to nablad/2
        # Also, the spectra marked with (!) tend to a constant
        # as k-> 0, which we can suppress with a low-pass filter.
        #
        # Importantly, we have corrected the spectra involving d^2 and s2 to
        # make the definitions of b2, bs equivalent to what we have adopted for
        # the EPT and LPT expansions.

        b1_list = [b11, b21, bs1, bk21, bsn1]
        b2_list = [b12, b22, bs2, bk22, bsn2]

        assert np.all([np.all(b1_list[i] == b2_list[i]) for i in range(len(b1_list))]), \
            'Two populations of tracers are not yet implemented for HEFT!'

        bL11 = b11 - 1 + self.delta_b(self.cosmo, self.ks, self.a_arr)
        bL12 = b12 - 1

        if bk21 is None:
            bk21 = np.zeros_like(self.a_arr)
        if bk22 is None:
            bk22 = np.zeros_like(self.a_arr)
        if bsn1 is None:
            bsn1 = np.zeros_like(self.a_arr)
        if bsn2 is None:
            bsn2 = np.zeros_like(self.a_arr)

        # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
        bterms_hh = [np.ones(self.nas),
                     2 * bL11, bL11 ** 2,
                     b21, b21 * bL11, 0.25 * b21 ** 2,
                     bs1, bs1 * bL11, 0.5 * bs1 * b21, 0.25 * bs1 ** 2,
                     bk21, bk21 * bL11, 0.5 * bk21 * b21, 0.5 * bk21 * bs1]
        pkvec = np.zeros(shape=(self.nas, 14, len(self.ks)))
        pkvec[:, :10] = self.lpt_table
        # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
        nabla_idx = [0, 1, 3, 6]
        # Higher derivative terms
        pkvec[:, 10:] = -self.ks ** 2 * pkvec[:, nabla_idx]

        bterms_hh = np.array(bterms_hh)
        p_hh = np.einsum('bz, zbk->zbk', bterms_hh, pkvec)
        p_sn = bsn1[0]*np.ones((self.nas, len(self.ks)))

        return p_hh, p_sn

    def get_pgm(self, b1, b2, bs, bk2=None, bsn=None, fnl=None):
        """ Get P_gm for a set of bias parameters from the heft component spectra
       
          Inputs: 
          -btheta: vector of bias + shot noise. See notes below fos tructure of terms
           
          Outputs:
          -p_gm: tracer-matter power spectrum    
        """

        if bk2 is None:
            bk2 = np.zeros_like(self.a_arr)
        if bsn is None:
            bsn = np.zeros_like(self.a_arr)

        bL1 = b1 - 1. + self.delta_b(self.cosmo, self.ks, self.a_arr)

        # hm correlations only have one kind of <1,delta_i> correlation
        bterms_hm = [np.ones(self.nas),
                     bL1, np.zeros(self.nas),
                     0.5*b2, np.zeros(self.nas), np.zeros(self.nas),
                     0.5*bs, np.zeros(self.nas), np.zeros(self.nas), np.zeros(self.nas),
                     0.5*bk2, np.zeros(self.nas), np.zeros(self.nas), np.zeros(self.nas)]
        pkvec = np.zeros(shape=(self.nas, 14, len(self.ks)))
        pkvec[:,:10] = self.lpt_table
        # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
        nabla_idx = [0, 1, 3, 6]

        # Higher derivative terms
        pkvec[:,10:] = -self.ks**2 * pkvec[:,nabla_idx]
        bterms_hm = np.array(bterms_hm)
        p_hm = np.einsum('bz, zbk->zk', bterms_hm, pkvec)

        return p_hm

def get_anzu_pk2d(cosmo, tracer1, tracer2=None, ptc=None, bsnx=None, fnl=None):
    """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the PT power spectrum for two quantities defined by
    two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.
    """

    #Commenting the lines below because we are using custom tracer objects
    if not isinstance(tracer1, ccl.nl_pt.PTTracer):
       raise TypeError("tracer1 must be of type `ccl.nl_pt.PTTracer`")
    if not isinstance(tracer2, ccl.nl_pt.PTTracer):
       raise TypeError("tracer2 must be of type `ccl.nl_pt.PTTracer`")

    if not isinstance(ptc, HEFTCalculator):
       raise TypeError("ptc should be of type `HEFTCalculator`")
    # z
    z_arr = 1. / ptc.a_arr - 1
    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        if hasattr(tracer1, 'bk2'):
            bk21 = tracer1.bk2(z_arr)
        else:
            bk21 = None
        if hasattr(tracer1, 'sn'):
            bsn1 = tracer1.sn(z_arr)
        else:
            bsn1 = None
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            if hasattr(tracer2, 'bk2'):
                bk22 = tracer2.bk2(z_arr)
            else:
                bk22 = None
            if hasattr(tracer2, 'sn'):
                bsn2 = tracer2.sn(z_arr)
            else:
                bsn2 = None

            if bsnx is not None:
                bsnx = bsnx*np.ones_like(z_arr)

            p_pt = ptc.get_pgg(b11, b21, bs1, b12, b22, bs2, bk21, bk22, bsn1, bsn2, bsnx, fnl = fnl)

        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(b11, b21, bs1, bk21, bsn1, fnl = fnl)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            if hasattr(tracer2, 'bk2'):
                bk22 = tracer2.bk2(z_arr)
            else:
                bk22 = None
            if hasattr(tracer2, 'sn'):
                bsn2 = tracer2.sn(z_arr)
            else:
                bsn2 = None
            
            p_pt = ptc.get_pgm(b12, b22, bs2, bk22, bsn2, fnl = fnl)
        elif (tracer2.type == 'M'):
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    else:
        raise NotImplementedError("Combination %s-%s not implemented yet" %
                                  (tracer1.type, tracer2.type))

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    pt_pk = ccl.Pk2D(a_arr=ptc.a_arr,
                     lk_arr=np.log(ptc.ks),
                     pk_arr=p_pt,
                     is_logp=False)
    return pt_pk
