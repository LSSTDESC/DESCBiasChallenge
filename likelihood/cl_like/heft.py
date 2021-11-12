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
            self.emu = LPTEmulator()
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
    def get_pgg(self, btheta1, btheta2=None):
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
        if btheta2 is not None:
            raise NotImplementedError("Two populations of tracers are not yet implemented for HEFT!")
        if len(btheta1) == 4:
            b1, b2, bs, sn = btheta1
            # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [np.ones(self.nas),
                         2*b1, b1**2,
                         b2, b2*b1, 0.25*b2**2,
                         2*bs, 2*bs*b1, bs*b2, bs**2]
            pkvec = self.lpt_table
        else:
            b1, b2, bs, bk2, sn = btheta1
            # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [np.ones(self.nas),
                         2*b1, b1**2,
                         b2, b2*b1, 0.25*b2**2,
                         2*bs, 2*bs*b1, bs*b2, bs**2,
                         2*bk2, 2*bk2*b1, bk2*b2, 2*bk2*bs]
            pkvec = np.zeros(shape=(self.nas, 14, len(self.ks)))
            pkvec[:,:10] = self.lpt_table
            # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
            nabla_idx = [0, 1, 3, 6]
            # Higher derivative terms
            pkvec[:,10:] = -self.ks**2 * pkvec[:,nabla_idx] 
        bterms_hh = np.array(bterms_hh)
        p_hh = np.einsum('bz, zbk->zk', bterms_hh, pkvec) + np.einsum('z, zk->zk', sn, np.ones(shape=(self.nas, len(self.ks)) ))
        return p_hh
    def get_pgm(self, btheta):
        """ Get P_gm for a set of bias parameters from the heft component spectra
       
          Inputs: 
          -btheta: vector of bias + shot noise. See notes below fos tructure of terms
           
          Outputs:
          -p_gm: tracer-matter power spectrum    
        """
        if len(btheta) == 4:
            b1, b2, bs, sn = btheta
            bterms_hm = [np.ones(self.nas),
                         b1, np.zeros(self.nas),
                         b2/2, np.zeros(self.nas), np.zeros(self.nas),
                         bs, np.zeros(self.nas), np.zeros(self.nas), np.zeros(self.nas)]
            pkvec = self.lpt_table
        else:
            # hm correlations only have one kind of <1,delta_i> correlation
            b1, b2, bs, bk2, sn = btheta
            bterms_hm = [np.ones(self.nas),
                         b1, np.zeros(self.nas),
                         b2/2, np.zeros(self.nas), np.zeros(self.nas),
                         bs, np.zeros(self.nas), np.zeros(self.nas), np.zeros(self.nas),
                         bk2, np.zeros(self.nas), np.zeros(self.nas), np.zeros(self.nas)]
            pkvec = np.zeros(shape=(self.nas, 14, len(self.ks)))
            pkvec[:,:10] = self.lpt_table
            # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
            nabla_idx = [0, 1, 3, 6]

            # Higher derivative terms
            pkvec[:,10:] = -self.ks**2 * pkvec[:,nabla_idx] 
        bterms_hm = np.array(bterms_hm)
        p_hm = np.einsum('bz, zbk->zk', bterms_hm, pkvec)
        return p_hm
def get_heft_pk2d(cosmo, tracer1, tracer2=None, ptc=None):
    """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the PT power spectrum for two quantities defined by
    two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.
    """
    print('TRACERS:', tracer1, tracer2)
    if tracer2 is None:
        tracer2 = tracer1
    if tracer2 is not None:
        if tracer2 is tracer1:
            pass 
        elif tracer2.type == 'M' or tracer1.type == 'M':
            pass
        else:
            raise NotImplementedError('Two-tracer correlations are not implemented yet!')
    #Commenting the lines below because we are using custom tracer objects
    #if not isinstance(tracer1, ccl.nl_pt.PTTracer):
    #    raise TypeError("tracer1 must be of type `ccl.nl_pt.PTTracer`")
    #if not isinstance(tracer2, ccl.nl_pt.PTTracer):
    #    raise TypeError("tracer2 must be of type `ccl.nl_pt.PTTracer`")

    #if not isinstance(ptc, LPTCalculator):
    #    raise TypeError("ptc should be of type `LPTCalculator`")
    # z
    z_arr = 1. / ptc.a_arr - 1
    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        bk21 = tracer1.bk2(z_arr)
        sn21 = tracer1.sn(z_arr)
        btheta1 = np.array([b11, b21, bs1, bk21, sn21])
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)

            bk22 = tracer2.bk2(z_arr)
            sn22 = tracer2.sn(z_arr)
            
            btheta2 = np.array([b21, b22, bs2, bk22, sn22])
            
            #Right now get_pgg will get tracer auto-spectrum.
            p_pt = ptc.get_pgg(btheta1)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(btheta1)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)

            bk22 = tracer2.bk2(z_arr)
            sn22 = tracer2.sn(z_arr)
            
            btheta2 = np.array([b21, b22, bs2, bk22, sn22])
            
            p_pt = ptc.get_pgm(btheta2)
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