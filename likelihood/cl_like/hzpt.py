import numpy as np
# from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
import pyccl as ccl
from gzpt import hzpt,tracers
from copy import copy
import time
#structure stolen directly from lpt.py/ept.py


class HZPTCalculator(object):
    """
    Args:
        log10k_min (float): decimal logarithm of the minimum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        log10k_max (float): decimal logarithm of the maximum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        a_arr (array_like): array of scale factors at which
            growth/bias will be evaluated.
    """
    def __init__(self, log10k_min=-4, log10k_max=2,
                 nk_per_decade=20, a_arr=None, h=None, k_filter=None):
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        self.h = h
        self.ksh = self.ks/self.h
        self.h3 = self.h**3
        if a_arr is None:
            a_arr = 1./(1+np.linspace(0., 4., 30)[::-1])
        self.a_s = a_arr
        #self.pt_table = None
        self.pt_table = np.zeros( (len(self.a_s),nk_total) )
        if k_filter is not None:
            self.wk_low = 1-np.exp(-(self.ks/k_filter)**2)
        else:
            self.wk_low = np.ones(nk_total)

    def update_pk(self, pk, Dz):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        if pk.shape != self.ks.shape:
            raise ValueError("Input spectrum has wrong shape")
        if Dz.shape != self.a_s.shape:
            raise ValueError("Input growth has wrong shape")
        self.models = []
        #FIXME make this more efficient by using their structure,
        #can refactor hzpt to operate BB on all z at once and separately keep ZA table
        pk_use = pk*self.h3#*Dz**2 #assuming normed to 1 at z=0, encode right linear evoln...
        t0=time.perf_counter()
        hzpt_model = hzpt(self.ksh,pk_use,config=False)
        t1=time.perf_counter()
        for D in Dz:
            if(D==Dz[0]):
                t0i=time.perf_counter()

            hzpt_model.update_redshift(Dz=D)
            if(D==Dz[0]):
                t0ii=time.perf_counter()
            tmp = copy(hzpt_model)
            # pk_hzpt = hzpt_model.
            # cleft.make_ptable(D=D, kmin=self.ks[0]/self.h,
            #                   kmax=self.ks[-1]/self.h, nk=self.ks.size)
            self.models.append(tmp) #shallow copy to avoid re-running SBF integrals
            if(D==Dz[0]):
                t0iii=time.perf_counter()
        # self.models = np.array(self.pt_table)
        # self.pt_table /= self.h3
        t2=time.perf_counter()
        print("HZPT: time to do 1 pktable: ", t0ii-t0i)
        print("HZPT: time to do 1 copy+append: ", t0iii-t0ii)
        print("HZPT: time to do first model: ", t1-t0)
        print("HZPT: time to do all redshift models: ", t2-t1)
        print("len(k)",len(self.ksh))
    def get_pgg(self, gg_params):
        #Needs to return a pt_table of size nz*nk
        if self.pt_table is None:
            raise ValueError("Please initialise calculator")
        pgg = np.zeros_like(self.pt_table)
        #this is not great, but ok for now
        for i in range(pgg.shape[0]):
            tt = tracers.AutoCorrelator(gg_params[i],self.models[i])
            pgg[i,:] = tt.Power()(self.ksh)/self.h3
        return pgg

    def get_pgm(self, gm_params):

        if self.pt_table is None:
            raise ValueError("Please initialise  calculator")
        pgm = np.zeros_like(self.pt_table)
        #this is not great, but ok for now
        for i in range(pgm.shape[0]):
            tm = tracers.CrossCorrelator(gm_params[i],self.models[i])
            pgm[i,:] = tm.Power()(self.ksh)/self.h3
        return pgm


def get_hzpt_pk2d(cosmo, tracer1, tracer2=None, ptc=None,
                 nonlin_pk_type='nonlinear',
                 extrap_order_lok=1, extrap_order_hik=2):
    """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the PT power spectrum for two quantities defined by
    two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.

    .. note:: see the ".. note::" about IA in lpt.py, ept.py

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        tracer1 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the first
            tracer being correlated.
        ptc (:class:`PTCalculator`): a perturbation theory
            calculator.
        tracer2 (:class:`~pyccl.nl_pt.tracers.PTTracer`): the second
            tracer being correlated. If `None`, the auto-correlation
            of the first tracer will be returned.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
        :class:`~pyccl.nl_pt.power.PTCalculator`: PT Calc [optional]
    """

    if tracer2 is None:
        tracer2 = tracer1
    if not isinstance(tracer1, ccl.nl_pt.PTTracer):
        raise TypeError("tracer1 must be of type `ccl.nl_pt.PTTracer`")
    if not isinstance(tracer2, ccl.nl_pt.PTTracer):
        raise TypeError("tracer2 must be of type `ccl.nl_pt.PTTracer`")

    if not isinstance(ptc, HZPTCalculator):
        raise TypeError("ptc should be of type `HZPTCalculator`")
    # z
    z_arr = 1. / ptc.a_s - 1
    #FIXME: We don't (yet) have redshift evolution of HZPT params...constant for now...

    #FIXME throw an error if tracer1 and tracer2 are NCs without overlap
    if (tracer1.type == 'NC'):
        if (tracer2.type == 'NC'):
            b1 = tracer1.b1(z_arr)
            sngg = tracer1.sngg(z_arr)
            A0gg = tracer1.A0gg(z_arr)
            Rgg = tracer1.Rgg(z_arr)
            R1hgg = tracer1.R1hgg(z_arr)
            p_pt = ptc.get_pgg(np.array([sngg,b1,A0gg,Rgg,R1hgg]).T)
        elif (tracer2.type == 'M'):
            b1 = tracer1.b1(z_arr)
            A0gm = tracer1.A0gm(z_arr)
            Rgm = tracer1.Rgm(z_arr)
            R1hgm = tracer1.R1hgm(z_arr)
            p_pt = ptc.get_pgm(np.array([b1,A0gm,Rgm,R1hgm]).T)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b1 = tracer1.b1(z_arr)
            A0gm = tracer1.A0gm(z_arr)
            Rgm = tracer1.Rgm(z_arr)
            R1hgm = tracer1.R1hgm(z_arr)
            p_pt = ptc.get_pgm(np.array([b1,A0gm,Rgm,R1hgm]).T)
        elif (tracer2.type == 'M'):
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    else:
        raise NotImplementedError("Combination %s-%s not implemented yet" %
                                  (tracer1.type, tracer2.type))


    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    pt_pk = ccl.Pk2D(a_arr=ptc.a_s,
                     lk_arr=np.log(ptc.ks),
                     pk_arr=p_pt,
                     is_logp=False)
    return pt_pk
