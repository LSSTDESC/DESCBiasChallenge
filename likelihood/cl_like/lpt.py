import numpy as np
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
import pyccl as ccl


class LPTCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations. These calculations are
    currently based on velocileptors
    (https://github.com/sfschen/velocileptors).

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
        if a_arr is None:
            a_arr = 1./(1+np.linspace(0., 4., 30)[::-1])
        self.a_s = a_arr
        self.h = h
        self.lpt_table = None
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
        cleft = RKECLEFT(self.ks/self.h, pk*self.h**3)
        self.lpt_table = []
        for D in Dz:
            cleft.make_ptable(D=D, kmin=self.ks[0]/self.h,
                              kmax=self.ks[-1]/self.h, nk=self.ks.size)
            self.lpt_table.append(cleft.pktable)
        self.lpt_table = np.array(self.lpt_table)
        self.lpt_table /= self.h**3

    def get_pgg(self, b11, b21, bs1, b12, b22, bs2):
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")
        Pdmdm = self.lpt_table[:, :, 1]
        Pdmd1 = 0.5*self.lpt_table[:, :, 2]
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pdmds = 0.5*self.lpt_table[:, :, 7]
        Pd1d1 = self.lpt_table[:, :, 3]
        Pd1d2 = 0.5*self.lpt_table[:, :, 5]
        Pd1ds = 0.5*self.lpt_table[:, :, 8]
        # TODO: check normalization
        Pd2d2 = self.lpt_table[:,:,6]*self.wk_low[None, :]
        Pd2ds = 0.5*self.lpt_table[:,:,9]*self.wk_low[None, :]
        Pdsds = self.lpt_table[:,:,10]*self.wk_low[None, :]
    
        pgg = (Pdmdm + 
               (b11 + b12)[:, None] * Pdmd1 +
               (b21 + b22)[:, None] * Pdmd2 +
               (bs1 + bs2)[:, None] * Pdmds +
               (b11*b12)[:, None] * Pd1d1 +
               (b11*b22 + b12*b21)[:, None] * Pd1d2 +
               (b11*bs2 + b12*bs1)[:, None] * Pd1ds +
               (b21*b22)[:, None] * Pd2d2 +
               (b21*bs2 + b22*bs1)[:, None] * Pd2ds +
               (bs1*bs2)[:, None] * Pdsds)
    
        return pgg

    def get_pgm(self, b1, b2, bs):
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")

        Pdmdm = self.lpt_table[:,:,1]
        Pdmd1 = 0.5*self.lpt_table[:,:,2]
        Pdmd2 = 0.5*self.lpt_table[:,:,4]
        Pdmds = 0.5*self.lpt_table[:,:,7]

        pgm = (Pdmdm + 
               b1[:, None] * Pdmd1 + 
               b2[:, None] * Pdmd2 + 
               bs[:, None] * Pdmds)
    
        return pgm


def get_lpt_pk2d(cosmo, tracer1, tracer2=None, ptc=None,
                 #nonlin_pk_type='nonlinear', # TODO
                 extrap_order_lok=1, extrap_order_hik=2):
    """Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the PT power spectrum for two quantities defined by
    two :class:`~pyccl.nl_pt.tracers.PTTracer` objects.

    .. note:: The full non-linear model for the cross-correlation
              between number counts and intrinsic alignments is
              still work in progress in FastPT. As a workaround
              CCL assumes a non-linear treatment of IAs, but only
              linearly biased number counts.

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

    if not isinstance(ptc, LPTCalculator):
        raise TypeError("ptc should be of type `LPTCalculator`")
    # z
    z_arr = 1. / ptc.a_s - 1

    '''
    if nonlin_pk_type == 'nonlinear':
        Pd1d1 = np.array([nonlin_matter_power(cosmo, ptc.ks, a)
                          for a in a_arr]).T
    elif nonlin_pk_type == 'linear':
        Pd1d1 = np.array([linear_matter_power(cosmo, ptc.ks, a)
                          for a in a_arr]).T
    else:
        raise NotImplementedError("Nonlinear option %s not implemented yet" %
                                  (nonlin_pk_type))
    '''

    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)

            p_pt = ptc.get_pgg(b11, b21, bs1, 
                               b12, b22, bs2)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(b11, b21, bs1)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))

    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            p_pt = ptc.get_pgm(b12, b22, bs2)
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
