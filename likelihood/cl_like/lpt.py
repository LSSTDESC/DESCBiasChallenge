import numpy as np
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
import pyccl as ccl


class LPTCalculator(object):
    """ This class implements a set of methods that can be
    used to compute the various components needed to estimate
    perturbation theory correlations. These calculations are
    currently based on FAST-PT
    (https://github.com/JoeMcEwen/FAST-PT).

    Args:
        with_NC (bool): set to True if you'll want to use
            this calculator to compute correlations involving
            number counts.
        with_IA(bool): set to True if you'll want to use
            this calculator to compute correlations involving
            intrinsic alignments.
        with_dd(bool): set to True if you'll want to use
            this calculator to compute the one-loop matter power
            spectrum.
        log10k_min (float): decimal logarithm of the minimum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        log10k_max (float): decimal logarithm of the maximum
            Fourier scale (in Mpc^-1) for which you want to
            calculate perturbation theory quantities.
        pad_factor (float): fraction of the log(k) interval
            you want to add as padding for FFTLog calculations
            within FAST-PT.
        low_extrap (float): decimal logaritm of the minimum
            Fourier scale (in Mpc^-1) for which FAST-PT will
            extrapolate.
        high_extrap (float): decimal logaritm of the maximum
            Fourier scale (in Mpc^-1) for which FAST-PT will
            extrapolate.
        P_window (array_like or None): 2-element array describing
            the tapering window used by FAST-PT. See FAST-PT
            documentation for more details.
        C_window (float): `C_window` parameter used by FAST-PT
            to smooth the edges and avoid ringing. See FAST-PT
            documentation for more details.
    """
    def __init__(self, log10k_min=-4, log10k_max=2,
                 nk_per_decade=20):
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        self.lpt_table = None

    def update_pk(self, pk, D, h):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        if pk.shape != self.ks.shape:
            raise ValueError("Input spectrum has wrong shape")
        cleft = RKECLEFT(self.ks/h, pk*h**3)
        self.lpt_table = []
        for DD in D:
            cleft.make_ptable(D=DD, kmin=self.ks[0]/h,
                              kmax=self.ks[-1]/h, nk=self.ks.size)
            cleftpk = cleft.pktable
            self.lpt_table.append(cleftpk)
        self.lpt_table = np.array(self.lpt_table)
        self.lpt_table /= h**3

    def get_pgg(self, b11, b21, bs1, b12, b22, bs2):
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")
        Pdmdm = self.lpt_table[:, :,1]
        Pdmd1 = 0.5*self.lpt_table[:, :,2]
        Pdmd2 = 0.5*self.lpt_table[:, :,4]
        Pdmds = 0.5*self.lpt_table[:, :,7]
        Pd1d1 = self.lpt_table[:, :,3]
        Pd1d2 = 0.5*self.lpt_table[:, :,5]
        Pd1ds = 0.5*self.lpt_table[:,:,8]
        # TODO: check normalization
        Pd2d2 = self.lpt_table[:,:,6]
        Pd2ds = 0.5*self.lpt_table[:,:,9]
        Pdsds = self.lpt_table[:,:,10]
    
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
        if self.cleft is None:
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


def get_lpt_pk2d(cosmo, tracer1, tracer2=None, lptc=None,
                 #nonlin_pk_type='nonlinear', # TODO
                 a_arr=None, extrap_order_lok=1, extrap_order_hik=2,
                 return_lptc=False):
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
        sub_lowk (bool): if True, the small-scale white noise
            contribution will be subtracted for number counts
            auto-correlations.
        nonlin_pk_type (str): type of 1-loop matter power spectrum
            to use. 'linear' for linear P(k), 'nonlinear' for the internal
            non-linear power spectrum, 'spt' for standard perturbation
            theory power spectrum. Default: 'nonlinear'.
        a_arr (array): an array holding values of the scale factor
            at which the power spectrum should be calculated for
            interpolation. If `None`, the internal values used by
            `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        return_ia_bb (bool): if `True`, the B-mode power spectrum
            for intrinsic alignments will be returned (if both
            input tracers are of type
            :class:`~pyccl.nl_pt.tracers.PTIntrinsicAlignmentTracer`)
            If `False` (default) E-mode power spectrum is returned.
        return_ia_ee_and_bb (bool): if `True`, the E-mode power spectrum
            for intrinsic alignments will be returned in addition to
            the B-mode one (if both input tracers are of type
            :class:`~pyccl.nl_pt.tracers.PTIntrinsicAlignmentTracer`)
            If `False` (default) E-mode power spectrum is returned.
            Supersedes `return_ia_bb`.
        return_ptc (bool): if `True`, the fastpt object used as the PT
            calculator (ptc) will also be returned. This feature may
            be useful if an input ptc is not specified and one is
            initialized when this function is called. If `False` (default)
            the ptc is not output, whether or not it is initialized as
            part of the function call.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: PT power spectrum.
        :class:`~pyccl.nl_pt.power.PTCalculator`: PT Calc [optional]
    """
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status)

    if tracer2 is None:
        tracer2 = tracer1
    if not isinstance(tracer1, ccl.nl_pt.PTTracer):
        raise TypeError("tracer1 must be of type `ccl.nl_pt.PTTracer`")
    if not isinstance(tracer2, ccl.nl_pt.PTTracer):
        raise TypeError("tracer2 must be of type `ccl.nl_pt.PTTracer`")

    if lptc is None:
        lptc = LPTCalculator()

    if not isinstance(lptc, LPTCalculator):
        raise TypeError("lptc should be of type `LPTCalculator`")
    # z
    z_arr = 1. / a_arr - 1
    if lptc.lpt_table is None:
        # P_lin(k) at z=0
        pk_lin_z0 = ccl.linear_matter_power(cosmo, lptc.ks, 1.)
        D = ccl.growth_factor(cosmo, a_arr)
        # update the PTC to have the require Pk components
        lptc.update_pk(pk_lin_z0, D, cosmo['h'])
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

            p_pt = lptc.get_pgg(b11, b21, bs1, 
                                b12, b22, bs2)
        elif (tracer2.type == 'M'):
            p_pt = lptc.get_pgm(b11, b21, bs1)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))

    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            p_pt = lptc.get_pgm(b12, b22, bs2)
        elif (tracer2.type == 'M'):
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    else:
        raise NotImplementedError("Combination %s-%s not implemented yet" %
                                  (tracer1.type, tracer2.type))

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    lpt_pk = ccl.Pk2D(a_arr=a_arr,
                      lk_arr=np.log(lptc.ks),
                      pk_arr=p_pt,
                      is_logp=False)
    if return_lptc:
        return lpt_pk, lptc
    else:
        return lpt_pk
