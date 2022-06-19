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
        self.lpt_table[:, :, 1:] /= self.h**3

    def get_pgg(self, Pnl, b11, b21, bs1, b12, b22, bs2, b3nl1=None, b3nl2=None,
                bk21=None, bk22=None, bsn1=None, bsn2=None, bsnx=None, Pgrad=None):
        # Clarification:
        # CLEFT uses the followint expansion for the galaxy overdensity:
        #   d_g = b1 d + b2 d2^2/2 + bs s^2
        # (see Eq. 4.4 of https://arxiv.org/pdf/2005.00523.pdf).
        # To add to the confusion, this is different from the prescription
        # used by EPT, where s^2 is divided by 2 :-|
        #
        # The LPT table below contains the following power spectra
        # in order:
        #  <1,1>
        #  2*<1,d>
        #  <d,d>
        #  2*<1,d^2/2>
        #  2*<d,d^2/2>
        #  <d^2/2,d^2/2> (!)
        #  2*<1,s^2>
        #  2*<d,s^2>
        #  2*<d^2/2,s^2> (!)
        #  <s^2,s^2> (!)
        #  2*<1, O3>
        #  2*<d, O3>
        #
        # EPT uses:
        #   d_g = b1 d + b2 d2^2/2 + bs s^2/2 + b3 psi/2 + bnabla nablad/2
        # So:
        #   a) The cross-correlations need to be divided by 2.
        #   b) The spectra involving b2 are for d^2/2, NOT d^2!!
        #   c) The spectra invoving bs are for s^2, NOT s^2/2!!
        #   d) The spectra involving b3 are for O3 - convert to O3/2
        #   e) The spectra involving bnabla are for nablad - convert to nablad/2
        # Also, the spectra marked with (!) tend to a constant
        # as k-> 0, which we can suppress with a low-pass filter.
        #
        # Importantly, we have corrected the spectra involving s2 to
        # make the definition of bs equivalent in the EPT and LPT
        # expansions.

        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")

        b1_list = [b11, b21, bs1, bk21, bsn1, b3nl1]
        b2_list = [b12, b22, bs2, bk22, bsn2, b3nl2]

        cross = True
        if np.all([np.all(b1_list[i] == b2_list[i]) for i in range(len(b1_list))]):
            cross = False

        bL11 = b11-1
        bL12 = b12-1
        if Pnl is None:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]
            Pd1d1 = self.lpt_table[:, :, 3]
            pgg = (Pdmdm + (bL11+bL12)[:, None] * Pdmd1 +
                   (bL11*bL12)[:, None] * Pd1d1)
        else:
            pgg = (b11*b12)[:, None]*Pnl
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pd1d2 = 0.5*self.lpt_table[:, :, 5]
        Pd2d2 = self.lpt_table[:, :, 6]*self.wk_low[None, :]
        Pdms2 = 0.25*self.lpt_table[:, :, 7]
        Pd1s2 = 0.25*self.lpt_table[:, :, 8]
        Pd2s2 = 0.25*self.lpt_table[:, :, 9]*self.wk_low[None, :]
        Ps2s2 = 0.25*self.lpt_table[:, :, 10]*self.wk_low[None, :]
        Pdmo3 = 0.25 * self.lpt_table[:, :, 11]
        Pd1o3 = 0.25 * self.lpt_table[:, :, 12]
        if Pgrad is not None:
            Pd1k2 = 0.5*Pgrad * (self.ks**2)[None, :]
        else:
            Pdmk2 = 0.5*Pdmdm * (self.ks**2)[None, :]
            Pd1k2 = 0.5*Pdmd1 * (self.ks**2)[None, :]
            Pd2k2 = Pdmd2 * (self.ks**2)[None, :]
            Ps2k2 = Pdms2 * (self.ks**2)[None, :]
            Pk2k2 = 0.25*Pdmdm * (self.ks**4)[None, :]

        if b3nl1 is None:
            b3nl1 = np.zeros_like(self.a_s)
        if b3nl2 is None:
            b3nl2 = np.zeros_like(self.a_s)
        if bk21 is None:
            bk21 = np.zeros_like(self.a_s)
        if bk22 is None:
            bk22 = np.zeros_like(self.a_s)
        if bsn1 is None:
            bsn1 = np.zeros_like(self.a_s)
        if bsn2 is None:
            bsn2 = np.zeros_like(self.a_s)
        if bsnx is None:
            bsnx = np.zeros_like(self.a_s)

        pgg += ((b21 + b22)[:, None] * Pdmd2 +
                (bs1 + bs2)[:, None] * Pdms2 +
                (bL11*b22 + bL12*b21)[:, None] * Pd1d2 +
                (bL11*bs2 + bL12*bs1)[:, None] * Pd1s2 +
                (b21*b22)[:, None] * Pd2d2 +
                (b21*bs2 + b22*bs1)[:, None] * Pd2s2 +
                (bs1*bs2)[:, None] * Ps2s2 +
                (b3nl1 + b3nl2)[:, None] * Pdmo3 +
                (bL11*b3nl2 + bL12*b3nl1)[:, None] * Pd1o3)

        if Pgrad is not None:
            pgg += (b12*bk21+b11*bk22)[:, None] * Pd1k2
        else:
            pgg += ((bk21 + bk22)[:, None] * Pdmk2 +
                    (bL12 * bk21 + bL11 * bk22)[:, None] * Pd1k2 +
                    (b22 * bk21 + b21 * bk22)[:, None] * Pd2k2 +
                    (bs2 * bk21 + bs1 * bk22)[:, None] * Ps2k2 +
                    (bk21 * bk22)[:, None] * Pk2k2)

        if not cross:
            pgg += bsn1[:, None]
        else:
            pgg += bsnx[:, None]

        return pgg

    def get_pgm(self, Pnl, b1, b2, bs, b3nl=None,
                bk2=None, bsn=None, Pgrad=None):
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")

        if bk2 is None:
            bk2 = np.zeros_like(self.a_s)
        if bsn is None:
            bsn = np.zeros_like(self.a_s)

        bL1 = b1-1
        if Pnl is None:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5*self.lpt_table[:, :, 2]
            pgm = Pdmdm + bL1[:, None] * Pdmd1
        else:
            pgm = b1[:, None]*Pnl
        Pdmd2 = 0.5*self.lpt_table[:, :, 4]
        Pdms2 = 0.25*self.lpt_table[:, :, 7]
        Pdmo3 = 0.25 * self.lpt_table[:, :, 11]

        if Pgrad is not None:
            Pdmk2 = 0.5*Pgrad * (self.ks**2)[None, :]
        else:
            Pdmk2 = 0.5*Pdmdm * (self.ks**2)[None, :]

        pgm += (b2[:, None] * Pdmd2 +
                bs[:, None] * Pdms2 +
                b3nl[:, None] * Pdmo3 +
                bk2[:, None] * Pdmk2)

        return pgm

    def get_pmm(self):
        if self.lpt_table is None:
            raise ValueError("Please initialise CLEFT calculator")

        pmm = self.lpt_table[:, :, 1]

        return pmm


def get_lpt_pk2d(cosmo, tracer1, tracer2=None, ptc=None, bsnx=None,
                 nonlin_pk_type='nonlinear',
                 nonloc_pk_type='spt',
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
        nonlin_pk_type (str): type of 1-loop matter power spectrum
            to use. 'linear' for linear P(k), 'nonlinear' for the internal
            non-linear power spectrum, 'spt' for standard perturbation
            theory power spectrum. Default: 'nonlinear'.
        nonloc_pk_type (str): type of "non-local" matter power spectrum
            to use (i.e. the cross-spectrum between the overdensity and
            its Laplacian divided by :math:`k^2`). Same options as
            `nonlin_pk_type`. Default: 'nonlinear'.
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

    if nonlin_pk_type == 'nonlinear':
        Pnl = np.array([ccl.nonlin_matter_power(cosmo, ptc.ks, a)
                        for a in ptc.a_s])
    elif nonlin_pk_type == 'linear':
        Pnl = np.array([ccl.linear_matter_power(cosmo, ptc.ks, a)
                        for a in ptc.a_s])
    elif nonlin_pk_type == 'spt':
        Pnl = None
    else:
        raise NotImplementedError("Nonlinear option %s not implemented yet" %
                                  (nonlin_pk_type))

    Pgrad = None
    if (((tracer1.type == 'NC') or (tracer2.type == 'NC')) and
            (nonloc_pk_type != nonlin_pk_type)):
        if nonloc_pk_type == 'nonlinear':
            Pgrad = np.array([ccl.nonlin_matter_power(cosmo, ptc.ks, a)
                              for a in ptc.a_s])
        elif nonloc_pk_type == 'linear':
            Pgrad = np.array([ccl.linear_matter_power(cosmo, ptc.ks, a)
                              for a in ptc.a_s])
        elif nonloc_pk_type == 'spt':
            Pgrad = None
        elif nonloc_pk_type == 'lpt':
            Pgrad = None
        else:
            raise NotImplementedError("Non-local option %s "
                                      "not implemented yet" %
                                      (nonloc_pk_type))

    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        if hasattr(tracer1, 'b3nl'):
            b31 = tracer1.b3nl(z_arr)
        else:
            b31 = None
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
            if hasattr(tracer2, 'b3nl'):
                b32 = tracer2.b3nl(z_arr)
            else:
                b32 = None
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

            p_pt = ptc.get_pgg(Pnl,
                               b11, b21, bs1,
                               b12, b22, bs2,
                               b31, b32,
                               bk21, bk22,
                               bsn1, bsn2, bsnx,
                               Pgrad)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(Pnl, b11, b21, bs1, b31,
                bk21, bsn1, Pgrad)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            if hasattr(tracer2, 'b3nl'):
                b32 = tracer2.b3nl(z_arr)
            else:
                b32 = None
            if hasattr(tracer2, 'bk2'):
                bk22 = tracer2.bk2(z_arr)
            else:
                bk22 = None
            if hasattr(tracer2, 'sn'):
                bsn2 = tracer2.sn(z_arr)
            else:
                bsn2 = None

            p_pt = ptc.get_pgm(Pnl, b12, b22, bs2, b32,
                bk22, bsn2, Pgrad)
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
