import numpy as np
from .. import ccllib as lib
from ..core import check
from ..pk2d import Pk2D
from ..power import linear_matter_power, nonlin_matter_power
from ..background import growth_factor
from .tracers import PTTracer
from velocileptors.LPT.cleft_fftw import CLEFT

try:
    import fastpt as fpt
    HAVE_FASTPT = True
except ImportError:
    HAVE_FASTPT = False


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
    def __init__(self, with_NC=False, with_IA=False, with_dd=True,
                 log10k_min=-4, log10k_max=2, nk_per_decade=20,
                 pad_factor=1, low_extrap=-5, high_extrap=3,
                 P_window=None, C_window=.75):
        assert HAVE_FASTPT, (
            "You must have the `FAST-PT` python package "
            "installed to use CCL to get PT observables! "
            "You can install it with pip install fast-pt.")

        self.with_dd = with_dd
        self.with_NC = with_NC
        self.with_IA = with_IA
        self.P_window = P_window
        self.C_window = C_window

        to_do = ['one_loop_dd']
        if self.with_NC:
            to_do.append('dd_bias')
        if self.with_IA:
            to_do.append('IA')

        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        self.ks = np.logspace(log10k_min, log10k_max, nk_total)
        n_pad = int(pad_factor * len(self.ks))

        self.pt = fpt.FASTPT(self.ks, to_do=to_do,
                             low_extrap=low_extrap,
                             high_extrap=high_extrap,
                             n_pad=n_pad)
        self.one_loop_dd = None
        self.dd_bias = None
        self.ia_ta = None
        self.ia_tt = None
        self.ia_mix = None

    def update_pk(self, pk):
        """ Update the internal PT arrays.

        Args:
            pk (array_like): linear power spectrum sampled at the
                internal `k` values used by this calculator.
        """
        if pk.shape != self.ks.shape:
            raise ValueError("Input spectrum has wrong shape")
        if self.with_NC:
            self._get_dd_bias(pk)
            self.with_dd = True
        elif self.with_dd:
            self._get_one_loop_dd(pk)
        if self.with_IA:
            self._get_ia_bias(pk)

    def _get_one_loop_dd(self, pk):
        # Precompute quantities needed for one-loop dd
        # power spectra. Only needed if dd_bias is not called.
        self.one_loop_dd = self.pt.one_loop_dd(pk,
                                               P_window=self.P_window,
                                               C_window=self.C_window)

    def _get_dd_bias(self, pk):
        # Precompute quantities needed for number counts
        # power spectra.
        self.dd_bias = self.pt.one_loop_dd_bias(pk,
                                                P_window=self.P_window,
                                                C_window=self.C_window)
        self.one_loop_dd = self.dd_bias[0:1]

    def _get_ia_bias(self, pk):
        # Precompute quantities needed for intrinsic alignment
        # power spectra.
        self.ia_ta = self.pt.IA_ta(pk,
                                   P_window=self.P_window,
                                   C_window=self.C_window)
        self.ia_tt = self.pt.IA_tt(pk,
                                   P_window=self.P_window,
                                   C_window=self.C_window)
        self.ia_mix = self.pt.IA_mix(pk,
                                     P_window=self.P_window,
                                     C_window=self.C_window)


    def get_pgg(self, cleft, b11, b21, bs1, b12, b22, bs2):
    
        Pdmdm = cleft.pktable[:,1]
        Pdmd1 = 0.5*cleft.pktable[:,2]
        Pdmd2 = 0.5*cleft.pktable[:,4]
        Pdmds = 0.5*cleft.pktable[:,7]
        Pd1d1 = cleft.pktable[:,3]
        Pd1d2 = 0.5*cleft.pktable[:,5]
        Pd1ds = 0.5*cleft.pktable[:,8]
        Pd2d2 = cleft.pktable[:,6]
        Pd2ds = 0.5*cleft.pktable[:,9]
        Pdsds = cleft.pktable[:,10]
    
        pgg = (Pdmdm + 
              (b11 + b12)[None, :] * Pdmd1 +
              (b21 + b22)[None, :] * Pdmd2 +
              (bs1 + bs2)[None, :] * Pdmds +
              (b11*b12)[None, :] * Pd1d1 +
              (b11*b22 + b12*b21)[None, :] * Pd1d2 +
              (b21*b22)[None, :] * Pd2d2 +
              (b11*bs2 + b12*bs1)[None, :] * Pd1ds +
              (b21*bs2 + b22*bs1)[None, :] * Pd2ds +
              (bs1*bs2)[None, :] * Ps2ds)
    
        return pgg


    def get_pgm(self, cleft, b1, b2, bs):

        Pdmdm = cleft.pktable[:,1]
        Pdmd1 = 0.5*cleft.pktable[:,2]
        Pdmd2 = 0.5*cleft.pktable[:,4]
        Pdmds = 0.5*cleft.pktable[:,7]

        pgm = (Pdmdm + 
              b1[None, :] * Pdmd1 + 
              b2[None, :] * Pdmd2 + 
              bs[None, :] * Pdmds)
    
        return pgm


    def get_pmm(self, Pd1d1_lin, g4):
        """ Get the one-loop matter power spectrum.

        Args:
            Pd1d1_lin (array_like): 1-loop linear matter power spectrum
                at the wavenumber values given by this object's
                `ks` list.
            g4 (array_like): fourth power of the growth factor at
                a number of redshifts.

        Returns:
            array_like: 2D array of shape `(N_k, N_z)`, where `N_k` \
                is the size of this object's `ks` attribute, and \
                `N_z` is the size of the input redshift-dependent \
                biases and growth factor.
        """
        P1loop = g4[None, :] * self.one_loop_dd[0][:, None]
        pmm = (Pd1d1_lin + P1loop)
        return pmm


def get_pt_pk2d(cosmo, tracer1, tracer2=None, ptc=None,
                sub_lowk=False, nonlin_pk_type='nonlinear',
                a_arr=None, extrap_order_lok=1, extrap_order_hik=2,
                return_ia_bb=False, return_ia_ee_and_bb=False,
                return_ptc=False):
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
    if not isinstance(tracer1, PTTracer):
        raise TypeError("tracer1 must be of type `PTTracer`")
    if not isinstance(tracer2, PTTracer):
        raise TypeError("tracer2 must be of type `PTTracer`")

    if ptc is None:
        with_NC = ((tracer1.type == 'NC')
                   or (tracer2.type == 'NC'))
        with_IA = ((tracer1.type == 'IA')
                   or (tracer2.type == 'IA'))
        with_dd = nonlin_pk_type == 'spt'
        ptc = LPTCalculator(with_dd=with_dd,
                           with_NC=with_NC,
                           with_IA=with_IA)
    if not isinstance(ptc, LPTCalculator):
        raise TypeError("ptc should be of type `LPTCalculator`")

    if (tracer1.type == 'NC') or (tracer2.type == 'NC'):
        if not ptc.with_NC:
            raise ValueError("Need number counts bias, "
                             "but calculator didn't compute it")
    if (tracer1.type == 'IA') or (tracer2.type == 'IA'):
        if not ptc.with_IA:
            raise ValueError("Need intrinsic alignment bias, "
                             "but calculator didn't compute it")
    if nonlin_pk_type == 'spt':
        if not ptc.with_dd:
            raise ValueError("Need 1-loop matter power spectrum, "
                             "but calculator didn't compute it")

    if return_ia_ee_and_bb:
        return_ia_bb = True

    # z
    z_arr = 1. / a_arr - 1
    # P_lin(k) at z=0
    pk_lin_z0 = linear_matter_power(cosmo, ptc.ks, 1.)

    cleft = CLEFT(ptc.ks, pz_lin_z0)
    cleft.make_ptable(kmin=1e-3,kmax=1.0,nk=500)

    # update the PTC to have the require Pk components
    ptc.update_pk(pk_lin_z0)

    if nonlin_pk_type == 'nonlinear':
        Pd1d1 = np.array([nonlin_matter_power(cosmo, ptc.ks, a)
                          for a in a_arr]).T
    elif nonlin_pk_type == 'linear':
        Pd1d1 = np.array([linear_matter_power(cosmo, ptc.ks, a)
                          for a in a_arr]).T
    elif nonlin_pk_type == 'spt':
        pklin = np.array([linear_matter_power(cosmo, ptc.ks, a)
                          for a in a_arr]).T
        Pd1d1 = ptc.get_pmm(pklin, ga4)
    else:
        raise NotImplementedError("Nonlinear option %s not implemented yet" %
                                  (nonlin_pk_type))

    if (tracer1.type == 'NC'):
        b11 = tracer1.b1(z_arr)
        b21 = tracer1.b2(z_arr)
        bs1 = tracer1.bs(z_arr)
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)

            p_pt = ptc.get_pgg(cleft,
                               b11, b21, bs1, 
                               b12, b22, bs2)
        elif (tracer2.type == 'M'):
            p_pt = ptc.get_pgm(cleft, 
                               b11, b21, bs1)
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))

    elif (tracer1.type == 'M'):
        if (tracer2.type == 'NC'):
            b12 = tracer2.b1(z_arr)
            b22 = tracer2.b2(z_arr)
            bs2 = tracer2.bs(z_arr)
            p_pt = ptc.get_pgm(cleft, 
                               b12, b22, bs2)
        elif (tracer2.type == 'M'):
            p_pt = Pd1d1
        else:
            raise NotImplementedError("Combination %s-%s not implemented yet" %
                                      (tracer1.type, tracer2.type))
    else:
        raise NotImplementedError("Combination %s-%s not implemented yet" %
                                  (tracer1.type, tracer2.type))

    # Once you have created the 2-dimensional P(k) array,
    # then generate a Pk2D object as described in pk2d.py.
    if return_ia_ee_and_bb:
        pt_pk_ee = Pk2D(a_arr=a_arr,
                        lk_arr=np.log(ptc.ks),
                        pk_arr=p_pt[0].T,
                        is_logp=False)
        pt_pk_bb = Pk2D(a_arr=a_arr,
                        lk_arr=np.log(ptc.ks),
                        pk_arr=p_pt[1].T,
                        is_logp=False)
        if return_ptc:
            return pt_pk_ee, pt_pk_bb, ptc
        else:
            return pt_pk_ee, pt_pk_bb
    else:
        pt_pk = Pk2D(a_arr=a_arr,
                     lk_arr=np.log(ptc.ks),
                     pk_arr=p_pt.T,
                     is_logp=False)
        if return_ptc:
            return pt_pk, ptc
        else:
            return pt_pk
