#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:21:36 2021

@author: nathanfindlay
"""
import pyccl.nl_pt as pt

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
    elif self.bz_model == 'Eulerian_PT':
        cosmo.compute_nonlin_power()
        pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
        ptc = pt.PTCalculator(with_NC=True, with_IA=False,
                              log10k_min=-4, log10k_max=2, nk_per_decade=20)
        return {'ptc': ptc, 'pk_mm': pkmm}
    else:
        raise LoggedError(self.log, "Unknown bias model %s" % self.bz_model)

def _get_pkxy(self, cosmo, clm, pkd, **pars):
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
    elif (self.bz_model == 'Eulerian_PT'):
        b_1 = 2.0
        b_2 = 1.0
        b_s = 1.0
        if (q1 == 'galaxy_density') and (q2 == 'galaxy_density'):
            ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s)
            pk_gg = pt.get_pt_pk2d(cosmo, ptt_g, ptc=pkd['ptc'])
            return pk_gg  # galaxy-galaxy
        elif ((q1 != 'galaxy_density') and (q2 != 'galaxy_density')):
            return pkd['pk_mm']  # matter-matter
        else:
            ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s)
            ptt_m = pt.PTMatterTracer()
            pk_gm = pt.get_pt_pk2d(cosmo, ptt_g, tracer2=ptt_m, ptc=pkd['ptc'])
            return pk_gm  # galaxy-matter
    else:
        raise LoggedError(self.log, "Unknown bias model %s" % self.bz_model)