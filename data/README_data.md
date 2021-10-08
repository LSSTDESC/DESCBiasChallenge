1. `fid_shear_const.fits`
 - Constant linear bias b=1.
 - Same bins for shear and clustering.
 - High-density clustering.

2. `fid_red_const.fits`
 - Constant linear bias b=2.
 - Different bins for shear and clustering.
 - Low-density clustering.

3. `fid_HSC_linear.fits`
 - Evolving linear bias b=0.95/D(z) (Nicola et al.)
 - Same bins for shear and clustering.
 - High-density clustering.

4. `fid_red_linear.fits`
 - Evolving linear bias b=1.5/D(z) (Zhou et al.)
 - Different bins for shear and clustering.
 - Low-density clustering.

5. `fid_HSC_HOD.fits`
 - HOD model (Nicola et al.)
 - Same bins for shear and clustering.
 - High-density clustering.

6. `fid_red_HOD.fits`
 - HOD model (Zhou et al.)
 - Different bins for shear and clustering.
 - Low-density clustering.

7. `abacus_HSC_abacus.fits`
 - Cosmology from Abacus simulation
 - HSC-like galaxies
 - Same bins for shear and clustering.
 - High-density clustering.

8. `abacus_red_abacus.fits`
 - Cosmology from Abacus simulation
 - Red-like galaxies
 - Different bins for shear and clustering.
 - Low-density clustering.

9. `abacus_red_AB_abacus.fits`
 - Cosmology from Abacus simulation
 - Red-like galaxies, with assembly bias
 - Different bins for shear and clustering
 - Low-density clustering.



The "HSC" and "red" HOD models used above follow the following prescription:
 - The parametrization is defined in the docstring of this [function](https://ccl.readthedocs.io/en/latest/api/pyccl.halos.profiles.html#pyccl.halos.profiles.HaloProfileHOD)
 - The only time-varying quantities are Mmin, M0 and M1. alpha and sigma_lnM are constant, and f_c is fixed to 1.
   - log10(M_min) = lMmin_0 + lMmin_p * (1/(1+z) - 1/(1+z_pivot))
   - log10(M_0) = lM0_0 + lM0_p * (1/(1+z) - 1/(1+z_pivot))
   - log10(M_1) = lM1_0 + lM1_p * (1/(1+z) - 1/(1+z_pivot))
   - sigma_lnM = constant
   - alpha = constant
   - f_c = 1
   - z_pivot = 0.65
 - The "HSC" parameters are based on [Nicola et al. 2019](https://arxiv.org/abs/1912.08209):
   - lMmin_0 = 11.88
   - lMmin_p = -0.5
   - lM0_0 = 11.88
   - lM0_p = -0.5
   - lM1_0 = 13.08
   - lM1_p = 0.9
   - alpha = 1
   - sigma_lnM = 0.4
 - The "red" parameters are based on [Zhou et al. 2020](https://arxiv.org/abs/2001.06018):
   - lMmin_0 = 12.95
   - lMmin_p = -2.0
   - lM0_0 = 12.3
   - lM0_p = 0.0
   - lM1_0 = 14.0
   - lM1_p = -1.5
   - alpha = 1.32
   - sigma_lnM = 0.25
 - I suggested using M200m as mass definition, but Boryana could only use Mvir as defined in Bryan and Norman (1998) for the Abacus sample.
