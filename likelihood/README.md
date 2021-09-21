# Likelihood


## Quick guide

The likelihood code is contained in [cl_like](cl_like)
- [cl_like/ccl.py](cl_like/ccl.py)` is the CCL wrapper that talks to cobaya. **You most probably do not need to touch this**.
- [cl_like/cl_like.py](cl_like/cl_like.py) contains the actual likelihood. This is where most modifications will probably take place (+ any extra files/modules that need to be added).
- `testrun.py` and `test.yml` show how one can run cobaya, and an example config file that links to a data file consisting of a simple linearly biased 3x2pt dataset. `test_abacus.yml` lists the fiducial cosmological parameters that should be used for the Abacus datasets.
- [kmax_test](kmax_test) contains all the scripts used by Nathan in his initial investigations.

## Tips for adding a new bias model
Adding a new bias model should most likely consist of adding a recipe to compute the galaxy-galaxy and galaxy-matter power spectra Pgg(k, z) and Pgm(k, z). These are the most probable steps you need to take in order to do so (all steps consist of modifications to methods in the `ClLike` class defined in [cl_like/cl_like.py](cl_like/cl_like.py):
 1. Modify the `_get_pk_data` method of `ClLike`.
   - Add another branch of the main `if` statement in method for to your new model. As you can see, you'll need to give your model a name (current names are `'Linear'`, `'EulerianPT'` and `'LagrangianPT'`).
   - Inside this `if` clause, add all calculations related to your bias model that depend **only** on cosmological parameters. This function is called by the cobaya CCL theory class only when the cosmological parameters change. If you add a dependence on any other parameters, it will mess up the optimal fast/slow split. You can see an example for the linear bias case (where the only quantity computed is the non-linear matter power spectrum) and for the Eulerian/Lagrangian PT models (where a set of perturbation theory power spectrum templates get precomputed). The cosmological parameters are stored in the CCL `Cosmology` object `cosmo` passed as an argument to this function.
   - Put all elements of your calculation that will be needed later on into a dictionary and return it (e.g. for PT this is the non-linear matter Pk and a PT calculator object).
 2. Modify the `_get_tracers` method of `ClLike`.
   - This method collects all the per-tracer information needed. I.e. this refers to quantities that are associated with a single tracer, and not with a power spectrum (which depends on 2 tracers). Examples of these are the redshift distributions, and the different bias parameters/functions. As you can see, in the case of linear and PT bias models, these are collected into CCL `PTNumberCountsTracer` (biases) and `NumberCountsTracer` (N(z)s) objects. If you need a different structure for your model, feel free to ask for help.
   - Add another branch to the `if` statement corresponding to `q == 'galaxy_density'` for your model. You should probably not need to modify any of the calculations corresponding to `'galaxy_shear'` or `'cmb_convergence'`, but ask for help otherwise.
 3. Modify the `_get_pkxy` method of `ClLike`.
   - This method returns the 3D power spectrum (P(k)) needed to compute the C_ell between two tracers (i.e. Pgg(k), Pgm(k) or Pmm(k)). The return value should be a ccl [Pk2D](https://ccl.readthedocs.io/en/latest/api/pyccl.pk2d.html) object, which is basically a glorified 2D interpolator.
   - Add a branch to the main `if` statement in this method for your bias model, and include all necessary calculations leading to the final `Pk2D` object there. You can access the result of `_get_pk_data` (which you possibly modified as described above) in the `pkd` input argument. The tracer information collected by `_get_tracers` is available in `trs`.

In most cases, it should be helpful to try and understand how the PT bias models are implemented, since the steps to implement most other models should be very similar. Feel free to ask for help if needed though!
