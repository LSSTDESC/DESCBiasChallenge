# kmax_test

Perform chi^2 minimisation of cosmological and bias parameters for given models at different minimum scales (kmax).

------------------------
## kmax_test.py 
To run: `python3 kmax_test.py <model> <input_data> <kmax>`

`<model>`is the bias model used to fit to the input data:
- `LIN` uses a linear bias model
- `EPT` uses Eulerian Perturbation Theory 
- `LPT` uses Lagrangian Perturbation Theory

`<input_data>` is the input data to fit:
- `red_const`,`red_linear`,`red_HOD`,`red_abacus`, `red_AB_abacus`
- `shear_const`,`HSC_linear`,`HSC_HOD`,`HSC_abacus`

Input data files must be downloaded to `DESCBiasChallenge/data`.

`kmax` is the chosen minimum scale up to which the fit is performed.

Results for each run are output to the corresponding folder in `kmax_test/results`.

-------------------------
## kmax_test.yml
Contains information required for the cobaya minimiser.
Cosmological parameter priors can be changed here.
_________________________
## param_plotter.py
Used to plot the variation with kmax for all parameters with the models.

After running kmax_test.py for chosen kmax range with all the models do: `python3 param_plotter.py <input_data>`

Seperate kmax results for each model will be compiled into single data files and all models will be plotted on the same figure.
