# Likelihood

The likelihood code is contained in `cl_like`
- `cl_like/ccl.py` is the CCL wrapper that talks to cobaya.
- `cl_like/cl_like.py` contains the actual likelihood. This is where the different bias models will eventually need to live.
- `testrun.py` and `test.yml` show how one can run cobaya, and an example config file that links to a data file consisting of a simple linearly biased 3x2pt dataset.
