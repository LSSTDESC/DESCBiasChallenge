# The data

## Cosmological models
Data were generated for two different cosmological models. All non-Abacus datasets were generated for a simple fiducial LCDM model (Omega_c=0.25, Omega_b=0.05 h=0.7, sigma_8=0.81, n_s=0.96, Eisenstein & Hu linear power spectrum).

## Galaxy samples
In all cases the shear sample (number density and redshift distributions) is roughly modelled after the SRD, and split into 5 broad redshift bins.

We have two choices for clustering samples:
 - "Red sample": red galaxies with higher bias, lower number density and better photo-z accuracy (sharper N(z)s). We split these into 6 redshift bins.
 - "HSC sample" or "shear sample": a broader galaxy sample mimicking the shear sample in number density and redshift distribution, with lower bias and poorer photo-z uncertainty.

## Data generation
[AbacusData.ipynb](AbacusData.ipynb) describes the manipulations done on the raw Abacus simulation power spectra to turn them into smooth P(k)s ready for Limber integration.

[datagen.py](datagen.py) is the program used to generate the different synthetic data vectors as sacc files.

[saccplotter.py](saccplotter.py) is a rudimentary program that can be used to visualize the contents of a sacc file.

[README_data.md](README_data.md) contains a brief description of the different data vectors generated so far.

## Getting the data
You can generate the data yourself, or it can be downloaded from [this link](http://intensitymapping.physics.ox.ac.uk/Data/data_DESCBiasChallenge.tar.gz).

E.g. run, from this folder:

```
wget intensitymapping.physics.ox.ac.uk/Data/data_DESCBiasChallenge.tar.gz
tar -xvf data_DESCBiasChallenge.tar.gz
```
