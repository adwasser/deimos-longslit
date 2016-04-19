# DEIMOS-LONGSLIT #
Quick DEIMOS longslit data reduction.

Contact: Asher Wasserman, adwasser@ucsc.edu

What this does:

* combines flat and bias frames
* obtains a trace of a continuum source
* extracts a spectrum and subtracts sky flux
* calibrates wavelength solution

What this does not do (yet!):

* any quality checks
* flux calibration

Dependencies:

* Numpy
* Scipy
* Astropy
* astroscrappy (https://github.com/astropy/astroscrappy)