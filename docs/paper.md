---
title: 'linea: Fast linear detrending for CHEOPS photometry'
tags:
  - Python
  - astronomy
  - photometry
authors:
  - name: Brett M. Morris
    orcid: 0000-0003-2528-3409
    affiliation: "1"
affiliations:
 - name: Center for Space and Habitability, University of Bern, 
         Gesellschaftsstrasse 6, CH-3012, Bern, Switzerland
   index: 1
date: 18 June 2020
bibliography: paper.bib
---

# Summary

The CHaracterizing ExOPlanets Satellite (CHEOPS) is a 30 cm telescope in orbit 
around the Earth, which seeks to measure properties of exoplanets using 
photometry, or brightness measurements of the planets and their host stars. The
photometry collected with CHEOPS arrives on users machines in the form of FITS 
files which contain photometry that inevitably contains trends as a function of 
several variables; for example, the photometry is often covariant with the 
stellar centroid position on the detector, the roll angle of the spacecraft, 
the flux of contaminants within the aperture, the background and the 
dark current.

We present ``linea``, a linear detrending toolkit for CHEOPS photometry.  
``linea`` features an efficient ``numpy`` implementation of linear least-squares 
regression [@Numpy:2011] which can be used to quickly remove trends from CHEOPS
photometry. The core principle of ``linea`` is that the vector of observed 
fluxes observed by CHEOPS can be represented by a linear combination of some 
observational vectors, like the roll angle and stellar centroid position. 

The ``linea`` API has handy classes and methods for reading in the output files 
produced by the CHEOPS Data Reduction Pipeline (DRP), assembling a design matrix
of the housekeeping vectors, regressing the design matrix against the observed
fluxes, and visualizing the results. We built-in two realistic, simulated 
example light curves and tutorials on how to detrend them including the phase 
curve of 55 Cnc e and four eclipses of WASP-189 b.

The mathematical formalism of the ``linea`` algorithms are detailed in the 
software's documentation. The ``linea`` package is built on the ``astropy`` 
package template [@Astropy:2018]. ``linea`` is open source and contributions are
welcome.

# References

