## Description

Calculates the structure of an 1D optically thick neutron star expanded envelope in GR.  There is a unique solution for every photospheric radius.
Referencing equations from [Paczynski & Anderson (1986)](http://adsabs.harvard.edu/abs/1986ApJ...302....1P).

Questions and comments -> simon.guichandut@mail.mcgill.ca


## How-to

Parameters are given in `params.txt`
* M : NS mass in solar mass units                                                             
* R : NS radius in km                                                                                        
* y_inner : Column density at the beginning of the envelope in g/cm2 
* comp : Composition of the wind
* mode : Either find the envelope solutions (rootsolve) or use pre-solved roots to make the envelope and produce plots (envelope)
* save : boolean for saving data and plots (0 will show plots in python interface)
* img : image format 


To make plots :

    python Plots.py

Plots and tables are in the `results` subfolder, organized in directories describing the model:
Composition_M/Msun_R/km_log(y_inner)-> e.g `He_1.4_10_8/`

Pre-solved roots are available in the `envelope_solutions` subfolder, organized in directories with the same name as above.


## Example plots

![](/results/He_1.4_12_8/plots/Temperature.png)
![](/results/He_1.4_12_8/plots/Density-Temperature.png)

