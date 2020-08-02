<p align="center">
  <img width="500" src="logo/sherlock2.png">
</p>

# The SHERLOCK PIPEline

The <b>SHERLOCK</b> (<b>S</b>earching for <b>H</b>ints of <b>E</b>xoplanets f<b>R</b>om <b>L</b>ightcurves 
<b>O</b>f spa<b>C</b>e-based see<b>K</b>ers) <b>PIPE</b>line is a user-friendly pipeline, which
minimizes the interaction of the user to the minimum when using data coming from Kepler or TESS missions. SHERLOCK makes use of previous well-known and well-tested codes which allow the exoplanets community to explore the public data from space-based missions without need of a deep knowledge of how the data are build and stored. 
In most of cases the user only needs to provide with a KOI-ID, EPIC-ID, TIC-ID or coordinates of the host star where wants to search for exoplanets.


## Main Developers
Active: <i>[F.J. Pozuelos](https://github.com/franpoz), 
[M. Dévora](https://github.com/martindevora)</i> 

## Additional contributors 
<i>A. Thuillier</i> & <i>[L. García](https://github.com/LionelGarcia)</i>

## Dependencies
The next Python libraries are <b>required</b> for <i>SHERLOCK</i> to be run:
* numpy
    * sudo apt-get install libblas-dev  liblapack-dev
    * sudo apt-get install gfortran
* cython (for lightkurve and pandas dependencies)
* pandas
* lightkurve
* transitleastsquares
* eleanor
* wotan
* matplotlib
* plotly
* lxml

The next libraries are <b>required</b> for <i>SHERLOCK Explorer</i> to be run:
* colorama

## Integration
SHERLOCK integrates with several third party services. Some of them are listed below:
* TESS, Kepler and K2 databases through [Lightkurve](https://github.com/KeplerGO/lightkurve) and 
[ELEANOR](https://adina.feinste.in/eleanor/)
* MAST and Vizier catalogs through [Lightkurve](https://github.com/KeplerGO/lightkurve)
* [NASA Exoplanet Archive API](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html)
* [TESS ExoFOP](https://exofop.ipac.caltech.edu/tess/view_toi.php)
