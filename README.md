# The SHERLOCK PIPEline
<b>S</b>earching for <b>H</b>ints of <b>E</b>xoplanets f<b>R</b>om <b>L</b>ightcurves 
<b>O</b>f spa<b>C</b>e-based see<b>K</b>ers

## Contributors
Active: <i>[Pozuelos](https://github.com/franpoz), 
[Dévora](https://github.com/martindevora) & Thuillier</i> 

Past: <i>García</i>

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
