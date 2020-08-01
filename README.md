# <img src="logo/sherlock.png"  alt="SHERLOCK"  style="float: left; margin-right: 10px;" />

# The SHERLOCK PIPEline

The SHERLOCK (<b>S</b>earching for <b>H</b>ints of <b>E</b>xoplanets f<b>R</b>om <b>L</b>ightcurves 
<b>O</b>f spa<b>C</b>e-based see<b>K</b>ers) <b>pipe</b>line is a ready-to-use and user-friendly pipeline, which
minimizes the interaction of the user to the minimum. SHERLOCK made use of previous well-known and well-tested codes which allow the exoplanets community 
to explore the public data from Kepler and TESS missions without need of a deep knowledge of how the data are build and stored. In most of 
the cases the user only needs to provide with a KOI-ID, EPIC-ID, TIC-ID or coordinates of the host star where wants to search for exoplanets. 
SHERLOCK made use of {\sc lightkurve}, {\sc wotan}, {\sc eleonor}, and {\sc transit least squares} packages to download, process, and search for exoplanets 
in any of the thousands of public lightcurves provided by Kepler and TESS missions. 


## Main Developers
Active: <i>[Pozuelos](https://github.com/franpoz), 
[Dévora](https://github.com/martindevora)</i> 

## Additional contributors 
<i>Antoine Thuillier</i>
<i>Lionel García</i>

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
