<p align="center">
  <img width="350" src="https://github.com/franpoz/SHERLOCK/blob/master/images/sherlock3.png?raw=true">
</p>

The <b>SHERLOCK</b> (<b>S</b>earching for <b>H</b>ints of <b>E</b>xoplanets f<b>R</b>om <b>L</b>ightcurves 
<b>O</b>f spa<b>C</b>e-based see<b>K</b>ers) <b>PIPE</b>line is a user-friendly pipeline, which
minimizes the interaction of the user to the minimum when using data coming from Kepler or TESS missions. SHERLOCK makes use of previous well-known and well-tested codes which allow the exoplanets community to explore the public data from space-based missions without need of a deep knowledge of how the data are built and stored. 
In most of cases the user only needs to provide with a KOI-ID, EPIC-ID, TIC-ID or coordinates of the host star where wants to search for exoplanets.

## Main Developers
Active: <i>[F.J. Pozuelos](https://github.com/franpoz), 
[M. Dévora](https://github.com/martindevora) </i> 

## Additional contributors 
<i>A. Thuillier</i> & <i>[L. García](https://github.com/LionelGarcia) </i>

## Installation
Some of the SHERLOCK dependencies need `numpy` and `Cython`  before being installed and therefore you should install them before trying the SHERLOCK installation. Take also into account that the dependencies brough by SHERLOCK also need additional operating system tools that need to be installed separately (C++ compiler, Fortran compiler, etc). So, be aware of counting with the next OS packages installed in your environment (e.g. for Python3.8 in a Linux distribution):
```
build-essential libssl-dev python3.8 python3-pip libbz2-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev tk-dev libpng-dev libfreetype6-dev llvm-9 llvm-9-dev gfortran gcc locales python3-tk libpython3.8-dev
```
In case you are running a non-Linux distribution you will need to guess your OS packages matching the ones we mention for Linux.

Therefore, once you have got the OS tools, the SHERLOCK package can be installed from the PyPi repositories (after installing `numpy` and `Cython`):

```
python3 -m pip install numpy
python3 -m pip install Cython
python3 -m pip install sherlockpipe
```

You can also use our Docker image from [DockerHub](https://hub.docker.com/repository/docker/sherlockpipe/sherlockpipe) 
or build it from our [Dockerfile](https://github.com/franpoz/SHERLOCK/blob/master/docker/Dockerfile). Therefore, you
can also use as a Singularity container meanwhile they support Docker.

## Launch
You can run SHERLOCK PIPEline as a standalone package by using:

```python3 -m sherlockpipe --properties my_properties.yaml```

You only need to provide a YAML file with any of the properties contained in the internal 
[properties.yaml](https://github.com/franpoz/SHERLOCK/blob/master/sherlockpipe/properties.yaml)
provided by the pipeline. The most important keys to be defined in your YAML file are those under
the `GLOBAL OBJECTS RUN SETUP` and `SECTOR OBJECTS RUN SETUP` sections because they contain the object ids
or files to be analysed in the execution. You'd need to fill at least one of those keys for the
pipeline to do anything. If you still have any doubts please refer to the 
[examples/properties](https://github.com/franpoz/SHERLOCK/tree/master/examples/properties) directory 

Additionally, you could only want to inspect the preparation stage of SHERLOCK and therefore, you can execute it without
running the analyse phase so you can watch the light curve, the periodogram and the initial report to take better
decisions to tune the execution parameters. Just launch SHERLOCK with:

 ```python3 -m sherlockpipe --properties my_properties.yaml --explore```
 
 and it will end as soon as it has processed the preparation stages for each object.

## Updates
SHERLOCK uses third party data to know TOIs, KOIs, EPICs and to handle FFIs and the vetting process.
This data gets frequently updated from the active missions and therefore SHERLOCK will perform better
if the metadata gets refreshed. You can simply run:

```python3 -m sherlockpipe.update```

and SHERLOCK will download the dependencies. It will store a timestamp to remember the last time it was
refreshed to prevent several unneeded calls. However, if you find that there are more updates and you need
them now, you can call:

```python3 -m sherlockpipe.update --force``` 

and SHERLOCK will ignore the timestamps and perform the update process. In addition, you could be interested
in wiping all the metadata and build it again. That's why you could execute:

```python3 -m sherlockpipe.update --clean```

This last command implies a `force` statement and the last executed time will be ignored too.

You can additionally let SHERLOCK refresh the OIs list before running your current execution by adding to the
YAML file the next line:

```UPDATE_OIS=True``` 

### Vetting
SHERLOCK PIPEline comes with a submodule to examine the most promising transit candidates
found by any of its executions. This is done via [LATTE](https://github.com/noraeisner/LATTE), 
[TPFPlotter](https://github.com/jlillo/tpfplotter) and 
[Triceratops](https://github.com/stevengiacalone/triceratops).
Please note that this feature is only enabled for TESS candidates.
You should be able to execute the vetting by calling:

```python3 -m sherlockpipe.vet --properties my_properties.yaml```

Through that command you will run the vetting process for the given parameters within your provided YAML file. 
You could watch the generated results under `$your_sherlock_object_results_dir/vetting` directory.
Please go to 
[examples/vetting/](https://github.com/franpoz/SHERLOCK/tree/master/examples/vetting.)
to learn how to inject the proper properties for the vetting process.

There is an additional simplified option which can be used to run the vetting. In case you are sure
there is a candidate from the Sherlock results which matches your desired parameters, you can run

```python3 -m sherlockpipe.vet --candidate ${theCandidateNumber}``` 

from the sherlock results directory. This execution will automatically read the transit
parameters from the Sherlock generated files.

### Fitting
SHERLOCK PIPEline comes with another submodule to fit the most promising transit candidates
found by any of its executions. This fit is done via 
[ALLESFITTER](https://github.com/MNGuenther/allesfitter) code. By calling:

```python3 -m sherlockpipe.fit --properties my_properties.yaml```

you will run the fitting process for the given parameters within your provided YAML file. 
You could watch the generated results under `$your_sherlock_object_results_dir/fit` directory.
Please go to 
[examples/fitting/](https://github.com/franpoz/SHERLOCK/tree/master/examples/fitting)
to learn how to inject the proper properties for the fitting process.

There is an additional simplified option which can be used to run the fit. In case you are sure
there is a candidate from the Sherlock results which matches your desired parameters, you can run

```python3 -m sherlockpipe.fit --candidate ${theCandidateNumber}``` 

from the sherlock results directory. This execution will automatically read the transit and star
parameters from the Sherlock generated files.

## SHERLOCK PIPEline Workflow
It is important to note that SHERLOCK PIPEline uses some csv files with TOIs, KOIs and EPIC IDs
from the TESS, Kepler and K2 missions. Therefore your first execution of the pipeline might
take longer because it will download the information.

### Provisioning of light curve
The light curve for every input object needs to be obtained from its mission database. For this we 
use the high level API of [Lightkurve](https://github.com/KeplerGO/lightkurve), which enables the
download of the desired light curves for TESS, Kepler and K2 missions. We also include Full Frame
Images from the TESS mission by the usage of [ELEANOR](https://adina.feinste.in/eleanor/). We 
always use the PDCSAP signal from the ones provided by any of those two packages.

### Pre-processing of light curve
In many cases we will find light curves which contain several systematics like noise, high dispersion
beside the borders, intense periodicities caused by pulsators, fast rotators, etc. SHERLOCK PIPEline
provides some methods to reduce these most important systematics.

#### Local noise reduction
For local noise, where very close measurements show high deviation from the local trend, we apply a
Savitzky-Golay filter. This has proved a highly increment of the SNR of found transits. This feature 
can be disabled with a flag.

#### High RMS areas masking
Sometimes the spacecrafts have to perform reaction wheels momentum dumps by firing thrusters,
sometimes there is high light scattering and sometimes the spacecraft can infer some jitter into
the signal. For all of those systematics we found that in many cases the data from those regions
should be discarded. Thus, SHERLOCK PIPEline includes a binned RMS computation where bins whose
RMS value is higher than a configurable factor multiplied by the median get automatically masked.
This feature can be disabled with a flag. 

#### Input time ranges masking
If enabled, this feature automatically disables 
[High RMS areas masking](https://github.com/franpoz/SHERLOCK#high-rms-areas-masking) 
for the assigned object. The user can input an array of time ranges to be masked into the 
original signal.

#### Detrend of intense periodicities
Our most common foes with high periodicities are fast-rotators, which infer a high sinusoidal-like
trend in the PDCSAP signal. This is why SHERLOCK PIPEline includes an automatic intense periodicities
detection and detrending during its preparation stage. This feature can be disabled with a flag.

#### Input period detrend
If enabled, this feature automatically disables 
[Detrend of intense periodicities](https://github.com/franpoz/SHERLOCK#detrend-of-intense-periodicities) 
for the assigned object. The user can input a period to be used for an initial detrend of the 
original signal. 

#### Custom user code
You can even inject your own python code to perform:
* A custom signal preparation task by implementing the
[CurvePreparer](https://github.com/franpoz/SHERLOCK/tree/master/sherlockpipe/sherlockpipe/curve_preparer/CurvePreparer.py)
class that we provide. Then, inject your python file into the `CUSTOM_PREPARER` property and let SHERLOCK
use your code.
* A custom best signal selection algorithm by implementing the 
[SignalSelector](https://github.com/franpoz/SHERLOCK/tree/master/sherlockpipe/sherlockpipe/scoring/SignalSelector.py). 
class that we provide. Then, inject your python file into the `CUSTOM_ALGORITHM` property and let SHERLOCK use your code.
* A custom search zone definition by implementing the
[SearchZone](https://github.com/franpoz/SHERLOCK/tree/master/sherlockpipe/sherlockpipe/search_zones/SearchZone.py).
class that we provide. Then, inject your python file into the `CUSTOM_SEARCH_ZONE` property and let SHERLOCK use your code. 
* Custom search modes: 'tls', 'bls', 'grazing', 'comet' or 'custom'. You can search for transits by using TLS, BLS,
TLS for a grazing template, TLS for a comet template or even inject your custom transit template (this is currently
included as an experimental feature).

For better understanding of usage please see the
[examples](https://github.com/franpoz/SHERLOCK/tree/master/examples/properties/custom_algorithms.yaml),
which references custom implementations that you can inspect in our 
[custom algorithms directory](https://github.com/franpoz/SHERLOCK/tree/master/examples/custom_algorithms)

### Main execution (run)
After the preparation stage, the SHERLOCK PIPEline will execute what we call `runs` iteratively:
* Several detrended fluxes with increasing window sizes will be extracted from the original 
PDCSAP light curve by using [wotan](https://github.com/hippke/wotan)
* For each detrended flux, the [TransitLeastSquares](https://github.com/hippke/tls) utility will 
be executed to find the most prominent transit.
* The best transit is chosen from all the ones found in the detrended fluxes. Here we have three 
different algorithms for the selection:
    * Basic: Selects the best transit signal only based in the highest SNR value.
    * Border-correct: Selects the best transit signal based in a corrected SNR value. This 
    correction is applied with a border-score factor, which is calculated from the found transits 
    which overlap or are very close to empty-measurements areas in the signal.
    * Quorum: Including the same correction from the border-correct algorithm, quorum will also
    increase the SNR values when several detrended fluxes 'agree' about their transit selection 
    (same ephemerids). The more detrended fluxes agree, the more SNR they get. This algorithm 
    can be slightly tuned by changing the stregth or weight of every detrend vote. It is currently 
    in testing stage and hasn't been used intensively.
    * Custom: You can also inject your own signal selection algorithm by implementing the 
    [SignalSelector](https://github.com/franpoz/SHERLOCK/tree/master/sherlockpipe/scoring/SignalSelector.py)
    class. See the [example](https://github.com/franpoz/SHERLOCK/tree/master/examples/properties/custom_algorithms.yaml).
* Measurements matching the chosen transit are masked in the original PDCSAP signal so they will
not be found by subsequent runs.

### Reporting
SHERLOCK PIPEline produces several information items under a new directory for every analysed
object:
* Object report log: The entire log of the object run is written here.
* Most Promising Candidates log: A summary of the parameters of the best transits found for each
run is written at the end of the object execution. Example content:
```
Listing most promising candidates for ID MIS_TIC 470381900_all:
Detrend no. Period  Duration  T0      SNR     SDE     FAP       Border_score  Matching OI   Semi-major axis   Habitability Zone   
1           2.5013  50.34     1816.69 13.30   14.95   0.000080  1.00          TOI 1696.01   0.02365           I                   
4           0.5245  29.65     1816.56 8.34    6.26    0.036255  1.00          nan           0.00835           I                   
5           0.6193  29.19     1816.43 8.76    6.57    0.019688  1.00          nan           0.00933           I                   
1           0.8111  29.04     1816.10 9.08    5.88    0.068667  0.88          nan           0.01116           I                   
2           1.0093  32.41     1817.05 8.80    5.59    nan       0.90          nan           0.01291           I                   
6           3.4035  45.05     1819.35 6.68    5.97    0.059784  1.00          nan           0.02904           I      
```
* Runs directories: Containing png images of the detrended fluxes and their suggested transits.
Example of one detrended flux transit selection image:
<p align="center">
  <img width="80%" src="https://github.com/franpoz/SHERLOCK/blob/master/images/example_run.png">
</p>

* Light curve csv file: The original (before pre-processing) PDCSAP signal stored in three columns: 
`#time`, `flux` and `flux_err`. Example content:
```
#time,flux,flux_err
1816.0895073542242,0.9916135,0.024114653
1816.0908962630185,1.0232307,0.024185425
1816.0922851713472,1.0293404,0.024151148
1816.0936740796774,1.000998,0.024186047
1816.0950629880074,1.0168158,0.02415397
1816.0964518968017,1.0344968,0.024141008
1816.0978408051305,1.0061758,0.024101004
...
```
* Candidates csv file: Containing the same information than the Most Promising Candidates log but
in a csv format so it can be read by future additions to the pipeline like vetting or fitting
endpoints.
* Lomb-Scargle periodogram plot: Showing the period strengths. Example:
<p align="center">
  <img width="80%" src="https://github.com/franpoz/SHERLOCK/blob/master/images/periodogram.png">
</p>

* RMS masking plot: In case the High RMS masking pre-processing is enabled. Example:
<p align="center">
  <img width="80%" src="https://github.com/franpoz/SHERLOCK/blob/master/images/rms.png">
</p>

* Phase-folded period plot: In case auto-detrend or manual period detrend is enabled.
<p align="center">
  <img width="80%" src="https://github.com/franpoz/SHERLOCK/blob/master/images/autodetrend.png">
</p>

### Dependencies
All the needed dependencies should be included by your `pip` installation of SHERLOCK. If you are
interested you can inspect the requirements list under 
[setup.py](https://github.com/franpoz/SHERLOCK/blob/master/setup.py).

## Testing
SHERLOCK Pipeline comes with a light automated tests suite which can be executed with:
```python3 -m unittest sherlock_tests.py```.
This suite tests several points from the pipeline:
* The construction of the Sherlock object.
* The parameters setup of the Sherlock object.
* The provisioning of objects of interest files.
* Load and filtering of objects of interest.
* Different kind of short Sherlock executions.

In case you want to test the entire SHERLOCK PIPEline functionality we encourage you to
run some (or all) the [manual examples](https://github.com/franpoz/SHERLOCK/tree/master/examples).
If so, please read the instructions provided there to execute them.

## Integration
SHERLOCK integrates with several third party services. Some of them are listed below:
* TESS, Kepler and K2 databases through [Lightkurve](https://github.com/KeplerGO/lightkurve), 
[ELEANOR](https://adina.feinste.in/eleanor/) and [LATTE](https://github.com/noraeisner/LATTE).
* MAST and Vizier catalogs through [Lightkurve](https://github.com/KeplerGO/lightkurve), 
[transitleastsquares](https://github.com/hippke/tls) and 
[Triceratops](https://github.com/stevengiacalone/triceratops)
* [NASA Exoplanet Archive API](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html)
* [TESS ExoFOP](https://exofop.ipac.caltech.edu/tess/view_toi.php).
