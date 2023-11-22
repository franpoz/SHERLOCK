<p align="center">
  <img width="350" src="https://github.com/franpoz/SHERLOCK/blob/master/images/sherlock3.png?raw=true">
</p>

<b>SHERLOCK</b> is an end-to-end pipeline that allows the users to explore the data from space-based missions to search for planetary candidates. It can be used to recover alerted candidates by the automatic pipelines such as SPOC and the QLP, the so-called Kepler objects of interest (KOIs) and TESS objects of interest (TOIs), and to search for candidates that remain unnoticed due to detection thresholds, lack of data exploration or poor photometric quality. To this end, SHERLOCK has six different modules to (1) acquire and prepare the light curves from their repositories, (2) search for planetary candidates, (3) vet the interesting signals, (4) perform a statistical validation, (5) model the signals to refine their ephemerides, and (6) compute the observational windows from ground-based observatories to trigger a follow-up campaign. To execute all these modules, the user only needs to fill in an initial YAML file with some basic information such as the star ID (KIC-ID, EPIC-ID, TIC-ID), the cadence to be used, etc., and use sequentially a few lines of code to pass from one step to the next. Alternatively, the user may provide with the light curve in a csv file, where the time, the normalized flux, and the flux error need to be given in columns comma-separated format. 




## Citation
We are currently working on a specific paper for SHERLOCK. In the meantime, the best way to cite SHERLOCK is by referencing the first paper where it was used [Pozuelos et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A..23P/abstract):

```
@ARTICLE{2020A&A...641A..23P,
       author = {{Pozuelos}, Francisco J. and {Su{\'a}rez}, Juan C. and {de El{\'\i}a}, Gonzalo C. and {Berdi{\~n}as}, Zaira M. and {Bonfanti}, Andrea and {Dugaro}, Agust{\'\i}n and {Gillon}, Micha{\"e}l and {Jehin}, Emmanu{\"e}l and {G{\"u}nther}, Maximilian N. and {Van Grootel}, Val{\'e}rie and {Garcia}, Lionel J. and {Thuillier}, Antoine and {Delrez}, Laetitia and {Rod{\'o}n}, Jose R.},
        title = "{GJ 273: on the formation, dynamical evolution, and habitability of a planetary system hosted by an M dwarf at 3.75 parsec}",
      journal = {\aap},
     keywords = {planets and satellites: dynamical evolution and stability, planets and satellites: formation, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2020,
        month = sep,
       volume = {641},
          eid = {A23},
        pages = {A23},
          doi = {10.1051/0004-6361/202038047},
archivePrefix = {arXiv},
       eprint = {2006.09403},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020A&A...641A..23P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

Also, you may be interested in having a look at recent papers that used SHERLOCK: \
[Pozuelos et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...672A..70P/abstract) \
[Delrez et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220902831D/abstract)\
[Dransfield et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.tmp.1364D/abstract) \
[Luque et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220410261L/abstract) \
[Schanche et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...657A..45S/abstract) \
[Wells et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..97W/abstract) \
[Benni et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.4956B/abstract)  
[Van Grootel et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...650A.205V/abstract) \
[Demory et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...642A..49D/abstract)


## Main Developers
Active: <i>[F.J. Pozuelos](https://github.com/franpoz), 
[M. Dévora](https://github.com/martindevora) </i> 

## Additional contributors 
<i>A. Thuillier</i> & <i>[L. García](https://github.com/LionelGarcia) </i> & <i>[Luis Cerdeño Mota](https://github.com/LuisCerdenoMota)</i>

## Documentation
Please visit [https://sherlock-ph.readthedocs.io](https://sherlock-ph.readthedocs.io) to get a complete set of explanations and tutorials to get started with <b>SHERLOCK</b>.

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
found by any of its executions. This is done via [WATSON](https://github.com/PlanetHunters/watson), capable of vetting
TESS and Kepler targets.
You should be able to execute the vetting by calling:

```python3 -m sherlockpipe.vet --properties my_properties.yaml```

Through that command you will run the vetting process for the given parameters within your provided YAML file. 
You could watch the generated results under `$your_sherlock_object_results_dir/vetting` directory.
Please go to 
[examples/vetting/](https://github.com/franpoz/SHERLOCK/tree/master/examples/vetting)
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

### Validation
SHERLOCK PIPEline implements a module to execute a statistical validation of a candidate by the usage
of 
[TRICERATOPS](https://github.com/stevengiacalone/triceratops). By calling:

```python3 -m sherlockpipe.validate --candidate ${theCandidateNumber}```

you will run the validation for one of the Sherlock candidates.

### Stability
SHERLOCK PIPEline also implements a module to execute a system stability computation by the usage
of 
[Rebound](https://github.com/hannorein/rebound) and [SPOCK](https://github.com/dtamayo/spock). By calling:

```python3 -m sherlockpipe.stability --bodies 1,2,4```

where the `--bodies` parameter is the set of the SHERLOCK accepted signals as CSV to be used in the scenarios 
simulation. You can also provide a 
[stability properties file](https://github.com/franpoz/SHERLOCK/tree/master/examples/properties/stability.yaml))
to run a custom stability simulation:

```python3 -m sherlockpipe.stability --properties stability.yaml```

and you can even combine SHERLOCK accepted signals with some additional bodies provided by the properties file:

```python3 -m sherlockpipe.stability --bodies 1,2,4 --properties stability.yaml```

The results will be stored into a `stability` directory containing the execution log and a `stability.csv`
containing one line per simulated scenario, sorted by the best results score.

### Observation plan
SHERLOCK PIPEline also adds now a tool to plan your observations from ground-based observatories by using 
[astropy](https://github.com/astropy/astropy) and [astroplan](https://github.com/astropy/astroplan). By calling:

```python3 -m sherlockpipe.plan --candidate ${theCandidateNumber} --observatories observatories.csv```

on the resulting `sherlockpipe.fit` directory, where the precise candidate ephemeris are placed. 
The `observatories.csv` file should contain the list of available observatories for your candidate follow-up. 
As an example, you can look at 
[this file](https://github.com/franpoz/SHERLOCK/blob/master/examples/observatories.csv).

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
beside the borders, high-amplitude periodicities caused by pulsators, fast rotators, etc. SHERLOCK PIPEline
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

#### Detrend of high-amplitude periodicities
Our most common foes with high periodicities are fast-rotators, which infer a high sinusoidal-like
trend in the PDCSAP signal. This is why SHERLOCK PIPEline includes an automatic high-amplitude periodicities
detection and detrending during its preparation stage. This feature can be disabled with a flag.

#### Input period detrend
If enabled, this feature automatically disables 
[Detrend of high-amplitude periodicities](https://github.com/franpoz/SHERLOCK#detrend-of-high-amplitude-periodicities) 
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
