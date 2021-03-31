## Searching for planetary candidates around  fast rotators 

In your search for exoplanets, it is likely that you will find some lightcurves which show a clear modulation. A typical example is a fast rotator, 
that is, a star which rotation period is less than a few days. An interesting example is TOI-540, which has an orbiting sub-Earth planet. The 
discovery paper can be consulted here: [TOI-540](https://ui.adsabs.harvard.edu/abs/2020arXiv200913623M/abstract)  

<p align="center">
  <img width="80%" src="https://github.com/franpoz/SHERLOCK/blob/master/examples/fast_rotator/TOI-540.png">
From Ment et al. (2020).
</p>
    

The problem in this kind of case is that the rapid rotation will affect the performance of the search algorithm. Therefore, the user needs to perform a pre-analysis 
to first model the rotation, and then proceed with the search for planetary candidates. 

To address this issue and save some user time, SHERLOCK's PIPEline has incorporated an option that automatically detects prominent peaks in a Lomb-Scargle periodogram, which it then models if
necessary.

Let us go through a full example. 

## The user-properties.yaml

We prepare our user parameter file to search for planetary candidates around TIC 200322593 (around which TOI-540b was found) as: 
    
```shell
######################################################################################################################
### GLOBAL OBJECTS RUN SETUP - All sectors analysed at once
######################################################################################################################
##
##
TARGETS:
  'TIC 200322593': 
    SECTORS: [6]
    AUTO_DETREND_ENABLED: True
    INITIAL_HIGH_RMS_MASK: True
    INITIAL_SMOOTH_ENABLED: True
    BEST_SIGNAL_ALGORITHM: 'quorum'
    DETRENDS_NUMBER: 12
    DETREND_CORES: 7
    CPU_CORES: 7
    MAX_RUNS: 2
```

The parameter `AUTO_DETREND_ENABLED: True` allows SHERLOCK to search for the modulation and correct for it.
The star has been observed in three sectors, but for simplicity here we focus only in one of them (Sector 6). 

Then, we only need to launch the .yaml file from our working folder as: 
    
 ```python3.6 -m sherlockpipe --properties user-properties.yaml```

## Results

Here you can see that SHERLOCK has detected a prominent period of ~0.73 days (in the discovery paper this value is given as 0.72+-0.08 days).


<p align="center">
  <img width="80%" src="https://github.com/franpoz/SHERLOCK/blob/master/examples/fast_rotator/Periodogram_TIC200322593_[6].png">
</p>

#

<p align="center">
  <img width="80%" src="https://github.com/franpoz/SHERLOCK/blob/master/examples/fast_rotator/Phase_detrend_period_TIC200322593_[6]_0.73_days.png">
</p>


Then, the pipeline corrected for this modulation and started a search for transits, yielding the following results in `TIC200322593_[6]_candidates.log` 

``` shell 
Listing most promising candidates for ID TIC200322593_[6]:
Detrend no. Period  Duration  T0      Depth   SNR     SDE     FAP       Border_score  Matching OI   Planet radius (R_Earth)  Rp/Rs     Semi-major axis   Habitability Zone   
4           1.2395  30.32     1468.82 1.642   21.26   24.11   0.000080  1.00          TOI 540.01    0.73528                  0.03940   0.01317           I                   
2           5.2769  61.82     1470.12 1.233   11.02   6.14    0.043858  1.00          nan           0.63712                  0.03484   0.03459           I  
```

TOI-540.01 was detected in the first run with SNR=21.26, which is much larger than the SNR reported by [EXOFOP](https://exofop.ipac.caltech.edu/tess/target.php?id=200322593)
i.e. 13.7. Good for SHERLOCK! 


<p align="center">
  <img width="80%" src="https://github.com/franpoz/SHERLOCK/blob/master/examples/fast_rotator/Run_1_ws=0.3372_TIC200322593_[6].png">
</p>
    


Moreover, an extra signal was found, but unfortunately it does not look very promising, but try yourself!



