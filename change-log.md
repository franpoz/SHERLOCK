### V14 31/07/2020  
This version moves and modularize the code into a python class. Features:
* Significant periods can be auto-detected and detrended before analysis if enabled in setup.
* Input period can be detrended before analysis if provided in setup.
* Phase-folded plots binning is always scaled so it groups 10 min time intervals.
* Border-score is provided for suggested transits so detections close to gaps in observations
can be quickly assessed. This score is only printed but might be used in the future to correct
SNRs.
* TOIs and CTOIs can be downloaded from ExoFop and used to match transits found by SHERLOCK to
already known candidates.
* Addition of TOIs and CTOIs filtering so the final SHERLOCK user can provide a dynamic selection
of candidate conditions to search for new hints.  
* SHERLOCK can now be run for several TICs, but their analysis will be serial. Be careful with 
this because runs can take much time (n * TIC count)
* Number of CPU cores to be used is configurable.
* Logging file is configurable.
* TIC results are stored in a folder matching its name, so the generated plots are better 
organised.
* Mask-mode is configurable so the transit found in every iteration can be entirely masked or 
subtracted from the light curve signal. The subtraction is implemented by reducing the intransit
values to 1, but it doesn't perform very well.
* Kepler and K2 objects are now supported. The K2 light curves are reduced by SFF algorithm.
* Default 0.1Msun and 0.1Rsun values are taken for uncatalogged stars for T14.
* No mass or radius assumption is used for TLS for uncatalogged stars.
* When uncatalogged star parameters in mass or radius, TLS is used without them.
* Auto-detrend method can be configured. For remarkable periodic pulsations or rotations 'cosine' 
method is advised.
* KOIs and EPIC OIs can be downloaded from Exoplanets Archive and used to match transits found by 
SHERLOCK to already known candidates. TOIs, KOIs and EPICs are loaded all together and can be queried 
with the same parameters.
* Savgol filter can be enabled to be applied to remove local noise for short cadence light curves.
* Auto high rms masking can be enabled to mask light curve areas where noise is too high above the
median RMS values.
* Addition of best signal scoring algorithm selection (basic, border-correct and quorum are available).
* Adds Habitability Zone calculation for final most promising candidates report.
* Light curves can now be analysed by providing the list of sectors.

### V13 10/06/2020       
This version chooses the signal with largest SNR to keep searching. The SDE and FAP are printed but 
are not used to choose which is the best signal. This is motivated by a bug discovered by Max. 
It has been reported to TLS developers and hopefully it will be fix soon. 
