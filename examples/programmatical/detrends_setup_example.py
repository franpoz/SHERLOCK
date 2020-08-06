from contextlib import contextmanager
from timeit import default_timer
from sherlockpipe.sherlock import Sherlock
from sherlockpipe.objectinfo.MissionObjectInfo import MissionObjectInfo


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: str(default_timer() - start)
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: str(end - start)


with elapsed_timer() as elapsed:
    # We will use only one object id so we can explain better the detrend configs that the coder can select
    # We will:
    # 1 Enable the initial smooth function, which reduces local noise in the signal.
    # 2 Enable the initial High RMS areas masking. This procedure will mask all the lightcurve time binned ranges by
    # the 'initial_rms_bin_hours' value with a threshold of the 'initial_rms_threshold' value * RMS_median.
    # 3 Set the number of detrends to be done from PDCSAP_FLUX for each run.
    # 4 Set the SHERLOCK PDCSAP_FLUX detrends method to Gaussian Processes.
    # 5 Set the number of CPU cores to be used by the detrending procedure.
    # 6 Enable the Auto-Detrend detection, which will search for strong periodicities in the light curve and do an
    # initial detrend for it based on the selected 'auto_detrend_method' method and the value of the
    # 'auto_detrend_ratio' value, which ensures that we are detrending the light curve at 'auto_detrend_ratio' times
    # the stronger period.
    sherlock = Sherlock([MissionObjectInfo("TIC 181804752", 'all')])\
        .setup_detrend(initial_smooth=True, initial_rms_mask=True, initial_rms_threshold=2.5, initial_rms_bin_hours=3,
                       n_detrends=12, detrend_method="gp", cores=2, auto_detrend_periodic_signals=True,
                       auto_detrend_ratio=1/3, auto_detrend_method="cosine")\
        .run()
    print("Analysis took " + elapsed() + "s")
