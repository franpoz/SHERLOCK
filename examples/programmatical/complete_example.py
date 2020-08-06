from contextlib import contextmanager
from timeit import default_timer
from sherlockpipe.sherlock import Sherlock
from sherlockpipe.objectinfo.InputObjectInfo import InputObjectInfo
from sherlockpipe.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from sherlockpipe.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from sherlockpipe.objectinfo.MissionObjectInfo import MissionObjectInfo


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: str(default_timer() - start)
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: str(end - start)


with elapsed_timer() as elapsed:
    # We will use one FFI object from TESS, one short cadence object from TESS restricted to one sector, one short
    # cadence object from Kepler and one short cadence object from K2.
    # We will also provide two objects whose light curve info source are input files. For the first one, we will mask
    # two time intervals and for the second one we will add an initial detrend period.
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

    # 7 Set the maximum number of runs to be executed.
    # 8 Select the period protect value, which TODO
    # 9 Select the min period for a transit to be fit.
    # 10 Select the max period for a transit to be fit.
    # 11 Select the binning for TODO
    # 12 Select the number of CPU cores to be used for the transit search.
    # 13 Select the min SNR, the min SDE and the max FAP to stop the runs execution for each object.
    # 14 Select the found transits masking method. We use subtract here as example, but it is discouraged.
    # 15 Select the best signal algorithm, which provides a different implementation to decide which of the detrend
    # signals is the stronger one to be selected.
    # 10 Set the strength of the quorum algorithm votes, which makes every vote that is found to increase the SNR by
    # a factor of 1.2 for our selection.
    sherlock = Sherlock([MissionFfiIdObjectInfo("TIC 181804752", 'all'),
                                        MissionObjectInfo("TIC 259168516", [15]),
                                        MissionObjectInfo('KIC 10905746', 'all'),
                                        MissionObjectInfo('EPIC 249631677', 'all'),
                                        MissionInputObjectInfo("TIC 181804752", 'example_lc.csv',
                                                               initial_mask=[[1625, 1626], [1645, 1646]]),
                                        InputObjectInfo("example_lc.csv", initial_detrend_period=0.8)]) \
        .setup_detrend(initial_smooth=True, initial_rms_mask=True, initial_rms_threshold=2.5, initial_rms_bin_hours=3,
                       n_detrends=12, detrend_method="gp", cores=2, auto_detrend_periodic_signals=True,
                       auto_detrend_ratio=1 / 3, auto_detrend_method="cosine") \
        .setup_transit_adjust_params(max_runs=10, period_protec=12, period_min=1, period_max=10, bin_minutes=20,
                                 run_cores=3, snr_min=6, sde_min=6, fap_max=0.08, mask_mode="subtract",
                                 best_signal_algorithm='quorum', quorum_strength=1.2)\
        .run()
    print("Analysis took " + elapsed() + "s")
