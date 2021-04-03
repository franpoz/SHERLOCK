from contextlib import contextmanager
from timeit import default_timer
from sherlockpipe.sherlock import Sherlock
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo

from sherlockpipe.sherlock_target import SherlockTarget


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
    # 1 Set the maximum number of runs to be executed.
    # 2 Select the period protect value, which limits the minimum detrend window length
    # 3 Select the min period for a transit to be fit.
    # 4 Select the max period for a transit to be fit.
    # 5 Select the binning for RMS calculation
    # 6 Select the number of CPU cores to be used for the transit search.
    # 7 Select the min SNR, the min SDE and the max FAP to stop the runs execution for each object.
    # 8 Select the found transits masking method. We use subtract here as example, but it is discouraged.
    # 9 Select the best signal algorithm, which provides a different implementation to decide which of the detrend
    # signals is the stronger one to be selected.
    # 10 Set the strength of the quorum algorithm votes, which makes every vote that is found to increase the SNR by
    # a factor of 1.2 for our selection.
    sherlock = Sherlock([SherlockTarget(MissionObjectInfo("TIC 181804752", 'all'),
                                        max_runs=10, period_protect=12, period_min=1, period_max=10, bin_minutes=20,
                                 cpu_cores=3, snr_min=6, sde_min=6, mask_mode="subtract",
                                 best_signal_algorithm='quorum', quorum_strength=1.2)])\
        .run()
    print("Analysis took " + elapsed() + "s")
