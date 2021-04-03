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
    # We will use only one mission object and will enable the automatic high RMS masking. We will set a RMS threshold
    # of 2 times the median with a binning of 3 hours.
    sherlock = Sherlock([SherlockTarget(MissionObjectInfo("TIC 181804752", 'all'), high_rms_enabled=True,
                                        high_rms_threshold=2.5, high_rms_bin_hours=3)])\
        .run()
    print("Analysis took " + elapsed() + "s")
