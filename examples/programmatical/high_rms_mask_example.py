from contextlib import contextmanager
from timeit import default_timer
from sherlockpipe.sherlock import Sherlock
from sherlockpipe.objectinfo.InputObjectInfo import InputObjectInfo
from sherlockpipe.objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
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
    # We will use only one mission object and will enable the automatic high RMS masking. We will set a RMS threshold
    # of 2 times the median with a binning of 3 hours.
    sherlock = Sherlock([MissionObjectInfo("TIC 181804752", 'all')])\
        .setup_detrend(initial_rms_mask=True, initial_rms_threshold=2.5, initial_rms_bin_hours=3)\
        .run()
    print("Analysis took " + elapsed() + "s")
