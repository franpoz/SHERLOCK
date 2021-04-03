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
    # We will use only one mission object and set the period to be initially detrended with a window size of 1/3 times
    # the given period.
    sherlock = Sherlock([SherlockTarget(MissionObjectInfo("TIC 181804752", 'all'), auto_detrend_ratio=1 / 3)])\
        .run()
    print("Analysis took " + elapsed() + "s")
