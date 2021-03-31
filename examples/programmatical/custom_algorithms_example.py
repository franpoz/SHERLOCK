from contextlib import contextmanager
from timeit import default_timer

from examples.custom_algorithms.ButterworthCurvePreparer import ButterworthCurvePreparer
from examples.custom_algorithms.NeptunianDesertSearchZone import NeptunianDesertSearchZone
from examples.custom_algorithms.RandomSignalSelector import RandomSignalSelector
from sherlockpipe.sherlock import Sherlock
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo

from sherlockpipe.sherlock_target import SherlockTarget


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: str(default_timer() - start)
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: str(end - start)


with elapsed_timer() as elapsed:
    # We will use one TIC from the TESS mission and add provide external customized algorithms for light curve
    # preparation, signal selection and search zone settings. These algorithms are extensions from SHERLOCK abstract
    # classes as you can inspect under the examples/custom_algorithms directory.
    sherlock = Sherlock(update_ois=False, sherlock_targets=[
        SherlockTarget(object_info=MissionFfiIdObjectInfo("TIC 181804752", 'all'), smooth_enabled=False, high_rms_enabled=False,
                       cpu_cores=2, auto_detrend_enabled=False, prepare_algorithm=ButterworthCurvePreparer(),
                       max_runs=10, custom_search_zone=NeptunianDesertSearchZone(),
                       custom_selection_algorithm=RandomSignalSelector())]) \
        .run()
    print("Analysis took " + elapsed() + "s")
