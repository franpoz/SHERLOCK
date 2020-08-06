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
    # Adding several kinds of objects to the run: one short cadence TIC, one FFI TIC, one coordinates FFI, one input
    # file related to a TIC and one plain input file.
    # Ensure that your input light curve CSV files have three columns: #TBJD,flux,flux_err
    sherlock = Sherlock([MissionObjectInfo("TIC 181804752", 'all'),
                                        MissionFfiIdObjectInfo("TIC 259168516", [14, 15]),
                                        MissionFfiCoordsObjectInfo(14, 19, 'all'),
                                        MissionInputObjectInfo("TIC 470381900", "example_lightcurve.csv"),
                                        InputObjectInfo("example_lc.csv")])\
        .run()
    print("Analysis took " + elapsed() + "s")
