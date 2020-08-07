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
    # We will use only one mission object and will mask a time range from 1550 to 1551 and from 1560 to 1561
    sherlock = Sherlock([MissionObjectInfo("TIC 181804752", 'all',
                                                    initial_mask=[[1550, 1551], [1560, 1561]])])\
        .run()
    print("Analysis took " + elapsed() + "s")
