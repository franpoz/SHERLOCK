import re
from abc import ABC, abstractmethod


class ObjectInfo(ABC):
    OBJECT_ID_REGEX = "^(KIC|TIC|EPIC)[-_ ]([0-9]+)$"
    NUMBERS_REGEX = "[0-9]+$"
    MISSION_ID_KEPLER = "KIC"
    MISSION_ID_KEPLER_2 = "EPIC"
    MISSION_ID_TESS = "TIC"
    initial_detrend_period = None
    initial_mask = None

    def __init__(self):
        pass

    @abstractmethod
    def sherlock_id(self):
        pass

    @abstractmethod
    def mission_id(self):
        pass
