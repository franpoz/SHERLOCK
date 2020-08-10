import re
from abc import ABC, abstractmethod


class ObjectInfo(ABC):
    """
    Root class to be extended to characterize the input object to be analysed by Sherlock.
    """
    OBJECT_ID_REGEX = "^(KIC|TIC|EPIC)[-_ ]([0-9]+)$"
    NUMBERS_REGEX = "[0-9]+$"
    MISSION_ID_KEPLER = "KIC"
    MISSION_ID_KEPLER_2 = "EPIC"
    MISSION_ID_TESS = "TIC"
    initial_detrend_period = None
    initial_mask = None

    def __init__(self, initial_mask=None, initial_detrend_period=None):
        self.initial_mask = initial_mask
        self.initial_detrend_period = initial_detrend_period

    @abstractmethod
    def sherlock_id(self):
        """
        Returns the unique name generated for Sherlock processing
        """
        pass

    @abstractmethod
    def mission_id(self):
        """
        Returns the real mission identifier
        """
        pass
