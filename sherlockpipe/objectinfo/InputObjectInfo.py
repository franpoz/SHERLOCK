import os
from sherlockpipe.objectinfo.ObjectInfo import ObjectInfo

class InputObjectInfo(ObjectInfo):
    def __init__(self, input_file, initial_mask=None, initial_detrend_period=None):
        super().__init__(initial_mask, initial_detrend_period)
        self.input_file = input_file

    def sherlock_id(self):
        return "INP_" + os.path.splitext(self.input_file)[0].replace("/", "_")

    def mission_id(self):
        return None
