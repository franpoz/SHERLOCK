import os
from sherlockpipe.objectinfo.ObjectInfo import ObjectInfo

class InputObjectInfo(ObjectInfo):
    """
    Implementation of ObjectInfo to be used to characterize objects which are to be loaded from a csv file.
    """
    def __init__(self, input_file, initial_mask=None, initial_detrend_period=None, star_info=None,
                 aperture_file=None):
        """
        @param input_file: the file to be used for loading the light curve
        @param initial_mask: an array of time ranges provided to mask them into the initial object light curve.
        @param initial_detrend_period: integer value specifying a fixed value for an initial period to be detrended
        from the initial light curve before processing.
        @param star_info: input star information
        @param aperture_file: the file containing 1s and 0s specifying the user selected aperture
        """
        super().__init__(initial_mask, initial_detrend_period, star_info, aperture_file)
        self.input_file = input_file

    def sherlock_id(self):
        return "INP_" + os.path.splitext(self.input_file)[0].replace("/", "_")

    def mission_id(self):
        return None
