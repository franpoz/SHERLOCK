from sherlockpipe.objectinfo.ObjectInfo import ObjectInfo


class MissionObjectInfo(ObjectInfo):
    """
    Implementation of ObjectInfo to be used to characterize short-cadence objects from TESS, Kepler and K2 missions.
    """
    def __init__(self, mission_id: str, sectors, initial_mask=None, initial_detrend_period=None, star_info=None,
                 aperture_file=None):
        """
        @param mission_id: the mission identifier. TIC ##### for TESS, KIC ##### for Kepler and EPIC ##### for K2.
        @param sectors: an array of integers specifying which sectors will be analysed for the object
        @param initial_mask: an array of time ranges provided to mask them into the initial object light curve.
        @param initial_detrend_period: integer value specifying a fixed value for an initial period to be detrended
        @param star_info: input star information
        @param aperture_file: the file containing 1s and 0s specifying the user selected aperture
        from the initial light curve before processing.
        """
        super().__init__(initial_mask, initial_detrend_period, star_info, aperture_file)
        self.id = mission_id
        self.sectors = sectors

    def sherlock_id(self):
        return self.id.replace(" ", "") + "_" + str(self.sectors)

    def mission_id(self):
        return self.id
