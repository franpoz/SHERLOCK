from sherlockpipe.objectinfo.ObjectInfo import ObjectInfo

class MissionFfiIdObjectInfo(ObjectInfo):
    """
    Implementation of ObjectInfo to be used to characterize long-cadence objects.
    """
    def __init__(self, mission_id, sectors, initial_mask=None, initial_detrend_period=None):
        """
        @param mission_id: the mission identifier. TIC ##### for TESS, KIC ##### for Kepler and EPIC ##### for K2.
        @param sectors: an array of integers specifying which sectors will be analysed for the object
        @param initial_mask: an array of time ranges provided to mask them into the initial object light curve.
        @param initial_detrend_period: integer value specifying a fixed value for an initial period to be detrended
        from the initial light curve before processing.
        """
        super().__init__(initial_mask, initial_detrend_period)
        self.id = mission_id
        self.sectors = sectors

    def sherlock_id(self):
        return self.id.replace(" ", "") + "_FFI_" + str(self.sectors)

    def mission_id(self):
        return self.id


