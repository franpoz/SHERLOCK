from sherlockpipe.objectinfo.ObjectInfo import ObjectInfo

class MissionFfiCoordsObjectInfo(ObjectInfo):
    def __init__(self, ra, dec, sectors, initial_mask=None, initial_detrend_period=None):
        super().__init__(initial_mask, initial_detrend_period)
        self.ra = ra
        self.dec = dec
        self.sectors = sectors

    def sherlock_id(self):
        return "FFI_" + str(self.ra) + "_" + str(self.dec) + "_" + str(self.sectors)

    def mission_id(self):
        return None


