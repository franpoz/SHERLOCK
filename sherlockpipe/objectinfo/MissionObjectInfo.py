from sherlockpipe.objectinfo.ObjectInfo import ObjectInfo


class MissionObjectInfo(ObjectInfo):
    def __init__(self, mission_id, sectors, initial_mask=None, initial_detrend_period=None):
        super().__init__(initial_mask, initial_detrend_period)
        self.id = mission_id
        self.sectors = sectors

    def sherlock_id(self):
        return "MIS_" + self.id + "_" + str(self.sectors)

    def mission_id(self):
        return self.id


