from objectinfo.ObjectInfo import ObjectInfo

class MissionInputObjectInfo(ObjectInfo):
    def __init__(self, mission_id, input_file, initial_mask=None, initial_detrend_period=None):
        super().__init__(initial_mask, initial_detrend_period)
        self.id = mission_id
        self.input_file = input_file

    def sherlock_id(self):
        return "INP_" + self.id

    def mission_id(self):
        return self.id
