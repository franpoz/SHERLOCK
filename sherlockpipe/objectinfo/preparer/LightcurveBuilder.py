import re
from abc import ABC, abstractmethod
from sherlockpipe.star.EpicStarCatalog import EpicStarCatalog
from sherlockpipe.star.KicStarCatalog import KicStarCatalog
from sherlockpipe.star.TicStarCatalog import TicStarCatalog


class LightcurveBuilder(ABC):
    OBJECT_ID_REGEX = "^(KIC|TIC|EPIC)[-_ ]([0-9]+)$"
    NUMBERS_REGEX = "[0-9]+$"
    MISSION_ID_KEPLER = "KIC"
    MISSION_ID_KEPLER_2 = "EPIC"
    MISSION_ID_TESS = "TIC"

    def __init__(self):
        self.star_catalogs = {}
        self.star_catalogs[self.MISSION_ID_KEPLER] = KicStarCatalog()
        self.star_catalogs[self.MISSION_ID_KEPLER_2] = EpicStarCatalog()
        self.star_catalogs[self.MISSION_ID_TESS] = TicStarCatalog()

    @abstractmethod
    def build(self, object_info):
        pass

    def parse_object_id(self, object_id):
        object_id_parsed = re.search(self.OBJECT_ID_REGEX, object_id)
        mission_prefix = object_id[object_id_parsed.regs[1][0]:object_id_parsed.regs[1][1]]
        id = object_id[object_id_parsed.regs[2][0]:object_id_parsed.regs[2][1]]
        if mission_prefix == self.MISSION_ID_KEPLER:
            mission = "Kepler"
        elif mission_prefix == self.MISSION_ID_KEPLER_2:
            mission = "K2"
        elif mission_prefix == self.MISSION_ID_TESS:
            mission = "TESS"
        else:
            raise ValueError("Invalid object id " + object_id)
        return mission, mission_prefix, int(id)