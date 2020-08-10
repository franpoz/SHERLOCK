import logging
import lightkurve as lk
from sherlockpipe.star import starinfo
from sherlockpipe.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from sherlockpipe.objectinfo.preparer.LightcurveBuilder import LightcurveBuilder
import pandas as pd


class MissionInputLightcurveBuilder(LightcurveBuilder):
    def __init__(self):
        super().__init__()

    def build(self, object_info):
        mission_id = object_info.mission_id()
        sherlock_id = object_info.sherlock_id()
        quarters = None
        sectors = None
        if isinstance(object_info, MissionInputObjectInfo):
            logging.info("Retrieving star catalog info...")
            mission, mission_prefix, id = super().parse_object_id(mission_id)
            if mission_prefix not in self.star_catalogs:
                raise ValueError("Wrong object id " + mission_id)
            star_info = starinfo.StarInfo(sherlock_id, *self.star_catalogs[mission_prefix].catalog_info(id))
        else:
            star_info = starinfo.StarInfo(sherlock_id)
            star_info.assume_model_mass()
            star_info.assume_model_radius()
        logging.info("Loading lightcurve from file " + object_info.input_file + ".")
        df = pd.read_csv(object_info.input_file, float_precision='round_trip', sep=',',
                         usecols=['#time', 'flux', 'flux_err'])
        lc = lk.LightCurve(time=df['#time'], flux=df['flux'], flux_err=df['flux_err'])
        transits_min_count = 1
        return lc, star_info, transits_min_count, sectors, quarters