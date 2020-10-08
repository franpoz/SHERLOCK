import logging
from sherlockpipe.star import starinfo
from sherlockpipe.objectinfo.ObjectProcessingError import ObjectProcessingError
from sherlockpipe.objectinfo.preparer.LightcurveBuilder import LightcurveBuilder
import lightkurve as lk


class MissionLightcurveBuilder(LightcurveBuilder):
    def __init__(self):
        super().__init__()

    def build(self, object_info):
        mission_id = object_info.mission_id()
        sherlock_id = object_info.sherlock_id()
        quarters = None
        sectors = None
        logging.info("Retrieving star catalog info...")
        mission, mission_prefix, id = super().parse_object_id(mission_id)
        if mission_prefix not in self.star_catalogs:
            raise ValueError("Wrong object id " + mission_id)
        star_info = starinfo.StarInfo(sherlock_id, *self.star_catalogs[mission_prefix].catalog_info(id))
        logging.info("Downloading lightcurve files...")
        if mission == "TESS" and object_info.sectors != 'all':
            lcf = lk.search_lightcurvefile(str(mission_id), mission=mission, cadence="short", sector=object_info.sectors)\
                .download_all()
        elif mission == "TESS":
            lcf = lk.search_lightcurvefile(str(mission_id), mission=mission, cadence="short").download_all()
        elif object_info.sectors != 'all':
            lcf = lk.search_lightcurvefile(str(mission_id), mission=mission, cadence="short", quarter=object_info.sectors)\
                .download_all()
        else:
            lcf = lk.search_lightcurvefile(str(mission_id), mission=mission, cadence="short").download_all()
        if lcf is None:
            raise ObjectProcessingError("Light curve not found for object id " + mission_id)
        lc = lcf.PDCSAP_FLUX.stitch().remove_nans()
        transits_min_count = 1 if len(lcf) == 0 else 2
        if mission_prefix == self.MISSION_ID_KEPLER or mission_id == self.MISSION_ID_KEPLER_2:
            quarters = [lcfile.quarter for lcfile in lcf]
        elif mission_prefix == self.MISSION_ID_TESS:
            sectors = [file.sector for file in lcf]
        if mission_prefix == self.MISSION_ID_KEPLER_2:
            logging.info("Correcting K2 motion in light curve...")
            quarters = [lcfile.campaign for lcfile in lcf]
            lc = lc.to_corrector("sff").correct(windows=20)
        return lc, star_info, transits_min_count, sectors, quarters