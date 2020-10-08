import logging
import eleanor
import re
import transitleastsquares as tls
import numpy as np
from astropy.coordinates import SkyCoord
from sherlockpipe.star import starinfo
from sherlockpipe.objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
from sherlockpipe.objectinfo.preparer.LightcurveBuilder import LightcurveBuilder
from sherlockpipe.star.TicStarCatalog import TicStarCatalog
from astropy import units as u
import lightkurve as lk


class MissionFfiLightcurveBuilder(LightcurveBuilder):
    def __init__(self):
        super().__init__()
        self.star_catalog = TicStarCatalog()

    def build(self, object_info):
        mission_id = object_info.mission_id()
        sherlock_id = object_info.sherlock_id()
        quarters = None
        sectors = None
        logging.info("Retrieving star catalog info...")
        mission, mission_prefix, id = super().parse_object_id(mission_id)
        transits_min_count = 1
        star_info = None
        quarters = None
        if mission_prefix not in self.star_catalogs:
            raise ValueError("Wrong object id " + mission_id)
        if mission_prefix == self.MISSION_ID_KEPLER or mission_prefix == self.MISSION_ID_KEPLER_2:
            if object_info.sectors != 'all':
                lcf = lk.search_lightcurvefile(str(mission_id), mission=mission, cadence="long",
                                               quarter=object_info.sectors).download_all()
            else:
                lcf = lk.search_lightcurvefile(str(mission_id), mission=mission, cadence="long").download_all()
            lc = lcf.PDCSAP_FLUX.stitch().remove_nans()
            transits_min_count = 1 if len(lcf) == 0 else 2
            if mission_prefix == self.MISSION_ID_KEPLER:
                quarters = [lcfile.quarter for lcfile in lcf]
            elif mission_prefix == self.MISSION_ID_KEPLER_2:
                logging.info("Correcting K2 motion in light curve...")
                quarters = [lcfile.campaign for lcfile in lcf]
                lc = lc.to_corrector("sff").correct(windows=20)
            star_info = starinfo.StarInfo(sherlock_id, *self.star_catalogs[mission_prefix].catalog_info(id))
        else:
            if isinstance(object_info, MissionFfiCoordsObjectInfo):
                coords = SkyCoord(ra=object_info.ra, dec=object_info.dec, unit=(u.deg, u.deg))
                star = eleanor.multi_sectors(coords=coords, sectors=object_info.sectors)
            else:
                object_id_parsed = re.search(super().NUMBERS_REGEX, object_info.id)
                object_id_parsed = object_info.id[object_id_parsed.regs[0][0]:object_id_parsed.regs[0][1]]
                star = eleanor.multi_sectors(tic=object_id_parsed, sectors=object_info.sectors)
            if star[0].tic:
                # TODO FIX star info objectid
                logging.info("Assotiated TIC is " + star[0].tic)
                star_info = starinfo.StarInfo(object_info.sherlock_id(), *self.star_catalog.catalog_info(int(star[0].tic)))
            data = []
            for s in star:
                datum = eleanor.TargetData(s, height=15, width=15, bkg_size=31, do_pca=True)
                data.append(datum)
            quality_bitmask = np.bitwise_and(data[0].quality.astype(int), 175)
            lc = data[0].to_lightkurve(data[0].pca_flux, quality_mask=quality_bitmask).remove_nans().flatten()
            sectors = [datum.source_info.sector for datum in data]
            if len(data) > 1:
                for datum in data[1:]:
                    quality_bitmask = np.bitwise_and(datum.quality, 175)
                    lc = lc.append(datum.to_lightkurve(datum.pca_flux, quality_mask=quality_bitmask).remove_nans().flatten())
                transits_min_count = 2
        return lc, star_info, transits_min_count, sectors, quarters
