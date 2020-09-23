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


class MissionFfiLightcurveBuilder(LightcurveBuilder):
    def __init__(self):
        super().__init__()
        self.star_catalog = TicStarCatalog()

    def build(self, object_info):
        transits_min_count = 1
        star_info = None
        quarters = None
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