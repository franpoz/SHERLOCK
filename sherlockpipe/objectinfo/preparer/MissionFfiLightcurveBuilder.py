import logging
import os
import sys
import sherlockpipe.eleanor
from sherlockpipe import constants

sys.modules['eleanor'] = sys.modules['sherlockpipe.eleanor']
import eleanor
from sherlockpipe.eleanor.targetdata import TargetData
import re
import numpy as np
import pandas
import astropy.io.fits as astropy_fits
from astropy.coordinates import SkyCoord
from sherlockpipe.star import starinfo
from sherlockpipe.objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
from sherlockpipe.objectinfo.preparer.LightcurveBuilder import LightcurveBuilder
from sherlockpipe.star.TicStarCatalog import TicStarCatalog
from astropy import units as u
import lightkurve as lk
from lightkurve.correctors import SFFCorrector

class MissionFfiLightcurveBuilder(LightcurveBuilder):
    def __init__(self):
        super().__init__()
        self.star_catalog = TicStarCatalog()

    def build(self, object_info, sherlock_dir):
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
        sectors = None if object_info.sectors == 'all' or mission != "TESS" else object_info.sectors
        quarters = None if object_info.sectors == 'all' or mission != "K2" else object_info.sectors
        campaigns = None if object_info.sectors == 'all' or mission != "Kepler" else object_info.sectors
        if mission_prefix == self.MISSION_ID_KEPLER or mission_prefix == self.MISSION_ID_KEPLER_2:
            lcf_search_results = lk.search_lightcurvefile(str(mission_id), mission=mission, cadence="long",
                                           author=self.authors[mission], sector=sectors, quarter=quarters,
                                           campaign=campaigns)
            lcf = lcf_search_results.download_all()
            lc_data = self.extract_lc_data(lcf)
            lc = lcf.PDCSAP_FLUX.stitch().remove_nans()
            transits_min_count = 1 if len(lcf) == 0 else 2
            if mission_prefix == self.MISSION_ID_KEPLER:
                quarters = [lcfile.quarter for lcfile in lcf]
            elif mission_prefix == self.MISSION_ID_KEPLER_2:
                logging.info("Correcting K2 motion in light curve...")
                quarters = [lcfile.campaign for lcfile in lcf]
                lc = SFFCorrector(lc).correct(windows=20)
            star_info = starinfo.StarInfo(sherlock_id, *self.star_catalogs[mission_prefix].catalog_info(id))
        else:
            if isinstance(object_info, MissionFfiCoordsObjectInfo):
                coords = SkyCoord(ra=object_info.ra, dec=object_info.dec, unit=(u.deg, u.deg))
                star = eleanor.source.multi_sectors(coords=coords, sectors=object_info.sectors,
                                                    post_dir=constants.USER_HOME_ELEANOR_CACHE)
            else:
                object_id_parsed = re.search(super().NUMBERS_REGEX, object_info.id)
                object_id_parsed = object_info.id[object_id_parsed.regs[0][0]:object_id_parsed.regs[0][1]]
                star = eleanor.multi_sectors(tic=object_id_parsed, sectors=object_info.sectors,
                                             post_dir=constants.USER_HOME_ELEANOR_CACHE)
            if star is None:
                raise ValueError("No data for this object")
            if star[0].tic:
                # TODO FIX star info objectid
                logging.info("Assotiated TIC is " + star[0].tic)
                star_info = starinfo.StarInfo(object_info.sherlock_id(), *self.star_catalog.catalog_info(int(star[0].tic)))
            data = []
            for s in star:
                datum = TargetData(s, height=15, width=15, bkg_size=31, do_pca=True)
                data.append(datum)
            quality_bitmask = np.bitwise_and(data[0].quality.astype(int), 175)
            lc_data = self.extract_eleanor_lc_data(data)
            lc = data[0].to_lightkurve(data[0].pca_flux, quality_mask=quality_bitmask).remove_nans().flatten()
            sectors = [datum.source_info.sector for datum in data]
            if len(data) > 1:
                for datum in data[1:]:
                    quality_bitmask = np.bitwise_and(datum.quality, 175)
                    lc = lc.append(datum.to_lightkurve(datum.pca_flux, quality_mask=quality_bitmask).remove_nans().flatten())
                transits_min_count = 2
        return lc, lc_data, star_info, transits_min_count, sectors, quarters

    def extract_eleanor_lc_data(selfself, eleanor_data):
        time = []
        flux = []
        flux_err = []
        background_flux = []
        quality = []
        centroids_x = []
        centroids_y = []
        motion_x = []
        motion_y = []
        [time.append(data.time) for data in eleanor_data]
        [flux.append(data.pca_flux) for data in eleanor_data]
        [flux_err.append(data.flux_err) for data in eleanor_data]
        [background_flux.append(data.flux_bkg) for data in eleanor_data]
        try:
            [quality.append(data.quality) for data in eleanor_data]
        except KeyError:
            logging.info("QUALITY info is not available.")
            [quality.append(np.full(len(data.time), np.nan)) for data in eleanor_data]
        [centroids_x.append(data.centroid_xs - data.cen_x) for data in eleanor_data]
        [centroids_y.append(data.centroid_ys - data.cen_y) for data in eleanor_data]
        [motion_x.append(data.x_com) for data in eleanor_data]
        [motion_y.append(data.y_com) for data in eleanor_data]
        time = np.concatenate(time)
        flux = np.concatenate(flux)
        flux_err = np.concatenate(flux_err)
        background_flux = np.concatenate(background_flux)
        quality = np.concatenate(quality)
        centroids_x = np.concatenate(centroids_x)
        centroids_y = np.concatenate(centroids_y)
        motion_x = np.concatenate(motion_x)
        motion_y = np.concatenate(motion_y)
        lc_data = pandas.DataFrame(columns=['time', 'flux', 'flux_err', 'background_flux', 'quality', 'centroids_x',
                                            'centroids_y', 'motion_x', 'motion_y'])
        lc_data['time'] = time
        lc_data['flux'] = flux
        lc_data['flux_err'] = flux_err
        lc_data['background_flux'] = background_flux
        lc_data['quality'] = quality
        lc_data['centroids_x'] = centroids_x
        lc_data['centroids_y'] = centroids_y
        lc_data['motion_x'] = motion_x
        lc_data['motion_y'] = motion_y
        return lc_data
