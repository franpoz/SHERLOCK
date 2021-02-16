import logging
import os
import numpy as np
import astropy.io.fits as astropy_fits
import re
import pandas
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
        self.authors = {}
        self.authors["Kepler"] = "Kepler"
        self.authors["K2"] = "K2"
        self.authors["TESS"] = "SPOC"

    @abstractmethod
    def build(self, object_info, sherlock_dir):
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

    def extract_lc_data(selfself, lcf):
        fit_files = [astropy_fits.open(lcf.filename) for lcf in lcf]
        time = []
        flux = []
        flux_err = []
        background_flux = []
        quality = []
        centroids_x = []
        centroids_y = []
        motion_x = []
        motion_y = []
        [time.append(fit_file[1].data['TIME']) for fit_file in fit_files]
        [flux.append(fit_file[1].data['PDCSAP_FLUX']) for fit_file in fit_files]
        [flux_err.append(fit_file[1].data['PDCSAP_FLUX_ERR']) for fit_file in fit_files]
        [background_flux.append(fit_file[1].data['SAP_BKG']) for fit_file in fit_files]
        try:
            [quality.append(fit_file[1].data['QUALITY']) for fit_file in fit_files]
        except KeyError:
            logging.info("QUALITY info is not available.")
            [quality.append(np.full(len(fit_file[1].data['TIME']), np.nan)) for fit_file in fit_files]
        [centroids_x.append(fit_file[1].data['MOM_CENTR1']) for fit_file in fit_files]
        [centroids_y.append(fit_file[1].data['MOM_CENTR2']) for fit_file in fit_files]
        [motion_x.append(fit_file[1].data['POS_CORR1']) for fit_file in fit_files]
        [motion_y.append(fit_file[1].data['POS_CORR2']) for fit_file in fit_files]
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
