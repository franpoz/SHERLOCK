import logging
import os

import numpy as np
import pandas

from sherlockpipe.star import starinfo
from sherlockpipe.objectinfo.ObjectProcessingError import ObjectProcessingError
from sherlockpipe.objectinfo.preparer.LightcurveBuilder import LightcurveBuilder
import lightkurve as lk
import matplotlib.pyplot as plt
import csv

class MissionLightcurveBuilder(LightcurveBuilder):
    def __init__(self):
        super().__init__()

    def build(self, object_info, sherlock_dir):
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
        sectors = None if object_info.sectors == 'all' or mission != "TESS" else object_info.sectors
        quarters = None if object_info.sectors == 'all' or mission != "K2" else object_info.sectors
        campaigns = None if object_info.sectors == 'all' or mission != "Kepler" else object_info.sectors
        if object_info.aperture_file is None:
            lcf_search_results = lk.search_lightcurve(str(mission_id), mission=mission, cadence="short",
                                           sector=sectors, quarter=quarters,
                                           campaign=campaigns, author=self.authors[mission])
            lcf = lcf_search_results.download_all()
            lc_data = self.extract_lc_data(lcf)
            if lcf is None:
                raise ObjectProcessingError("Light curve not found for object id " + mission_id)
            lc = None
            matching_objects = []
            for i in range(0, len(lcf.PDCSAP_FLUX)):
                if lcf.PDCSAP_FLUX[i].label == mission_id:
                    if lc is None:
                        lc = lcf.PDCSAP_FLUX[i].normalize()
                    else:
                        lc = lc.append(lcf.PDCSAP_FLUX[i].normalize())
                else:
                    matching_objects.append(lcf.PDCSAP_FLUX[i].label)
            if len(matching_objects) > 0:
                logging.warning("================================================")
                logging.warning("TICS IN THE SAME PIXEL: " + str(matching_objects))
                logging.warning("================================================")
            lc = lc.remove_nans()
            transits_min_count = self.__calculate_transits_min_count(len(lcf))
            if mission_prefix == self.MISSION_ID_KEPLER or mission_id == self.MISSION_ID_KEPLER_2:
                quarters = [lcfile.quarter for lcfile in lcf]
            elif mission_prefix == self.MISSION_ID_TESS:
                sectors = [file.sector for file in lcf]
            if mission_prefix == self.MISSION_ID_KEPLER_2:
                logging.info("Correcting K2 motion in light curve...")
                quarters = [lcfile.campaign for lcfile in lcf]
                lc = lc.to_corrector("sff").correct(windows=20)
            return lc, lc_data, star_info, transits_min_count, np.unique(sectors), np.unique(quarters)
        else:
            logging.info("Using user apertures!")
            tpf_search_results = lk.search_targetpixelfile(str(mission_id), mission=mission, cadence="short",
                                             sector=sectors, quarter=quarters, campaign=campaigns,
                                             author=self.authors[mission])
            tpfs = tpf_search_results.download_all()
            apertures = {}
            if isinstance(object_info.aperture_file, str):
                aperture = []
                with open(object_info.aperture_file, 'r') as fd:
                    reader = csv.reader(fd)
                    for row in reader:
                        aperture.append(row)
                    aperture = np.array(aperture)
                for tpf in tpfs:
                    if mission_prefix == self.MISSION_ID_KEPLER:
                        apertures[tpf.quarter] = aperture
                    elif mission_prefix == self.MISSION_ID_TESS:
                        apertures[tpf.sector] = aperture
                    elif mission_prefix == self.MISSION_ID_KEPLER_2:
                        apertures[tpf.campaign] = aperture
            else:
                for sector, aperture_file in object_info.aperture_file.items():
                    aperture = []
                    with open(aperture_file, 'r') as fd:
                        reader = csv.reader(fd)
                        for row in reader:
                            aperture.append(row)
                    aperture = np.array(aperture)
                    apertures[sector] = aperture
            lc = None
            for tpf in tpfs:
                if mission_prefix == self.MISSION_ID_KEPLER:
                    sector = tpf.quarter
                elif mission_prefix == self.MISSION_ID_TESS:
                    sector = tpf.sector
                elif mission_prefix == self.MISSION_ID_KEPLER_2:
                    sector = tpf.campaign
                aperture = apertures[sector].astype(bool)
                tpf.plot(aperture_mask=aperture, mask_color='red')
                plt.savefig(sherlock_dir + "/Aperture_[" + str(sector) + "].png")
                plt.close()
                if mission_prefix == self.MISSION_ID_KEPLER:
                    corrector = lk.KeplerCBVCorrector(tpf)
                    corrector.plot_cbvs([1, 2, 3, 4, 5, 6, 7])
                    raw_lc = tpf.to_lightcurve(aperture_mask=aperture).remove_nans()
                    plt.savefig(sherlock_dir + "/Corrector_components[" + str(sector) + "].png")
                    plt.close()
                    it_lc = corrector.correct([1, 2, 3, 4, 5])
                    ax = raw_lc.plot(color='C3', label='SAP Flux', linestyle='-')
                    it_lc.plot(ax=ax, color='C2', label='CBV Corrected SAP Flux', linestyle='-')
                    plt.savefig(sherlock_dir + "/Raw_vs_CBVcorrected_lc[" + str(sector) + "].png")
                    plt.close()
                elif mission_prefix == self.MISSION_ID_KEPLER_2:
                    raw_lc = tpf.to_lightcurve(aperture_mask=aperture).remove_nans()
                    it_lc = raw_lc.to_corrector("sff").correct(windows=20)
                    ax = raw_lc.plot(color='C3', label='SAP Flux', linestyle='-')
                    it_lc.plot(ax=ax, color='C2', label='CBV Corrected SAP Flux', linestyle='-')
                    plt.savefig(sherlock_dir + "/Raw_vs_SFFcorrected_lc[" + str(sector) + "].png")
                    plt.close()
                elif mission_prefix == self.MISSION_ID_TESS:
                    temp_lc = tpf.to_lightcurve(aperture_mask=aperture)
                    where_are_NaNs = np.isnan(temp_lc.flux)
                    temp_lc = temp_lc[np.where(~where_are_NaNs)]
                    regressors = tpf.flux[np.argwhere(~where_are_NaNs), ~aperture]
                    temp_token_lc = [temp_lc[i: i + 2000] for i in range(0, len(temp_lc), 2000)]
                    regressors_token = [regressors[i: i + 2000] for i in range(0, len(regressors), 2000)]
                    it_lc = None
                    raw_it_lc = None
                    item_index = 0
                    for temp_token_lc_item in temp_token_lc:
                        regressors_token_item = regressors_token[item_index]
                        design_matrix = lk.DesignMatrix(regressors_token_item, name='regressors').pca(5).append_constant()
                        corr_lc = lk.RegressionCorrector(temp_token_lc_item).correct(design_matrix)
                        if it_lc is None:
                            it_lc = corr_lc
                            raw_it_lc = temp_token_lc_item
                        else:
                            it_lc = it_lc.append(corr_lc)
                            raw_it_lc = raw_it_lc.append(temp_token_lc_item)
                        item_index = item_index + 1
                    ax = raw_it_lc.plot(label='Raw light curve')
                    it_lc.plot(ax=ax, label='Corrected light curve')
                    plt.savefig(sherlock_dir + "/Raw_vs_DMcorrected_lc[" + str(sector) + "].png")
                    plt.close()
                if lc is None:
                    lc = it_lc.normalize()
                else:
                    lc = lc.append(it_lc.normalize())
            lc = lc.remove_nans()
            lc.plot(label="Normalized light curve")
            plt.savefig(sherlock_dir + "/Normalized_lc[" + str(sector) + "].png")
            plt.close()
            transits_min_count = self.__calculate_transits_min_count(len(tpfs))
            if mission_prefix == self.MISSION_ID_KEPLER or mission_id == self.MISSION_ID_KEPLER_2:
                quarters = [lcfile.quarter for lcfile in tpfs]
            elif mission_prefix == self.MISSION_ID_TESS:
                sectors = [file.sector for file in tpfs]
            if mission_prefix == self.MISSION_ID_KEPLER_2:
                logging.info("Correcting K2 motion in light curve...")
                quarters = [lcfile.campaign for lcfile in tpfs]
            lc_data = None
            return lc, lc_data, star_info, transits_min_count, np.unique(sectors), np.unique(quarters)

    def __calculate_transits_min_count(self, len_data):
        return 1 if len_data == 1 else 2

