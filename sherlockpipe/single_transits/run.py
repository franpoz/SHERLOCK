import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from sherlockpipe.loading import common
from sherlockpipe.single_transits.search import MoriartySearch
from sherlockpipe.vetting.vetter import Vetter
from sherlockpipe.bayesian_fit.allesfitter_data_extractor import AllesfitterDataExtractor
import alexfitter
from lcbuilder.helper import LcbuilderHelper
import astropy.units as u


def run_moriarty(object_dir, ignore_candidates: list[int], batch_size, threshold,
                 disable_snr_projected=False, lc_file: str = 'lc_predetrends.csv'):
    """
    Run the Moriarty single-transit search on a set of candidates.

    Loads candidate data from disk, configures known transit masks from
    previous SHERLOCK results, and launches the MoriartySearch pipeline.

    Parameters
    ----------
    object_dir : str or None
        Directory containing the object's results. Defaults to current directory.
    ignore_candidates : list of int
        Candidates to mask as known transits.
    batch_size : int
        Batch size for SANTO model predictions.
    threshold : float
        Probability threshold for single-transit classification.
    disable_snr_projected : bool, optional
        If True, skip the projected SNR check. Default False.
    lc_file : str, optional
        light curve file to use. Default lc_predetrends.csv
    """
    object_dir = os.getcwd() if object_dir is None else object_dir
    candidates = pd.read_csv(object_dir + "/candidates.csv")
    if len(candidates) > 0:
        object_id = candidates.loc[0, 'Object Id']
    else:
        dir_name = os.path.basename(object_dir.rstrip('/'))
        object_id = dir_name.split('_')[0].replace('TIC', 'TIC ').replace('KIC', 'KIC ').replace('EPIC', 'EPIC ')
        logging.info("Empty candidates.csv, using directory name for object ID: " + object_id)
    moriarty = MoriartySearch(object_dir, object_id, True, candidates, batch_size=batch_size,
                              threshold=threshold, disable_snr_projected=disable_snr_projected, lc_file=lc_file)
    file_dir = moriarty.object_dir + "/moriarty.log"
    if os.path.exists(file_dir):
        os.remove(file_dir)
    if not isinstance(logging.root, logging.RootLogger):
        logging.root = logging.RootLogger(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    while len(logger.handlers) > 0:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.FileHandler(file_dir)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.info("Starting single transits search")
    if ignore_candidates is not None and len(ignore_candidates) > 0:
        candidates = candidates.iloc[ignore_candidates]
        candidates.reset_index(drop=True, inplace=True)
    else:
        candidates = None
    transits_mask = []
    if candidates is not None:
        for index, candidate_row in candidates.iterrows():
            fit_results_dir = moriarty.object_dir + f'/fit_[{ignore_candidates[index] + 1}]/main'
            if os.path.exists(fit_results_dir):
                logging.info("Reading fit results from " + fit_results_dir)
                ns_derived_file = fit_results_dir + "/results/ns_derived_table.csv"
                ns_file = fit_results_dir + "/results/ns_table.csv"
                fit_derived_results = pd.read_csv(ns_derived_file)
                fit_results = pd.read_csv(ns_file)
                alles = alexfitter.allesclass(fit_results_dir)
                candidate_no = 0
                candidates.loc[index, 'period'], _, _ = AllesfitterDataExtractor.extract_period(candidate_no, fit_results, alles)
                candidates.loc[index, 't0'], _, _ = AllesfitterDataExtractor.extract_epoch(candidate_no, fit_results, alles)
                candidates.loc[index, 'duration'], _, _ = AllesfitterDataExtractor.extract_duration(candidate_no, fit_derived_results)
                candidates.loc[index, 'duration'] = candidates.loc[index, 'duration'] * 60
                candidates.loc[index, 'depth'], _, _ = AllesfitterDataExtractor.extract_depth(candidate_no, fit_derived_results)
                candidates.loc[index, 'depth'] = candidates.loc[index, 'depth'] / 1000
                candidates.loc[index, 'a'], _, _ = AllesfitterDataExtractor.extract_semimajor_axis(candidate_no, fit_derived_results)
                rp, _, _ = AllesfitterDataExtractor.extract_radius(candidate_no, fit_derived_results)
                #candidates.loc[:, 'rp_rs'] = rp / LcbuilderHelper.convert_from_to(star_df["radius"], u.Rsun, u.Rearth)
                logging.info("Selected signal number " + str(index))
            transits_mask.append({"P": candidate_row["period"], "T0": candidate_row["t0"],
                                  "D": candidate_row["duration"] * 2})
    moriarty.transits_mask = transits_mask
    moriarty.search_candidates_df = candidates
    moriarty.run(1)

