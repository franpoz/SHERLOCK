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


def run_moriarty(object_dir, ignore_candidates: list[int], batch_size, threshold):
    object_dir = os.getcwd() if object_dir is None else object_dir
    candidates = pd.read_csv(object_dir + "/candidates.csv")
    object_id = candidates.loc[0, 'Object Id']
    moriarty = MoriartySearch(object_dir, object_id, True, candidates, batch_size=batch_size,
                              threshold=threshold)
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
    if ignore_candidates is not None:
        candidates = candidates.iloc[ignore_candidates]
        candidates.reset_index(drop=True, inplace=True)
    else:
        candidates = None
    transits_mask = []
    if candidates is not None:
        for index, candidate_row in candidates.iterrows():
            fit_results_dir = moriarty.object_dir + f'/fit_[{ignore_candidates[index] + 1}]'
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

