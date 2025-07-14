import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from sherlockpipe.loading import common
from sherlockpipe.vetting.vetter import Vetter
from sherlockpipe.bayesian_fit.allesfitter_data_extractor import AllesfitterDataExtractor
import alexfitter
import lcbuilder.helper.LcbuilderHelper
import astropy.units as u


def run_vet(object_dir, candidate, properties, cpus=os.cpu_count() - 1, run_iatson=False, run_gpt=False, gpt_key=None,
            only_summary=False, triceratops_bins=100, triceratops_scenarios=5,
            triceratops_curve_file=None, triceratops_contrast_curve_file=None,
            triceratops_additional_stars_file=None,
            triceratops_sigma_mode='flux_err',
            triceratops_ignore_ebs=False,
            triceratops_resolved_companion=None,
            triceratops_ignore_background_stars=False,
            sectors=None):
    object_dir = os.getcwd() if object_dir is None else object_dir
    candidates = pd.read_csv(object_dir + "/candidates.csv")
    if candidate is not None:
        vetting_dir = object_dir + "/vet_" + str(candidate)
    else:
        vetting_dir = object_dir + "/vet_" + str(Path(properties).stem)
    if os.path.exists(vetting_dir) or os.path.isdir(vetting_dir):
        shutil.rmtree(vetting_dir, ignore_errors=True)
    os.mkdir(vetting_dir)
    vetter = Vetter(object_dir, vetting_dir, candidate is not None, candidates)
    file_dir = vetter.watson.object_dir + "/vetting.log"
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
    logging.info("Starting vetting")
    star_df = pd.read_csv(vetter.object_dir() + "/params_star.csv")
    star_df = star_df.iloc[0]
    transits_df = None
    if candidate is None:
        user_properties = common.load_from_yaml(properties)
        candidate = pd.DataFrame(columns=['id', 'period', 'depth', 'depth_err' ,'t0', 'sectors', 'number', 'curve', 'rp_rs', 'a'])
        candidate = candidate.append(user_properties, ignore_index=True)
        candidate['id'] = star_df["obj_id"]
    else:
        candidate_selection = int(candidate)
        if candidate_selection < 1 or candidate_selection > len(candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        candidates = candidates.rename(columns={'Object Id': 'id'})
        candidates['number'] = 1
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate.loc[:, 'number'] = candidate_selection
        transits_df_file = vetter.object_dir() + "/transits_stats.csv"
        fit_results_dir = vetter.object_dir() + f"/fit_[{candidate_number}]"
        if os.path.exists(transits_df_file):
            transits_df = pd.read_csv(vetter.object_dir() + "/transits_stats.csv")
            transits_df = transits_df[transits_df["candidate"] == candidate_selection - 1]
            if len(transits_df) == 0:
                logging.info("Not NAN transits found for candidate in transits_stats.csv file")
                transits_df = None
        if os.path.exists(fit_results_dir):
            logging.info("Reading fit results from " + fit_results_dir)
            ns_derived_file = object_dir + "/results/ns_derived_table.csv"
            ns_file = object_dir + "/results/ns_table.csv"
            fit_derived_results = pd.read_csv(object_dir + "/results/ns_derived_table.csv")
            fit_results = pd.read_csv(object_dir + "/results/ns_table.csv")
            alles = alexfitter.allesclass(object_dir)
            candidates_count = len(fit_results[fit_results["#name"].str.contains("_period")])
            candidate_no = 0
            candidate.loc[:, 'period'] = AllesfitterDataExtractor.extract_period(candidate_no, fit_results, alles)
            candidate.loc[:, 't0'] = AllesfitterDataExtractor.extract_epoch(candidate_no, fit_results, alles)
            candidate.loc[:, 'duration'] = AllesfitterDataExtractor.extract_duration(candidate_no, fit_derived_results)
            candidate.loc[:, 'depth'] = AllesfitterDataExtractor.extract_depth(candidate_no, fit_derived_results)
            candidate.loc[:, 'a'] = AllesfitterDataExtractor.extract_semimajor_axis(candidate_no, fit_derived_results)
            rp = AllesfitterDataExtractor.extract_radius(candidate_no, fit_results)
            candidate.loc[:, 'rp_rs'] = (AllesfitterDataExtractor.extract_radius(candidate_no, fit_results) /
                                         LcbuilderHelper.convert_from_to(star_df["radius"], u.Rsun, u.Rearth))

        logging.info("Selected signal number " + str(candidate_selection))
    transits_mask = []
    for i in range(0, int(candidate['number']) - 1):
        transits_mask.append({"P": candidates.iloc[i]["period"], "T0": candidates.iloc[i]["t0"],
                              "D": candidates.iloc[i]["duration"] * 2})
    vetter.run(cpus, candidate=candidate, star_df=star_df, transits_df=transits_df, transits_mask=transits_mask,
               iatson_enabled=run_iatson, gpt_enabled=run_gpt, gpt_api_key=gpt_key, only_summary=only_summary,
               triceratops_bins=triceratops_bins, triceratops_scenarios=triceratops_scenarios,
               triceratops_curve_file=triceratops_curve_file,
               triceratops_contrast_curve_file=triceratops_contrast_curve_file,
               triceratops_additional_stars_file=triceratops_additional_stars_file,
               triceratops_sigma_mode=triceratops_sigma_mode,
               triceratops_ignore_ebs=triceratops_ignore_ebs,
               triceratops_resolved_companion=triceratops_resolved_companion,
               triceratops_ignore_background_stars=triceratops_ignore_background_stars,
               sectors=sectors
               )
