import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator

from sherlockpipe.bayesian_fit.fit_report import FitReport
from sherlockpipe.bayesian_fit.fitter import Fitter
from sherlockpipe.bayesian_fit.fitter2 import Fitter2
from sherlockpipe.loading import common


def run_fit(args):
    index = 0
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    candidates_df = pd.read_csv(object_dir + "/candidates.csv")
    candidate_selections = []
    if args.candidate is not None:
        candidate_selections = str(args.candidate)
        candidate_selections = candidate_selections.split(",")
        candidate_selections = list(map(int, candidate_selections))
        fitting_dir = object_dir + "/fit_" + str(candidate_selections)
    else:
        fitting_dir = object_dir + "/fit_" + str(Path(args.properties).stem)
    if os.path.exists(fitting_dir) or os.path.isdir(fitting_dir):
        shutil.rmtree(fitting_dir, ignore_errors=True)
    os.mkdir(fitting_dir)
    file_dir = fitting_dir + "/fit.log"
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
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    if args.candidate is None:
        user_properties = common.load_from_yaml(args.properties)
        selected_candidates_df = pd.DataFrame(
            columns=['id', 'period', 't0', 'duration', 'cpus', 'rp_rs', 'a', 'number', 'name',
                     'lc'])
        selected_candidates_df = selected_candidates_df.append(user_properties["planets"], ignore_index=True)
        user_star_df = pd.DataFrame(columns=['R_star', 'M_star'])
        if "star" in user_properties and user_properties["star"] is not None:
            user_star_df = user_star_df.append(user_properties["star"], ignore_index=True)
            if user_star_df.iloc[0]["R_star"] is not None:
                star_df.at[0, "R_star"] = user_star_df.iloc[0]["R_star"]
            if user_star_df.iloc[0]["M_star"] is not None:
                star_df.at[0, "M_star"] = user_star_df.iloc[0]["M_star"]
            if user_star_df.iloc[0]["ld_a"] is not None:
                star_df.at[0, "ld_a"] = user_star_df.iloc[0]["ld_a"]
            if user_star_df.iloc[0]["ld_b"] is not None:
                star_df.at[0, "ld_b"] = user_star_df.iloc[0]["ld_b"]
            if ("a" not in user_properties["planet"] or user_properties["planet"]["a"] is None) \
                    and star_df.iloc[0]["M_star"] is not None and not np.isnan(star_df.iloc[0]["M_star"]):
                selected_candidates_df.at[0, "a"] = HabitabilityCalculator() \
                    .calculate_semi_major_axis(user_properties["planet"]["period"],
                                               user_properties["star"]["M_star"])
            elif ("a" not in user_properties["planet"] or user_properties["planet"]["a"] is None) \
                    and (star_df.iloc[0]["M_star"] is None or np.isnan(star_df.iloc[0]["M_star"])):
                raise ValueError("Cannot guess semi-major axis without star mass.")
        selected_candidates_df['number'] = user_properties["number"]
        selected_candidates_df['curve'] = user_properties["curve"]
        if selected_candidates_df.iloc[0]["a"] is None or np.isnan(selected_candidates_df.iloc[0]["a"]):
            raise ValueError("Semi-major axis is neither provided nor inferred.")
        if selected_candidates_df.iloc[0]["name"] is None:
            raise ValueError("You need to provide a name for your candidate.")
        if selected_candidates_df.iloc[0]["lc"] is None:
            raise ValueError("You need to provide a light curve relative path for your candidate.")
    else:
        selected_candidates_df = pd.read_csv(object_dir + "/candidates.csv")
        selected_candidates_df = selected_candidates_df.rename(columns={'Object Id': 'TICID'})
        selected_candidates_df["number"] = ""
        selected_candidates_df["name"] = ""
        for candidate_selection in candidate_selections:
            selected_candidates_df.loc[candidate_selection - 1, ['number']] = candidate_selection
            selected_candidates_df.loc[candidate_selection - 1, ['name']] = 'SOI_' + \
                                                                      str(selected_candidates_df['number'][
                                                                              candidate_selection - 1])
        selected_candidates_df = selected_candidates_df.iloc[
            [candidate_selection - 1 for candidate_selection in candidate_selections]]
        logging.info("Selected signal numbers " + str(candidate_selections))
    if args.cpus is None:
        cpus = os.cpu_count() - 1
    else:
        cpus = args.cpus
    fitter = Fitter(object_dir, fitting_dir, args.only_initial, len(selected_candidates_df) == 1, candidates_df,
                    args.mcmc, args.detrend)
    fitter.fit(selected_candidates_df, star_df, cpus, fitting_dir, args.tolerance, args.fit_orbit)

def run_fit2(args):
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    candidates_df = pd.read_csv(object_dir + "/candidates.csv")
    candidate_selections = []
    if args.candidate is not None:
        candidate_selections = str(args.candidate)
        candidate_selections = candidate_selections.split(",")
        candidate_selections = list(map(int, candidate_selections))
        fitting_dir = object_dir + "/fit_" + str(candidate_selections)
    else:
        fitting_dir = object_dir + "/fit_" + str(Path(args.properties).stem)
    if os.path.exists(fitting_dir) or os.path.isdir(fitting_dir):
        shutil.rmtree(fitting_dir, ignore_errors=True)
    os.mkdir(fitting_dir)
    file_dir = fitting_dir + "/fit.log"
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
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    if args.candidate is None:
        user_properties = common.load_from_yaml(args.properties)
        selected_candidates_df = pd.DataFrame(
            columns=['id', 'period', 't0', 'duration', 'cpus', 'rp_rs', 'a', 'number', 'name',
                     'lc'])
        selected_candidates_df = selected_candidates_df.append(user_properties["planets"], ignore_index=True)
        user_star_df = pd.DataFrame(columns=['R_star', 'M_star'])
        if "star" in user_properties and user_properties["star"] is not None:
            user_star_df = user_star_df.append(user_properties["star"], ignore_index=True)
            if user_star_df.iloc[0]["R_star"] is not None:
                star_df.at[0, "R_star"] = user_star_df.iloc[0]["R_star"]
            if user_star_df.iloc[0]["M_star"] is not None:
                star_df.at[0, "M_star"] = user_star_df.iloc[0]["M_star"]
            if user_star_df.iloc[0]["ld_a"] is not None:
                star_df.at[0, "ld_a"] = user_star_df.iloc[0]["ld_a"]
            if user_star_df.iloc[0]["ld_b"] is not None:
                star_df.at[0, "ld_b"] = user_star_df.iloc[0]["ld_b"]
            if ("a" not in user_properties["planet"] or user_properties["planet"]["a"] is None) \
                    and star_df.iloc[0]["M_star"] is not None and not np.isnan(star_df.iloc[0]["M_star"]):
                selected_candidates_df.at[0, "a"] = HabitabilityCalculator() \
                    .calculate_semi_major_axis(user_properties["planet"]["period"],
                                               user_properties["star"]["M_star"])
            elif ("a" not in user_properties["planet"] or user_properties["planet"]["a"] is None) \
                    and (star_df.iloc[0]["M_star"] is None or np.isnan(star_df.iloc[0]["M_star"])):
                raise ValueError("Cannot guess semi-major axis without star mass.")
        selected_candidates_df['number'] = user_properties["number"]
        selected_candidates_df['curve'] = user_properties["curve"]
        if selected_candidates_df.iloc[0]["a"] is None or np.isnan(selected_candidates_df.iloc[0]["a"]):
            raise ValueError("Semi-major axis is neither provided nor inferred.")
        if selected_candidates_df.iloc[0]["name"] is None:
            raise ValueError("You need to provide a name for your candidate.")
        if selected_candidates_df.iloc[0]["lc"] is None:
            raise ValueError("You need to provide a light curve relative path for your candidate.")
    else:
        selected_candidates_df = pd.read_csv(object_dir + "/candidates.csv")
        selected_candidates_df = selected_candidates_df.rename(columns={'Object Id': 'TICID'})
        selected_candidates_df["number"] = ""
        selected_candidates_df["name"] = ""
        for candidate_selection in candidate_selections:
            selected_candidates_df.loc[candidate_selection - 1, ['number']] = candidate_selection
            selected_candidates_df.loc[candidate_selection - 1, ['name']] = 'SOI_' + \
                                                                      str(selected_candidates_df['number'][
                                                                              candidate_selection - 1])
        selected_candidates_df = selected_candidates_df.iloc[
            [candidate_selection - 1 for candidate_selection in candidate_selections]]
        logging.info("Selected signal numbers " + str(candidate_selections))
    if args.cpus is None:
        cpus = os.cpu_count() - 1
    else:
        cpus = args.cpus
    if args.only_fit_candidate is None:
        only_fit_candidate_df = selected_candidates_df
    else:
        fit_candidate_numbers = [int(candidate_number) for candidate_number in str(args.only_fit_candidate).split(',')]
        only_fit_candidate_df = selected_candidates_df.loc[selected_candidates_df['number'].isin(fit_candidate_numbers)]
    fitter = Fitter2(object_dir, fitting_dir, args.only_initial, len(selected_candidates_df) == 1, candidates_df,
                    args.mcmc, args.detrend, rho_err_multi=args.rho_err_multi, sigma_err_multi=args.sigma_err_multi,
                     yerr_err_multi=args.yerr_err_multi)
    fitter.fit(selected_candidates_df, only_fit_candidate_df, star_df, cpus, fitting_dir, args.tolerance, args.fit_orbit)
    FitReport(fitting_dir,
              star_df['obj_id'].iloc[0],
              star_df['ra'].iloc[0],
              star_df['dec'].iloc[0],
              star_df['v'].iloc[0],
              star_df['j'].iloc[0],
              star_df['h'].iloc[0],
              star_df['k'].iloc[0],
              selected_candidates_df).create_report()
