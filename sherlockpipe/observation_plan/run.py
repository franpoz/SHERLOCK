"""Run script for observation plan"""
import logging
import shutil
import os

import alexfitter
import numpy as np
import pandas as pd
from lcbuilder.lcbuilder_class import LcBuilder
from sherlockpipe.observation_plan.observation_report import ObservationReport
from sherlockpipe.observation_plan.planner import Planner
from astropy.time import Time
from sherlockpipe.bayesian_fit.allesfitter_data_extractor import AllesfitterDataExtractor


def run_plan(args):
    if not isinstance(logging.root, logging.RootLogger):
        logging.root = logging.RootLogger(logging.INFO)
    if args.observatories is None and (args.lat is None or args.lon is None or args.alt is None):
        raise ValueError("You either need to set the 'observatories' property or the lat, lon and alt.")
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    ns_derived_file = object_dir + "/results/ns_derived_table.csv"
    ns_file = object_dir + "/results/ns_table.csv"
    if not os.path.exists(ns_derived_file) or not os.path.exists(ns_file):
        raise ValueError("Bayesian fit posteriors files {" + ns_file + ", " + ns_derived_file + "} not found")
    plan_dir = object_dir + "/plan"
    if os.path.exists(plan_dir):
        shutil.rmtree(plan_dir, ignore_errors=True)
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    object_id = star_df.iloc[0]["obj_id"]
    ra = star_df.iloc[0]["ra"]
    dec = star_df.iloc[0]["dec"]
    fit_derived_results = pd.read_csv(object_dir + "/results/ns_derived_table.csv")
    fit_results = pd.read_csv(object_dir + "/results/ns_table.csv")
    candidates_count = len(fit_results[fit_results["#name"].str.contains("_period")])
    alles = alexfitter.allesclass(object_dir)
    since = Time.now() if args.since is None else Time(args.since, scale='utc')
    percentile = 99.7 if args.error_sigma == 3 else 95 if args.error_sigma == 2 else 68
    for i in np.arange(0, candidates_count):
        period, period_low_err, period_up_err = AllesfitterDataExtractor.extract_period(i, fit_results, alles)
        epoch, epoch_low_err, epoch_up_err = AllesfitterDataExtractor.extract_epoch(i, fit_results, alles)
        duration, duration_low_err, duration_up_err = AllesfitterDataExtractor.extract_duration(i, fit_derived_results, alles)
        depth, depth_low_err, depth_up_err = AllesfitterDataExtractor.extract_depth(i, fit_derived_results, alles)
        period_row = fit_results[fit_results["#name"].str.contains("_period")].iloc[i]
        name = object_id + "_" + period_row["#name"].replace("_period", "")
        mission, mission_prefix, id_int = LcBuilder().parse_object_info(object_id)
        if args.time_unit is None and mission == "TESS":
            time_unit = 'btjd'
        elif args.time_unit is None and mission == "Kepler" or mission == "K2":
            time_unit = 'bkjd'
        elif args.time_unit is not None:
            time_unit = args.time_unit
        else:
            time_unit = 'jd'
        observatories_df, observables_df, alert_date, plan_dir, images_dir = \
            Planner.create_observation_observables(object_id, object_dir, ra, dec, since, name, epoch,
                                                   epoch_low_err, epoch_up_err, period, period_low_err,
                                                   period_up_err, duration, args.observatories, args.tz, args.lat,
                                                   args.lon, args.alt, args.max_days, args.min_altitude,
                                                   args.moon_min_dist, args.moon_max_dist, args.transit_fraction,
                                                   args.baseline, not args.no_error_alert, time_unit)
        report = ObservationReport(observatories_df, observables_df, alert_date, object_id, name, plan_dir, ra, dec,
                                   epoch, epoch_low_err, epoch_up_err, period, period_low_err, period_up_err, duration,
                                   duration_low_err, duration_up_err, depth, depth_low_err, depth_up_err,
                                   args.transit_fraction, args.moon_min_dist, args.moon_max_dist, args.min_altitude,
                                   args.max_days, star_df.iloc[0]["v"], star_df.iloc[0]["j"], star_df.iloc[0]["h"],
                                   star_df.iloc[0]["k"])
        report.create_report()
        shutil.rmtree(images_dir, ignore_errors=True)
