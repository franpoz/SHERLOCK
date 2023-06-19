import logging
import os
import shutil
import sys

import json
from pathlib import Path

import pandas as pd
from numpy import arange

from sherlockpipe.loading import common
from sherlockpipe.loading.common import get_from_dict
from sherlockpipe.system_stability.megno import MegnoStabilityCalculator
from sherlockpipe.system_stability.stability_calculator import PlanetInput


def run_stability(args):
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    stability_dir = object_dir + "/stability" if args.properties is None else object_dir + "/stability_" + str(Path(args.properties).stem)
    if os.path.exists(stability_dir) or os.path.isdir(stability_dir):
        shutil.rmtree(stability_dir, ignore_errors=True)
    os.mkdir(stability_dir)
    file_dir = stability_dir + "/stability.log"
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
    logging.info("Starting stability validation")
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    star_mass = star_df.iloc[0]["M_star"]
    star_mass_low_err = star_df.iloc[0]["M_star_lerr"]
    star_mass_up_err = star_df.iloc[0]["M_star_uerr"]
    star_mass_low = star_mass if star_mass_low_err is None else star_mass - star_mass_low_err
    star_mass_up = star_mass if star_mass_up_err is None else star_mass + star_mass_up_err
    star_mass_bins = args.star_mass_bins
    free_params = args.free_params.split(",") if args.free_params is not None else []
    planets_params = []
    if args.properties is None:
        candidates = pd.read_csv(object_dir + "/../candidates.csv")
        ns_derived_file = object_dir + "/results/ns_derived_table.csv"
        ns_file = object_dir + "/results/ns_table.csv"
        if not os.path.exists(ns_derived_file) or not os.path.exists(ns_file):
            raise ValueError("Bayesian fit posteriors files {" + ns_file + ", " + ns_derived_file + "} not found")
        fit_derived_results = pd.read_csv(object_dir + "/results/ns_derived_table.csv")
        fit_results = pd.read_csv(object_dir + "/results/ns_table.csv")
        candidates_count = len(fit_results[fit_results["#name"].str.contains("_period")])
        ecc_rows = fit_derived_results[fit_derived_results["#property"].str.contains("Eccentricity")]
        arg_periastron_rows = fit_derived_results[
            fit_derived_results["#property"].str.contains("Argument of periastron")]
        for i in arange(0, candidates_count):
            period_row = fit_results[fit_results["#name"].str.contains("_period")].iloc[i]
            period = float(period_row["median"])
            period_low_err = float(period_row["lower_error"])
            period_up_err = float(period_row["upper_error"])
            inc_row = fit_derived_results[fit_derived_results["#property"].str.contains("Inclination")].iloc[i]
            inclination = float(inc_row["value"])
            inc_low_err = float(inc_row["lower_error"])
            inc_up_err = float(inc_row["upper_error"])
            duration_row = \
            fit_derived_results[fit_derived_results["#property"].str.contains("Total transit duration")].iloc[i]
            duration = float(duration_row["value"])
            duration_low_err = float(duration_row["lower_error"])
            duration_up_err = float(duration_row["upper_error"])
            radius_row = fit_derived_results[fit_derived_results["#property"].str.contains("R_\{\\\\oplus}")].iloc[i]
            radius = float(radius_row["value"])
            radius_low_err = float(radius_row["lower_error"])
            radius_up_err = float(radius_row["upper_error"])
            if len(ecc_rows) > 0:
                ecc_row = ecc_rows.iloc[i]
                eccentricity = float(ecc_row["value"])
                ecc_low_err = float(ecc_row["lower_error"])
                ecc_up_err = float(ecc_row["upper_error"])
            else:
                eccentricity = 0.0
                ecc_low_err = 0.1
                ecc_up_err = 0.1
            if len(arg_periastron_rows) > 0:
                arg_periastron_row = arg_periastron_rows.iloc[i]
                arg_periastron = float(arg_periastron_row["value"])
                arg_periastron_low_err = float(arg_periastron_row["lower_error"])
                arg_periastron_up_err = float(arg_periastron_row["upper_error"])
            else:
                arg_periastron = 0.0
                arg_periastron_low_err = 20.0
                arg_periastron_up_err = 20.0
            planets_params.append(
                PlanetInput(period=period, period_low_err=period_low_err, period_up_err=period_up_err,
                            radius=radius, radius_low_err=radius_low_err, radius_up_err=radius_up_err,
                            eccentricity=eccentricity, ecc_low_err=ecc_low_err, ecc_up_err=ecc_up_err,
                            inclination=inclination, inc_low_err=inc_low_err, inc_up_err=inc_up_err,
                            omega=arg_periastron, omega_low_err=arg_periastron_low_err,
                            omega_up_err=arg_periastron_up_err, mass_bins=args.mass_bins, period_bins=args.period_bins,
                            ecc_bins=args.ecc_bins, inc_bins=args.inc_bins, omega_bins=args.omega_bins))
    else:
        user_properties = common.load_from_yaml(args.properties)
        user_planet_params = []
        for planet in user_properties["BODIES"]:
            period_bins = args.period_bins if "P_BINS" not in planet else planet["P_BINS"]
            ecc_bins = args.ecc_bins if "E_BINS" not in planet else planet["E_BINS"]
            mass_bins = args.mass_bins if "M_BINS" not in planet else planet["M_BINS"]
            inc_bins = args.inc_bins if "I_BINS" not in planet else planet["I_BINS"]
            om_bins = args.omega_bins if "O_BINS" not in planet else planet["O_BINS"]
            user_planet_params.append(PlanetInput(period=get_from_dict(planet, "P"),
                                                  period_low_err=get_from_dict(planet, "P_LOW"),
                                                  period_up_err=get_from_dict(planet, "P_UP"),
                                                  radius=get_from_dict(planet, "R"),
                                                  radius_low_err=get_from_dict(planet, "R_LOW"),
                                                  radius_up_err=get_from_dict(planet, "R_UP"),
                                                  mass=get_from_dict(planet, "M"),
                                                  mass_low_err=get_from_dict(planet, "M_LOW"),
                                                  mass_up_err=get_from_dict(planet, "M_UP"),
                                                  eccentricity=get_from_dict(planet, "E_LOW"),
                                                  ecc_low_err=get_from_dict(planet, "E_LOW"),
                                                  ecc_up_err=get_from_dict(planet, "E_UP"),
                                                  inclination=get_from_dict(planet, "I"),
                                                  inc_low_err=get_from_dict(planet, "I_LOW"),
                                                  inc_up_err=get_from_dict(planet, "I_UP"),
                                                  omega=get_from_dict(planet, "O"),
                                                  omega_low_err=get_from_dict(planet, "O_LOW"),
                                                  omega_up_err=get_from_dict(planet, "O_UP"),
                                                  period_bins=period_bins, mass_bins=mass_bins, ecc_bins=ecc_bins,
                                                  inc_bins=inc_bins, omega_bins=om_bins))
        planets_params = planets_params + user_planet_params
        if "STAR" in user_properties:
            star_mass = star_mass_low if "M" not in user_properties["STAR"] else user_properties["STAR"]["M"]
            star_mass_low = star_mass_low if "M_LOW" not in user_properties["STAR"] else star_mass - \
                                                                                         user_properties["STAR"][
                                                                                             "M_LOW"]
            star_mass_up = star_mass_up if "M_UP" not in user_properties["STAR"] else star_mass + \
                                                                                      user_properties["STAR"]["M_UP"]
            star_mass_bins = args.star_mass_bins if "M_BINS" not in user_properties["STAR"] else \
            user_properties["STAR"]["M_BINS"]
    stability_calculator = MegnoStabilityCalculator(args.years)
    logger.info("%.0f planets to be simulated", len(planets_params))
    logger.info("Lowest star mass: %.2f", star_mass_low)
    logger.info("Highest star mass: %.2f", star_mass_up)
    logger.info("Star mass bins: %.0f", star_mass_bins)
    for key, body in enumerate(planets_params):
        logger.info("Body %.0f: %s", key, json.dumps(body.__dict__))
    stability_calculator.run(stability_dir, star_mass_low, star_mass_up, star_mass_bins, planets_params, args.cpus, free_params)