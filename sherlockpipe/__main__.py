import pickle
import sys

import lightkurve
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder

from lcbuilder.star.starinfo import StarInfo
from sherlockpipe import sherlock
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from argparse import ArgumentParser
from os import path
import yaml
import importlib.util
from pathlib import Path

from sherlockpipe.sherlock_target import SherlockTarget


def load_module(module_path):
    spec = importlib.util.spec_from_file_location("customs", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_star_info(target):
    input_star_info = None
    if isinstance(target, dict) and "STAR" in target and target["STAR"] is not None:
        star_properties = target["STAR"]
        input_star_info = StarInfo(id, tuple(
            star_properties["LD_COEFFICIENTS"]) if "LD_COEFFICIENTS" in star_properties else None,
                                   star_properties["TEFF"] if "TEFF" in star_properties else None,
                                   star_properties["LUM"] if "LUM" in star_properties else None,
                                   star_properties["LOGG"] if "LOGG" in star_properties else None,
                                   star_properties["RADIUS"] if "RADIUS" in star_properties else None,
                                   star_properties[
                                       "RADIUS_LOWER_ERROR"] if "RADIUS_LOWER_ERROR" in star_properties else None,
                                   star_properties[
                                       "RADIUS_UPPER_ERROR"] if "RADIUS_UPPER_ERROR" in star_properties else None,
                                   star_properties["MASS"] if "MASS" in star_properties else None,
                                   star_properties[
                                       "MASS_LOWER_ERROR"] if "MASS_LOWER_ERROR" in star_properties else None,
                                   star_properties[
                                       "MASS_UPPER_ERROR"] if "MASS_UPPER_ERROR" in star_properties else None,
                                   star_properties["RA"] if "RA" in star_properties else None,
                                   star_properties["DEC"] if "DEC" in star_properties else None)
    return input_star_info


def get_aperture(properties, id):
    input_aperture_file = None
    if properties["APERTURE"] is not None and properties["APERTURE"][id] is not None:
        input_aperture_file = properties["APERTURE"][id]
    return input_aperture_file


def extract_sectors(object_info):
    mission, mission_prefix, id_int = LcBuilder().parse_object_info(object_info.mission_id())
    if mission == "Kepler":
        lcf_search_results = lightkurve.search_targetpixelfile(object_info.mission_id(), mission=object_info.mission_id(),
                                                           cadence="long")
        object_sectors = lcf_search_results.download_all().quarter
    elif mission == "K2":
        lcf_search_results = lightkurve.search_targetpixelfile(object_info.mission_id(), mission=object_info.mission_id(),
                                                           cadence="long")
        object_sectors = lcf_search_results.download_all().campaign
    elif mission == "TESS":
        lcf_search_results = lightkurve.search_tesscut(object_info.mission_id())
        object_sectors = lcf_search_results.download_all().sector
    return object_sectors


def get_from_user_or_config(target, user_properties, key):
    value = None
    if key in user_properties:
        value = user_properties[key]
    if isinstance(target, dict):
        if key in target:
            value = target[key]
    return value

def get_from_user(target, key):
    value = None
    if isinstance(target, dict) and key in target:
        value = target[key]
    return value

def get_from_user_or_default(target, key, default):
    value = None
    if isinstance(target, dict) and key in target:
        value = target[key]
    return value if value is not None else default


def get_from_user_or_config_or_default(target, user_properties, key, default):
    value = None
    if key in user_properties:
        value = user_properties[key]
    if isinstance(target, dict) and key in target:
        value = target[key]
    return value if value is not None else default


def extract_custom_class(module_path):
    class_module = None
    if module_path is not None:
        class_module = load_module(module_path)
        class_name = Path(module_path.replace(".py", "")).name
        class_module = getattr(class_module, class_name)
        globals()[class_name] = class_module
        pickle.dumps(class_module)
        class_module = class_module()
    return class_module


if __name__ == '__main__':
    ap = ArgumentParser(description='Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-based seeKers')
    ap.add_argument('--properties', help="Additional properties to be loaded into Sherlock run ", required=True)
    ap.add_argument('--explore', dest='explore', action='store_true',
                    help="Whether to run using mcmc or ns. Default is ns.")
    args = ap.parse_args()
    resources_dir = path.join(path.dirname(__file__))
    file_dir = resources_dir + "/" + 'properties.yaml' if resources_dir != "" and resources_dir is not None \
        else 'properties.yaml'
    print(resources_dir)
    sherlock_user_properties = yaml.load(open(file_dir), yaml.SafeLoader)
    user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
    sherlock_user_properties.update(user_properties)
    sherlock.Sherlock([], args.explore, sherlock_user_properties["UPDATE_OIS"],
                      sherlock_user_properties["UPDATE_FORCE"], sherlock_user_properties["UPDATE_CLEAN"]).run()
    sherlock_targets = []
    mission_lightcurve_builder = MissionLightcurveBuilder()
    lcbuilder = LcBuilder()
    for target, target_configs in sherlock_user_properties["TARGETS"].items():
        aperture = get_from_user(target_configs, "APERTURE")
        sectors = get_from_user_or_default(target_configs, "SECTORS", "all")
        file = get_from_user(target_configs, "FILE")
        author = get_from_user(target_configs, "AUTHOR")
        star_info = get_star_info(target_configs)
        min_sectors = get_from_user_or_config(target_configs, sherlock_user_properties, "MIN_SECTORS")
        max_sectors = get_from_user_or_config(target_configs, sherlock_user_properties, "MAX_SECTORS")
        bin_minutes = get_from_user_or_config(target_configs, sherlock_user_properties, "BIN_MINUTES")
        mask_mode = get_from_user_or_config(target_configs, sherlock_user_properties, "MASK_MODE")
        cpu_cores = get_from_user_or_config(target_configs, sherlock_user_properties, "CPU_CORES")
        max_runs = get_from_user_or_config(target_configs, sherlock_user_properties, "MAX_RUNS")
        period_min = get_from_user_or_config(target_configs, sherlock_user_properties, "PERIOD_MIN")
        period_max = get_from_user_or_config(target_configs, sherlock_user_properties, "PERIOD_MAX")
        period_protect = get_from_user_or_config(target_configs, sherlock_user_properties, "PERIOD_PROTECT")
        best_signal_algorithm = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                        "BEST_SIGNAL_ALGORITHM")
        quorum_strength = get_from_user_or_config(target_configs, sherlock_user_properties, "QUORUM_STRENGTH")
        min_quorum = get_from_user_or_config(target_configs, sherlock_user_properties, "MIN_QUORUM")
        fit_method = get_from_user_or_config(target_configs, sherlock_user_properties, "FIT_METHOD")
        oversampling = get_from_user_or_config(target_configs, sherlock_user_properties, "OVERSAMPLING")
        t0_fit_margin = get_from_user_or_config(target_configs, sherlock_user_properties, "T0_FIT_MARGIN")
        duration_grid_step = get_from_user_or_config(target_configs, sherlock_user_properties, "DURATION_GRID_STEP")
        initial_mask = get_from_user_or_config(target_configs, sherlock_user_properties, "INITIAL_MASK")
        initial_transit_mask = get_from_user_or_config(target_configs, sherlock_user_properties, "INITIAL_TRANSIT_MASK")
        sde_min = get_from_user_or_config(target_configs, sherlock_user_properties, "SDE_MIN")
        snr_min = get_from_user_or_config(target_configs, sherlock_user_properties, "SNR_MIN")
        custom_search_zone = extract_custom_class(get_from_user_or_config(target_configs, sherlock_user_properties, "CUSTOM_SEARCH_ZONE"))
        search_zone = get_from_user_or_config(target_configs, sherlock_user_properties, "SEARCH_ZONE")
        custom_transit_template = extract_custom_class(get_from_user_or_config(target_configs, sherlock_user_properties,
                                                          "CUSTOM_TRANSIT_TEMPLATE"))
        custom_selection_algorithm = extract_custom_class(get_from_user_or_config(target_configs, sherlock_user_properties,
                                                             "CUSTOM_SELECTION_ALGORITHM"))
        prepare_algorithm = get_from_user_or_config(target_configs, sherlock_user_properties, "PREPARE_ALGORITHM")
        detrend_cores = get_from_user_or_config(target_configs, sherlock_user_properties, "DETREND_CORES")
        detrends_number = get_from_user_or_config(target_configs, sherlock_user_properties, "DETRENDS_NUMBER")
        detrend_l_max = get_from_user_or_config(target_configs, sherlock_user_properties, "DETREND_L_MAX")
        detrend_l_min = get_from_user_or_config(target_configs, sherlock_user_properties, "DETREND_L_MIN")
        detrend_method = get_from_user_or_config(target_configs, sherlock_user_properties, "DETREND_METHOD")
        auto_detrend_period = get_from_user_or_config(target_configs, sherlock_user_properties, "AUTO_DETREND_PERIOD")
        auto_detrend_ratio = get_from_user_or_config(target_configs, sherlock_user_properties, "AUTO_DETREND_RATIO")
        auto_detrend_method = get_from_user_or_config(target_configs, sherlock_user_properties, "AUTO_DETREND_METHOD")
        auto_detrend_enabled = get_from_user_or_config(target_configs, sherlock_user_properties, "AUTO_DETREND_ENABLED")
        smooth_enabled = get_from_user_or_config(target_configs, sherlock_user_properties, "INITIAL_SMOOTH_ENABLED")
        high_rms_bin_hours = get_from_user_or_config(target_configs, sherlock_user_properties, "INITIAL_HIGH_RMS_BIN_HOURS")
        high_rms_threshold = get_from_user_or_config(target_configs, sherlock_user_properties, "INITIAL_HIGH_RMS_THRESHOLD")
        high_rms_enabled = get_from_user_or_config(target_configs, sherlock_user_properties, "INITIAL_HIGH_RMS_MASK")
        outliers_sigma = get_from_user_or_config(target_configs, sherlock_user_properties, "OUTLIERS_SIGMA")
        exptime = get_from_user_or_config(target_configs, sherlock_user_properties, "EXPTIME")
        mission = None
        mode = get_from_user_or_config_or_default(target_configs, sherlock_user_properties, "MODE", "GLOBAL")
        built_object_info = lcbuilder.build_object_info(target, author, sectors, file, exptime,
                                                        initial_mask, initial_transit_mask, auto_detrend_period,
                                                        star_info, aperture)
        sherlock_target = SherlockTarget(built_object_info, high_rms_enabled, high_rms_threshold,
                       high_rms_bin_hours, smooth_enabled,
                       auto_detrend_enabled, auto_detrend_method, auto_detrend_ratio,
                       auto_detrend_period,
                       detrend_method, detrend_l_min, detrend_l_max, detrends_number, detrend_cores,
                       prepare_algorithm,
                       custom_selection_algorithm,
                       custom_transit_template,
                       search_zone, custom_search_zone,
                       snr_min, sde_min,
                       min_sectors, max_sectors,
                       bin_minutes,
                       mask_mode,
                       cpu_cores, max_runs, period_min,
                       period_max, period_protect, best_signal_algorithm, quorum_strength,
                       min_quorum, fit_method, oversampling,
                       t0_fit_margin, duration_grid_step, outliers_sigma)
        if mode == "GLOBAL" or mode == "BOTH":
            sherlock_targets.append(sherlock_target)
        if mode == "SECTOR" or mode == "BOTH" and isinstance(built_object_info, (
                MissionObjectInfo, MissionFfiCoordsObjectInfo, MissionFfiIdObjectInfo)):
            if sectors == 'all':
                sectors = extract_sectors(built_object_info)
            for sector in sectors:
                object_info = lcbuilder.build_object_info(target, author, [sector], file, exptime,
                                                        initial_mask, initial_transit_mask, auto_detrend_period,
                                                        star_info, aperture)
                sherlock_target = SherlockTarget(object_info, high_rms_enabled, high_rms_threshold,
                               high_rms_bin_hours, smooth_enabled,
                               auto_detrend_enabled, auto_detrend_method, auto_detrend_ratio,
                               auto_detrend_period,
                               detrend_method, detrend_l_min, detrend_l_max, detrends_number, detrend_cores,
                               prepare_algorithm,
                               custom_selection_algorithm,
                               custom_transit_template,
                               search_zone, custom_search_zone,
                               snr_min, sde_min,
                               min_sectors, max_sectors,
                               bin_minutes,
                               mask_mode,
                               cpu_cores, max_runs, period_min,
                               period_max, period_protect, best_signal_algorithm, quorum_strength,
                               min_quorum, fit_method, oversampling,
                               t0_fit_margin, duration_grid_step, outliers_sigma)
                sherlock_targets.append(sherlock_target)
        if mode != "GLOBAL" and mode != "BOTH" and not (mode == "SECTOR" or mode == "BOTH" and isinstance(built_object_info, (
                MissionObjectInfo, MissionFfiCoordsObjectInfo, MissionFfiIdObjectInfo))):
            raise ValueError("Not a valid run mode: " + str(mode) + " for target: " + str(target))
    sherlock.Sherlock(sherlock_targets, args.explore).run()
