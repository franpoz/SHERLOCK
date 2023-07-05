import os
import traceback

import numpy as np
import lightkurve
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.objectinfo import MissionObjectInfo
from lcbuilder.star.starinfo import StarInfo

from sherlockpipe.search import sherlock
from sherlockpipe.loading.common import load_from_yaml, get_from_dict_or_default, get_from_user_or_config_or_default, \
    get_from_dict, get_from_user_or_config, extract_custom_class
from sherlockpipe.search.sherlock_target import SherlockTarget
from os import path


"""Used to launch the SHERLOCK search and also containing few helper methods"""


def get_star_info(object_id: str, target: dict) -> StarInfo:
    """
    Reads the properties for the target star and returns a StarInfo

    :param str object_id: the target object id
    :param dict target: the dictionary containing the target definition
    :return StarInfo: returns a StarInfo
    """
    input_star_info = None
    if isinstance(target, dict) and "STAR" in target and target["STAR"] is not None:
        star_properties = target["STAR"]
        input_star_info = StarInfo(object_id=object_id, ld_coefficients=tuple(
            star_properties["LD_COEFFICIENTS"]) if "LD_COEFFICIENTS" in star_properties else None,
                                   teff=star_properties["TEFF"] if "TEFF" in star_properties else None,
                                   lum=star_properties["LUM"] if "LUM" in star_properties else None,
                                   logg=star_properties["LOGG"] if "LOGG" in star_properties else None,
                                   radius=star_properties["RADIUS"] if "RADIUS" in star_properties else None,
                                   radius_min=star_properties[
                                       "RADIUS_LOWER_ERROR"] if "RADIUS_LOWER_ERROR" in star_properties else None,
                                   radius_max=star_properties[
                                       "RADIUS_UPPER_ERROR"] if "RADIUS_UPPER_ERROR" in star_properties else None,
                                   mass=star_properties["MASS"] if "MASS" in star_properties else None,
                                   mass_min=star_properties[
                                       "MASS_LOWER_ERROR"] if "MASS_LOWER_ERROR" in star_properties else None,
                                   mass_max=star_properties[
                                       "MASS_UPPER_ERROR"] if "MASS_UPPER_ERROR" in star_properties else None,
                                   ra=star_properties["RA"] if "RA" in star_properties else None,
                                   dec=star_properties["DEC"] if "DEC" in star_properties else None)
    return input_star_info


def extract_sectors(object_info: MissionObjectInfo, cache_dir: str) -> np.ndarray:
    """
    Given the object info and the cache directory, it searches in `lightkurve` for the target pixel files and retrieves
    the sectors, quarters or campaigns either as an integer (if only one) or as a list.

    :param MissionObjectInfo object_info: the object info with the target information
    :param str cache_dir: the lightkurve cache directory
    :return object: either an integer or a list of integers
    """
    mission, mission_prefix, id_int = LcBuilder().parse_object_info(object_info.mission_id())
    lcf_search_results = lightkurve.search_targetpixelfile(object_info.mission_id(),
                                                           mission=mission, cadence="long")\
        .download_all(download_dir=cache_dir)
    sector_name, sectors = LcbuilderHelper.mission_lightkurve_sector_extraction(mission, lcf_search_results)
    return sectors


def run_search(properties: str, explore: bool, results_dir: None, cpus: int = None):
    """
    Executes the SHERLOCK search reading the given properties file.

    :param properties: the properties directory
    :param explore: whether SHERLOCK should run only the steps previous to the search.
    :param cpus: the number of processes to be used
    :param results_dir: the output directory to store the results
    :return dict: the final used properties dictionary
    """
    resources_dir = os.path.dirname(path.join(path.dirname(__file__)))
    file_dir = resources_dir + "/" + 'properties.yaml' if resources_dir != "" and resources_dir is not None \
        else 'properties.yaml'
    sherlock_user_properties = load_from_yaml(file_dir)
    user_properties = load_from_yaml(properties)
    sherlock_user_properties.update(user_properties)
    sherlock.Sherlock([], explore, sherlock_user_properties["UPDATE_OIS"],
                      sherlock_user_properties["UPDATE_FORCE"], sherlock_user_properties["UPDATE_CLEAN"],
                      results_dir=results_dir).run()
    sherlock_targets = []
    lcbuilder = LcBuilder()
    cache_dir = get_from_dict_or_default(sherlock_user_properties, "CACHE_DIR", os.path.expanduser('~') + "/")
    for target, target_configs in sherlock_user_properties["TARGETS"].items():
        try:
            aperture = get_from_dict(target_configs, "APERTURE")
            file = get_from_user_or_config(target_configs, sherlock_user_properties, "FILE")
            author = get_from_user_or_config(target_configs, sherlock_user_properties, "AUTHOR")
            star_info = get_star_info(target, target_configs)
            min_sectors = get_from_user_or_config(target_configs, sherlock_user_properties, "MIN_SECTORS")
            max_sectors = get_from_user_or_config(target_configs, sherlock_user_properties, "MAX_SECTORS")
            bin_minutes = get_from_user_or_config(target_configs, sherlock_user_properties, "BIN_MINUTES")
            mask_mode = get_from_user_or_config(target_configs, sherlock_user_properties, "MASK_MODE")
            cpu_cores = get_from_user_or_config(target_configs, sherlock_user_properties, "CPU_CORES") \
                if cpus is None else cpus
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
            initial_transit_mask = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                           "INITIAL_TRANSIT_MASK")
            initial_ois_mask = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                           "INITIAL_OIS_MASK")
            sde_min = get_from_user_or_config(target_configs, sherlock_user_properties, "SDE_MIN")
            snr_min = get_from_user_or_config(target_configs, sherlock_user_properties, "SNR_MIN")
            custom_search_zone = extract_custom_class(
                get_from_user_or_config(target_configs, sherlock_user_properties, "CUSTOM_SEARCH_ZONE"))
            search_zone = get_from_user_or_config(target_configs, sherlock_user_properties, "SEARCH_ZONE")
            custom_transit_template = extract_custom_class(
                get_from_user_or_config(target_configs, sherlock_user_properties,
                                        "CUSTOM_TRANSIT_TEMPLATE"))
            custom_selection_algorithm = extract_custom_class(
                get_from_user_or_config(target_configs, sherlock_user_properties,
                                        "CUSTOM_SELECTION_ALGORITHM"))
            prepare_algorithm = extract_custom_class(get_from_user_or_config(target_configs, sherlock_user_properties,
                                                                             "PREPARE_ALGORITHM"))
            detrend_cores = get_from_user_or_config(target_configs, sherlock_user_properties, "DETREND_CORES") \
                if cpus is None else cpus
            detrends_number = get_from_user_or_config(target_configs, sherlock_user_properties, "DETRENDS_NUMBER")
            detrend_l_max = get_from_user_or_config(target_configs, sherlock_user_properties, "DETREND_L_MAX")
            detrend_l_min = get_from_user_or_config(target_configs, sherlock_user_properties, "DETREND_L_MIN")
            detrend_method = get_from_user_or_config(target_configs, sherlock_user_properties, "DETREND_METHOD")
            auto_detrend_period = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                          "AUTO_DETREND_PERIOD")
            auto_detrend_ratio = get_from_user_or_config(target_configs, sherlock_user_properties, "AUTO_DETREND_RATIO")
            auto_detrend_method = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                          "AUTO_DETREND_METHOD")
            auto_detrend_enabled = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                           "AUTO_DETREND_ENABLED")
            smooth_enabled = get_from_user_or_config(target_configs, sherlock_user_properties, "INITIAL_SMOOTH_ENABLED")
            high_rms_bin_hours = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                         "INITIAL_HIGH_RMS_BIN_HOURS")
            high_rms_threshold = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                         "INITIAL_HIGH_RMS_THRESHOLD")
            high_rms_enabled = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                       "INITIAL_HIGH_RMS_MASK")
            lower_outliers_sigma = get_from_user_or_config(target_configs, sherlock_user_properties, "UPPER_OUTLIERS_SIGMA")
            upper_outliers_sigma = get_from_user_or_config(target_configs, sherlock_user_properties, "LOWER_OUTLIERS_SIGMA")
            exptime = get_from_user_or_config(target_configs, sherlock_user_properties, "EXPTIME")
            eleanor_corr_flux = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                        "ELEANOR_CORRECTED_FLUX")
            reduce_simple_oscillations = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                                 "SIMPLE_OSCILLATIONS_REDUCTION")
            oscillation_snr_threshold = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                                "OSCILLATIONS_MIN_SNR")
            oscillation_amplitude_threshold = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                                      "OSCILLATIONS_AMPLITUDE_THRESHOLD")
            oscillation_ws_scale = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                           "OSCILLATIONS_WS_PERCENT")
            oscillation_min_period = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                             "OSCILLATIONS_MIN_PERIOD")
            oscillation_max_period = get_from_user_or_config(target_configs, sherlock_user_properties,
                                                             "OSCILLATIONS_MAX_PERIOD")
            exptime_binning = get_from_user_or_config(target_configs, sherlock_user_properties, "EXPTIME_BINNING")
            ignore_original = get_from_user_or_config(target_configs, sherlock_user_properties, "IGNORE_ORIGINAL")
            pickle_mode = get_from_user_or_config(target_configs, sherlock_user_properties, "PICKLE_MODE")
            mission = None
            mode = get_from_user_or_config_or_default(target_configs, sherlock_user_properties, "MODE", "GLOBAL")
            sectors = get_from_user_or_config_or_default(target_configs, sherlock_user_properties, "SECTORS", "all")
            use_harmonics_spectra = get_from_user_or_config_or_default(target_configs, sherlock_user_properties, "USE_HARMONICS_SPECTRA", False)
            truncate_border = get_from_user_or_config_or_default(target_configs, sherlock_user_properties, "TRUNCATE_BORDERS_DAYS", 0)
            min_transits_count = get_from_user_or_config_or_default(target_configs, sherlock_user_properties,
                                                                 "MIN_TRANSITS_COUNT", 1)

            if sectors != "all" and len(np.array(sectors).shape) > 1:
                if mode == "GLOBAL" or mode == "BOTH":
                    for sectors_subset in sectors:
                        built_object_info = lcbuilder.build_object_info(target, author, sectors_subset, file, exptime,
                                                                        initial_mask, initial_transit_mask,
                                                                        star_info, aperture, eleanor_corr_flux,
                                                                        upper_outliers_sigma,
                                                                        high_rms_enabled, high_rms_threshold,
                                                                        high_rms_bin_hours,
                                                                        smooth_enabled, auto_detrend_enabled,
                                                                        auto_detrend_method,
                                                                        auto_detrend_ratio, auto_detrend_period,
                                                                        prepare_algorithm,
                                                                        reduce_simple_oscillations,
                                                                        oscillation_snr_threshold,
                                                                        oscillation_amplitude_threshold,
                                                                        oscillation_ws_scale,
                                                                        oscillation_min_period, oscillation_max_period,
                                                                        exptime_binning, truncate_border,
                                                                        lower_outliers_sigma=lower_outliers_sigma)
                        sherlock_target = SherlockTarget(built_object_info,
                                                         detrend_method, detrend_l_min, detrend_l_max, detrends_number,
                                                         detrend_cores,
                                                         custom_selection_algorithm,
                                                         custom_transit_template,
                                                         search_zone, custom_search_zone,
                                                         snr_min, sde_min,
                                                         min_sectors, max_sectors,
                                                         bin_minutes,
                                                         mask_mode,
                                                         cpu_cores, max_runs, period_min,
                                                         period_max, period_protect, best_signal_algorithm,
                                                         quorum_strength,
                                                         min_quorum, fit_method, oversampling,
                                                         t0_fit_margin, duration_grid_step, properties, cache_dir,
                                                         ignore_original,
                                                         pickle_mode, use_harmonics_spectra, ois_mask=initial_ois_mask,
                                                         min_transits_count=min_transits_count)
                        sherlock_targets.append(sherlock_target)
                if mode == "SECTOR" or mode == "BOTH" and isinstance(built_object_info, MissionObjectInfo.MissionObjectInfo):
                    if sectors == 'all':
                        sectors = extract_sectors(built_object_info, cache_dir)
                    sectors_unique = np.unique(np.array(sectors).flatten())
                    for sector in sectors_unique:
                        object_info = lcbuilder.build_object_info(target, author, [sector], file, exptime,
                                                                  initial_mask, initial_transit_mask,
                                                                  star_info, aperture, eleanor_corr_flux,
                                                                  upper_outliers_sigma,
                                                                  high_rms_enabled, high_rms_threshold,
                                                                  high_rms_bin_hours,
                                                                  smooth_enabled, auto_detrend_enabled,
                                                                  auto_detrend_method,
                                                                  auto_detrend_ratio, auto_detrend_period,
                                                                  prepare_algorithm,
                                                                  reduce_simple_oscillations, oscillation_snr_threshold,
                                                                  oscillation_amplitude_threshold, oscillation_ws_scale,
                                                                  oscillation_min_period, oscillation_max_period,
                                                                  exptime_binning, truncate_border,
                                                                  lower_outliers_sigma=lower_outliers_sigma)
                        sherlock_target = SherlockTarget(object_info,
                                                         detrend_method, detrend_l_min, detrend_l_max, detrends_number,
                                                         detrend_cores,
                                                         custom_selection_algorithm,
                                                         custom_transit_template,
                                                         search_zone, custom_search_zone,
                                                         snr_min, sde_min,
                                                         min_sectors, max_sectors,
                                                         bin_minutes,
                                                         mask_mode,
                                                         cpu_cores, max_runs, period_min,
                                                         period_max, period_protect, best_signal_algorithm,
                                                         quorum_strength,
                                                         min_quorum, fit_method, oversampling,
                                                         t0_fit_margin, duration_grid_step, properties, cache_dir,
                                                         ignore_original,
                                                         pickle_mode, use_harmonics_spectra, ois_mask=initial_ois_mask,
                                                         min_transits_count=min_transits_count)
                        sherlock_targets.append(sherlock_target)
            else:
                built_object_info = lcbuilder.build_object_info(target, author, sectors, file, exptime,
                                                                initial_mask, initial_transit_mask,
                                                                star_info, aperture, eleanor_corr_flux,
                                                                upper_outliers_sigma,
                                                                high_rms_enabled, high_rms_threshold,
                                                                high_rms_bin_hours,
                                                                smooth_enabled, auto_detrend_enabled,
                                                                auto_detrend_method,
                                                                auto_detrend_ratio, auto_detrend_period,
                                                                prepare_algorithm,
                                                                reduce_simple_oscillations, oscillation_snr_threshold,
                                                                oscillation_amplitude_threshold, oscillation_ws_scale,
                                                                oscillation_min_period, oscillation_max_period,
                                                                exptime_binning, truncate_border,
                                                                lower_outliers_sigma=lower_outliers_sigma)
                sherlock_target = SherlockTarget(built_object_info,
                                                 detrend_method, detrend_l_min, detrend_l_max, detrends_number,
                                                 detrend_cores,
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
                                                 t0_fit_margin, duration_grid_step, properties, cache_dir,
                                                 ignore_original, pickle_mode, use_harmonics_spectra,
                                                 ois_mask=initial_ois_mask, min_transits_count=min_transits_count)
                if mode == "GLOBAL" or mode == "BOTH":
                    sherlock_targets.append(sherlock_target)
                if mode == "SECTOR" or mode == "BOTH" and isinstance(built_object_info, MissionObjectInfo.MissionObjectInfo):
                    if sectors == 'all':
                        sectors = extract_sectors(built_object_info, cache_dir)
                    for sector in sectors:
                        object_info = lcbuilder.build_object_info(target, author, [sector], file, exptime,
                                                                  initial_mask, initial_transit_mask,
                                                                  star_info, aperture, eleanor_corr_flux,
                                                                  upper_outliers_sigma,
                                                                  high_rms_enabled, high_rms_threshold,
                                                                  high_rms_bin_hours,
                                                                  smooth_enabled, auto_detrend_enabled,
                                                                  auto_detrend_method,
                                                                  auto_detrend_ratio, auto_detrend_period,
                                                                  prepare_algorithm,
                                                                  reduce_simple_oscillations, oscillation_snr_threshold,
                                                                  oscillation_amplitude_threshold, oscillation_ws_scale,
                                                                  oscillation_min_period, oscillation_max_period,
                                                                  exptime_binning, truncate_border,
                                                                  lower_outliers_sigma=lower_outliers_sigma)
                        sherlock_target = SherlockTarget(object_info,
                                                         detrend_method, detrend_l_min, detrend_l_max, detrends_number,
                                                         detrend_cores,
                                                         custom_selection_algorithm,
                                                         custom_transit_template,
                                                         search_zone, custom_search_zone,
                                                         snr_min, sde_min,
                                                         min_sectors, max_sectors,
                                                         bin_minutes,
                                                         mask_mode,
                                                         cpu_cores, max_runs, period_min,
                                                         period_max, period_protect, best_signal_algorithm,
                                                         quorum_strength,
                                                         min_quorum, fit_method, oversampling,
                                                         t0_fit_margin, duration_grid_step, properties, cache_dir,
                                                         ignore_original,
                                                         pickle_mode, use_harmonics_spectra, ois_mask=initial_ois_mask,
                                                         min_transits_count=min_transits_count)
                        sherlock_targets.append(sherlock_target)
                if mode != "GLOBAL" and mode != "BOTH" and not (mode == "SECTOR" or mode == "BOTH" and
                                                                isinstance(built_object_info, MissionObjectInfo.MissionObjectInfo)):
                    raise ValueError("Not a valid run mode: " + str(mode) + " for target: " + str(target))
        except Exception as e:
            print("Error found for target " + target)
            traceback.print_exc()
            print("Continuing with the target list")
    sherlock.Sherlock(sherlock_targets, explore, cache_dir=cache_dir, results_dir=results_dir).run()
    return sherlock_user_properties
