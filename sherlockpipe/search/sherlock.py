import logging
import math
import multiprocessing
import pickle
import shutil
from typing import List
import traceback
import lightkurve
import pandas
import wotan
import matplotlib.pyplot as plt
import foldedleastsquares as tls
import numpy as np
import os
import sys

from lcbuilder.HarmonicSelector import HarmonicSelector
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.curve_preparer.Flattener import Flattener
from lcbuilder.curve_preparer.Flattener import FlattenInput
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.InvalidNumberOfSectorsError import InvalidNumberOfSectorsError
from watson.watson import Watson

from sherlockpipe.ois.OisManager import OisManager
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator

from sherlockpipe.plot.plotting import save_transit_plot
from sherlockpipe.scoring.helper import compute_border_score, harmonic_spectrum
from sherlockpipe.search.sherlock_target import SherlockTarget
from sherlockpipe.search.transitresult import TransitResult
from multiprocessing import Pool
from scipy import stats

from sherlockpipe.update import Updater


class Sherlock:
    """
    Main SHERLOCK PIPEline class to be used for loading input, setting up the running parameters and launch the
    analysis of the desired TESS, Kepler, K2 or csv objects light curves.
    """
    MASK_MODES = ['mask', 'subtract']
    RESULTS_DIR = './'
    VALID_DETREND_METHODS = ["biweight", "gp"]
    VALID_PERIODIC_DETREND_METHODS = ["biweight", "gp", "cosine", "cofiam"]
    NUM_CORES = multiprocessing.cpu_count() - 1
    OBJECT_ID_REGEX = "^(KIC|TIC|EPIC)[-_ ]([0-9]+)$"
    NUMBERS_REGEX = "[0-9]+$"
    transits_min_count = {}
    wl_min = {}
    wl_max = {}
    report = {}
    ois = None
    config_step = 0
    use_ois = False

    def __init__(self, sherlock_targets: List[SherlockTarget], explore: bool = False, update_ois: bool = False,
                 update_force: bool = False, update_clean: bool = False, cache_dir: str = os.path.expanduser('~') + "/",
                 results_dir: str = None):
        """
        Initializes a Sherlock object, loading the OIs from the csvs, setting up the detrend and transit configurations,
        storing the provided object_infos list and initializing the builders to be used to prepare the light curves for
        the provided object_infos.

        :param bool update_ois: Flag to signal SHERLOCK for updating the TOIs, KOIs and EPICs
        :param List[SherlockTarget] sherlock_targets: a list of objects information to be analysed
        :param bool explore: whether to only run the prepare stage for all objects
        :param bool update_ois: whether ois files should be updated
        :param bool update_force: whether a complete update of metadata should be done
        :param bool update_clean: whether current metadata should be wiped-out before update
        :param str cache_dir: directory to store caches for sherlock.
        :param str results_dir: directory to store results
        """
        self.explore = explore
        self.cache_dir = cache_dir
        self.__setup_logging()
        self.ois_manager = OisManager(self.cache_dir)
        self.setup_files(update_ois, update_force, update_clean, results_dir)
        self.sherlock_targets = sherlock_targets
        self.habitability_calculator = HabitabilityCalculator()
        self.lcbuilder = LcBuilder()

    def setup_files(self, refresh_ois, refresh_force, refresh_clean, results_dir=None):
        """
        Loads the objects of interest data from the downloaded CSVs.

        :param refresh_ois: Flag update the TOIs, KOIs and EPICs
        :param results_dir: Stores the directory to be used for the execution.
        :return: the Sherlock object itself
        """
        self.results_dir = results_dir if results_dir is not None else self.RESULTS_DIR
        self.load_ois(refresh_ois, refresh_force, refresh_clean)
        return self

    def refresh_ois(self):
        """
        Downloads the TOIs, KOIs and EPIC OIs into csv files.

        :return: the Sherlock object itself
        """
        self.ois_manager.update_tic_csvs()
        self.ois_manager.update_kic_csvs()
        self.ois_manager.update_epic_csvs()
        return self

    def load_ois(self, refresh_ois, refresh_force, refresh_clean):
        """
        Loads the csv OIs files into memory

        :return: the Sherlock object itself
        """
        Updater(self.cache_dir).update(refresh_clean, refresh_ois, refresh_force)
        self.ois = self.ois_manager.load_ois()
        return self

    def filter_hj_ois(self):
        """
        Filters the in-memory OIs given some basic filters associated to hot jupiters properties. This method is added
        as an example

        :return: the Sherlock object itself
        """
        self.use_ois = True
        self.ois = self.ois[self.ois["Disposition"].notnull()]
        self.ois = self.ois[self.ois["Period (days)"].notnull()]
        self.ois = self.ois[self.ois["Planet Radius (R_Earth)"].notnull()]
        self.ois = self.ois[self.ois["Planet Insolation (Earth Flux)"].notnull()]
        self.ois = self.ois[self.ois["Depth (ppm)"].notnull()]
        self.ois = self.ois[(self.ois["Disposition"] == "KP") | (self.ois["Disposition"] == "CP")]
        self.ois = self.ois[self.ois["Period (days)"] < 10]
        self.ois = self.ois[self.ois["Planet Radius (R_Earth)"] > 5]
        self.ois = self.ois[self.ois["Planet Insolation (Earth Flux)"] > 4]
        self.ois.sort_values(by=['Object Id', 'OI'])
        return self

    def filter_multiplanet_ois(self):
        """
        Filters the in-memory OIs given some basic filters associated to multiplanet targets. This method is added
        as an example

        :return: the Sherlock object itself
        """
        self.use_ois = True
        self.ois = self.ois[self.ois["Disposition"].notnull()]
        self.ois = self.ois[self.ois["Period (days)"].notnull()]
        self.ois = self.ois[self.ois["Period (days)"] > 0]
        self.ois = self.ois[self.ois["Depth (ppm)"].notnull()]
        self.ois = self.ois[
            (self.ois["Disposition"] == "KP") | (self.ois["Disposition"] == "CP") | (self.ois["Disposition"] == "PC")]
        self.ois = self.ois[self.ois.duplicated(subset=['Object Id'], keep=False)]
        self.ois.sort_values(by=['Object Id', 'OI'])
        return self

    def filter_high_snr_long_period_ois(self):
        """
        Filters the in-memory OIs given some basic filters associated to big and long-period targets. This method is added
        as an example

        :return: the Sherlock object itself
        """
        self.use_ois = True
        self.ois = self.ois[self.ois["Disposition"].notnull()]
        self.ois = self.ois[self.ois["Period (days)"].notnull()]
        self.ois = self.ois[self.ois["Period (days)"] > 20]
        self.ois = self.ois[self.ois["Depth (ppm)"].notnull()]
        self.ois = self.ois[self.ois["Depth (ppm)"] > 7500]
        self.ois = self.ois[
            (self.ois["Disposition"] == "KP") | (self.ois["Disposition"] == "CP") | (self.ois["Disposition"] == "PC")]
        self.ois.sort_values(by=['Object Id', 'OI'])
        return self

    def filter_ois(self, function):
        """Applies a function accepting the Sherlock objects of interests dataframe and stores the result into the
        Sherlock same ois dataframe.

        :param function: the function to be applied to filter the Sherlock OIs.
        :return: the Sherlock object itself
        """
        self.use_ois = True
        self.ois = function(self.ois)
        return self

    def limit_ois(self, offset=0, limit=0):
        """
        Limits the in-memory loaded OIs given an offset and a limit (like a pagination)

        :param offset: the position where the subset must start
        :param limit: maximum number of ois to be returned
        :return: the Sherlock object itself
        """
        if limit == 0:
            limit = len(self.ois.index)
        self.ois = self.ois[offset:limit]
        return self

    def run(self):
        """
        Entrypoint of Sherlock which launches the main execution for all the input object_infos
        """
        if len(self.sherlock_targets) == 0 and self.use_ois:
            self.sherlock_targets = [MissionObjectInfo('all', object_id)
                                     for object_id in self.ois["Object Id"].astype('string').unique()]
        for sherlock_target in self.sherlock_targets:
            self.__run_object(sherlock_target)

    def __min_transits_count(self, lc_build, sherlock_target):
        return sherlock_target.min_transits_count \
                if sherlock_target.min_transits_count > 0 else lc_build.transits_min_count

    def __run_object(self, sherlock_target):
        """
        Performs the analysis for one object_info

        :param sherlock_target: The object to be analysed.
        """
        sherlock_id = sherlock_target.object_info.sherlock_id()
        mission_id = sherlock_target.object_info.mission_id()
        try:
            lc_build = self.__prepare(sherlock_target)
            object_dir = self.__init_object_dir(sherlock_id)
            if lc_build.lc_data is not None:
                lc_build.lc_data.to_csv(object_dir + "/lc_data.csv")
            time = lc_build.lc.time.value
            flux = lc_build.lc.flux.value
            flux_err = lc_build.lc.flux_err.value
            min_transits_count = self.__min_transits_count(lc_build, sherlock_target)
            period_grid, oversampling = LcbuilderHelper.calculate_period_grid(time, sherlock_target.period_min,
                                                                sherlock_target.period_max,
                                                                sherlock_target.oversampling,
                                                                lc_build.star_info,
                                                                1)
            id_run = 1
            best_signal_score = 1
            self.report[sherlock_id] = []
            logging.info('================================================')
            logging.info('SEARCH RUNS with period grid: [%.2f - %.2f] and length %.0f', np.min(period_grid),
                         np.max(period_grid), len(period_grid))
            logging.info('================================================')
            lcs, wl = self.__detrend(sherlock_target, time, flux, flux_err,
                                     lc_build.star_info)
            lcs = np.concatenate(([flux], lcs), axis=0)
            object_info = sherlock_target.object_info
            object_dir = self.__init_object_dir(object_info.sherlock_id())
            i = 0
            for lc in lcs:
                lc_df = pandas.DataFrame(columns=['#time', 'flux', 'flux_err'])
                args = np.argwhere(~np.isnan(lc)).flatten()
                lc_df['#time'] = time[args]
                lc_df['flux'] = lc[args]
                lc_df['flux_err'] = np.array(flux_err[args])
                flux_err_mean = np.nanmean(lc_df['flux_err'])
                lc_df.loc[(lc_df['flux_err'] == 0) | np.isnan(lc_df['flux_err']), 'flux_err'] = flux_err_mean
                lc_df.to_csv(object_dir + "/lc_" + str(i) + ".csv", index=False)
                i = i + 1
            wl = np.concatenate(([0], wl), axis=0)
            transits_stats_df = pandas.DataFrame(columns=['candidate', 't0', 'depth', 'depth_err'])
            while not self.explore and best_signal_score == 1 and id_run <= sherlock_target.max_runs:
                self.__setup_object_logging(sherlock_id, False)
                object_report = {}
                logging.info("________________________________ run %s________________________________", id_run)
                transit_results, signal_selection = \
                    self.__analyse(sherlock_target, time, lcs, flux_err, lc_build.star_info, id_run,
                                   min_transits_count, lc_build.cadence, self.report[sherlock_id], wl,
                                   period_grid, lc_build.detrend_period)
                all_nan_results = len(np.argwhere(~np.isnan(signal_selection.transit_result.t0s)).flatten()) == 0
                if not all_nan_results:
                    for index in np.arange(len(signal_selection.transit_result.t0s)):
                        transits_stats_df = transits_stats_df.append({'candidate': str(int(id_run - 1)),
                                                  't0': signal_selection.transit_result.t0s[index],
                                                  'depth': signal_selection.transit_result.depths[index],
                                                  'depth_err': signal_selection.transit_result.depths_err[index]},
                                                 ignore_index=True)
                transits_stats_df = transits_stats_df.sort_values(by=['candidate', 't0'], ascending=True)
                transits_stats_df.to_csv(object_dir + "transits_stats.csv", index=False)
                best_signal_score = signal_selection.score
                object_report["Object Id"] = mission_id
                object_report["run"] = id_run
                object_report["score"] = best_signal_score
                object_report["curve"] = str(signal_selection.curve_index)
                object_report["snr"] = signal_selection.transit_result.snr
                object_report["sde"] = signal_selection.transit_result.sde
                object_report["fap"] = signal_selection.transit_result.fap
                object_report["border_score"] = signal_selection.transit_result.border_score
                object_report["harmonic"] = signal_selection.transit_result.harmonic
                object_report["period"] = signal_selection.transit_result.period
                object_report["per_err"] = signal_selection.transit_result.per_err
                object_report["duration"] = signal_selection.transit_result.duration * 60 * 24
                object_report["t0"] = signal_selection.transit_result.t0
                object_report["depth"] = signal_selection.transit_result.depth
                object_report["depth_err"] = signal_selection.transit_result.depth_err
                object_report["depth_sig"] = signal_selection.transit_result.depth / signal_selection.transit_result.depth_err
                if signal_selection.transit_result is not None and signal_selection.transit_result.results is not None\
                        and not all_nan_results:
                    object_report['rp_rs'] = signal_selection.transit_result.results.rp_rs
                    real_transit_args = np.argwhere(~np.isnan(signal_selection.transit_result
                                                              .results.transit_depths))
                    object_report["transits"] = np.array(signal_selection.transit_result
                                                         .results.transit_times)[real_transit_args.flatten()]
                    object_report["transits"] = ','.join(map(str, object_report["transits"]))
                else:
                    object_report['rp_rs'] = 0
                    object_report["transits"] = ""
                object_report["sectors"] = ','.join(map(str, lc_build.sectors)) \
                    if lc_build.sectors is not None and len(lc_build.sectors) > 0 else None
                object_report["oi"] = self.__find_matching_oi(sherlock_target.object_info, object_report["period"],
                                                              object_report['t0'])
                if best_signal_score == 1:
                    logging.info('New best signal is good enough to keep searching. Going to the next run.')
                    time, lcs = self.__apply_mask_from_transit_results(sherlock_target, time, lcs, transit_results,
                                                                        signal_selection.curve_index)
                    id_run += 1
                    if id_run > sherlock_target.max_runs:
                        logging.info("Max runs limit of %.0f is reached. Stopping.", sherlock_target.max_runs)
                else:
                    logging.info('New best signal does not look very promising. End')
                self.report[sherlock_id].append(object_report)
                self.__setup_object_report_logging(sherlock_id, True)
                logging.info("Listing most promising candidates for ID %s:", sherlock_id)
                logging.info("%-12s%-10s%-10s%-10s%-8s%-8s%-11s%-11s%-8s%-8s%-14s%-14s%-12s%-25s%-10s%-18s%-20s",
                             "Detrend no.", "Period", "Per_err", "Duration", "T0", "Depth", "Depth_err", "Depth_sig",
                             "SNR", "SDE", "Border_score", "Matching OI", "Harmonic",
                             "Planet radius (R_Earth)", "Rp/Rs", "Semi-major axis", "Habitability Zone")
                if sherlock_id in self.report:
                    candidates_df = pandas.DataFrame(columns=['curve', 'period', 'per_err', 'duration', 't0', 'depth',
                                                              'snr', 'sde', 'border_score', 'oi', 'rad_p', 'rp_rs',
                                                              'a', 'hz'])
                    i = 1
                    for report in self.report[sherlock_id]:
                        a, habitability_zone = self.habitability_calculator\
                            .calculate_hz_score(lc_build.star_info.teff, lc_build.star_info.mass,
                                                lc_build.star_info.lum, report["period"])
                        report['a'] = a
                        report['hz'] = habitability_zone
                        if lc_build.star_info.radius_assumed:
                            report['rad_p'] = np.nan
                        else:
                            report['rad_p'] = self.__calculate_planet_radius(lc_build.star_info, report["depth"])
                        logging.info("%-12s%-10.4f%-10.5f%-10.2f%-8.2f%-8.3f%-11.3f%-11.3f%-8.2f%-8.2f%-14.2f%-14s%-12s%-25.5f%-10.5f%-18.5f%-20s",
                                     report["curve"], report["period"], report["per_err"],
                                     report["duration"], report["t0"], report["depth"], report["depth_err"],
                                     report["depth_sig"], report["snr"], report["sde"],
                                     report["border_score"], report["oi"], report["harmonic"],
                                     report['rad_p'], report['rp_rs'], a, habitability_zone)
                        candidates_df = candidates_df.append(report, ignore_index=True)
                        i = i + 1
                    candidates_df.to_csv(object_dir + "candidates.csv", index=False)
        except InvalidNumberOfSectorsError:
            logging.exception("Invalid number of sectors exception")
            self.__remove_object_dir(sherlock_id)
        except Exception as e:
            logging.exception("Unexpected exception")

    def noise(self, time, flux, signal_power):
        from scipy.signal import periodogram, welch
        f_p, psd_p = periodogram(flux)
        f_w, psd_w = welch(flux)
        power_p = np.trapz(psd_p, f_p)
        power_w = np.trapz(psd_w, f_w)
        snr_p = signal_power / power_p
        snr_w = signal_power / power_w
        return snr_p

    def __init_object_dir(self, object_id, clean_dir=False):
        dir = self.results_dir + str(object_id)
        dir = dir.replace(" ", "_")
        dir = dir if not self.explore else dir + '_explore'
        if clean_dir:
            self.__remove_object_dir(object_id)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir + "/"

    def __remove_object_dir(self, object_id):
        dir = self.results_dir + str(object_id)
        dir = dir.replace(" ", "_")
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def __init_object_run_dir(self, object_id, run_no, clean_dir=False):
        dir = self.results_dir + str(object_id) + "/" + str(run_no)
        dir = dir.replace(" ", "_")
        if clean_dir and os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir + "/"

    def __setup_logging(self):
        if not isinstance(logging.root, logging.RootLogger):
            logging.root = logging.RootLogger(logging.INFO)
        logging.captureWarnings(True)
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger()
        while len(logger.handlers) > 0:
            logger.handlers.pop()
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def __setup_object_logging(self, object_id, clean_dir=True):
        if not isinstance(logging.root, logging.RootLogger):
            logging.root = logging.RootLogger(logging.INFO)
        object_dir = self.__init_object_dir(object_id, clean_dir)
        logger = logging.getLogger()
        while len(logger.handlers) > 1:
            logger.handlers.pop()
        formatter = logging.Formatter()
        handler = logging.FileHandler(object_dir + str(object_id) + "_report.log")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return object_dir

    def __setup_object_report_logging(self, object_id, clean=False):
        if not isinstance(logging.root, logging.RootLogger):
            logging.root = logging.RootLogger(logging.INFO)
        object_dir = self.__setup_object_logging(object_id, False)
        logger = logging.getLogger()
        logger.handlers.pop()
        formatter = logging.Formatter()
        file = object_dir + str(object_id) + "_candidates.log"
        if clean and os.path.exists(file):
            os.remove(file)
        handler = logging.FileHandler(file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def __prepare(self, sherlock_target):
        object_info = sherlock_target.object_info
        sherlock_id = object_info.sherlock_id()
        object_dir = self.__setup_object_logging(sherlock_id)
        logging.info('SHERLOCK (Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-base seeKers)')
        logging.info('Version %s', sys.modules["sherlockpipe"].__version__)
        logging.info('MODE: %s', "ANALYSE" if not self.explore else "EXPLORE")
        logging.info('ID: %s', sherlock_id)
        if sherlock_target.source_properties_file is not None:
            logging.info("Storing properties file %s", sherlock_target.source_properties_file)
            shutil.copy(sherlock_target.source_properties_file, object_dir + "/properties.yaml")
        logging.info('================================================')
        logging.info('USER DEFINITIONS')
        logging.info('================================================')
        if sherlock_target.detrend_method == "gp":
            logging.info('Detrend method: Gaussian Process Matern 2/3')
        else:
            logging.info('Detrend method: Bi-Weight')
        logging.info('No of detrend models applied: %s', sherlock_target.detrends_number)
        logging.info('Period planet protected: %.1f', sherlock_target.period_protect)
        logging.info('Minimum Period (d): %.1f', sherlock_target.period_min)
        logging.info('Maximum Period (d): %.1f', sherlock_target.period_max)
        logging.info('Binning size (min): %.1f', sherlock_target.bin_minutes)
        if object_info.initial_mask is not None:
            logging.info('Mask: yes')
        else:
            logging.info('Mask: no')
        if object_info.initial_transit_mask is not None:
            logging.info('Transit Mask: yes')
        else:
            logging.info('Transit Mask: no')
        logging.info('Threshold limit for SNR: %.1f', sherlock_target.snr_min)
        logging.info('Threshold limit for SDE: %.1f', sherlock_target.sde_min)
        if object_info.outliers_sigma:
            logging.info('Sigma threshold for upper outliers clipping: %.1f', object_info.outliers_sigma)
        if object_info.lower_outliers_sigma:
            logging.info('Sigma threshold for lower outliers clipping: %.1f', object_info.lower_outliers_sigma)
        logging.info("Fit method: %s", sherlock_target.fit_method)
        if sherlock_target.oversampling is not None:
            logging.info('Oversampling: %.3f', sherlock_target.oversampling)
        if sherlock_target.duration_grid_step is not None:
            logging.info('Duration step: %.3f', sherlock_target.duration_grid_step)
        if sherlock_target.t0_fit_margin is not None:
            logging.info('T0 Fit Margin: %.3f', sherlock_target.t0_fit_margin)
        logging.info('Signal scoring algorithm: %s', sherlock_target.best_signal_algorithm)
        if sherlock_target.best_signal_algorithm == sherlock_target.VALID_SIGNAL_SELECTORS[2]:
            logging.info('Quorum algorithm vote strength: %.2f',
                         sherlock_target.signal_score_selectors[sherlock_target.VALID_SIGNAL_SELECTORS[2]].strength)
        elif sherlock_target.best_signal_algorithm == sherlock_target.VALID_SIGNAL_SELECTORS[5]:
            logging.info('Quorum algorithm vote strength: %.2f',
                         sherlock_target.signal_score_selectors[sherlock_target.VALID_SIGNAL_SELECTORS[5]].strength)
        mission, mission_prefix, object_num = self.lcbuilder.parse_object_info(object_info.mission_id())
        if object_info.reduce_simple_oscillations and \
                object_info.oscillation_max_period < object_info.oscillation_min_period:
            logging.info("Stellar oscillation period has been set to empty. Defaulting to 1/3 the minimum search period")
            object_info.oscillation_max_period = sherlock_target.period_min / 3
        lc_build = self.lcbuilder.build(object_info, object_dir, cpus=sherlock_target.cpu_cores)
        min_transits_count = self.__min_transits_count(lc_build, sherlock_target)
        logging.info('Minimum number of transits: %s', min_transits_count)
        lightcurve_timespan = lc_build.lc.time[len(lc_build.lc.time) - 1] - lc_build.lc.time[0]
        if sherlock_target.search_zone is not None and not (lc_build.star_info.mass_assumed or
                                                            lc_build.star_info.radius_assumed):
            logging.info("Selected search zone: %s. Minimum and maximum periods will be calculated.",
                         sherlock_target.search_zone)
            min_and_max_period = sherlock_target.search_zones_resolvers[
                sherlock_target.search_zone].calculate_period_range(lc_build.star_info)
            if min_and_max_period is None:
                logging.info("Selected search zone was %s but cannot be calculated. Defaulting to minimum and " +
                             "maximum input periods", sherlock_target.search_zone)
            else:
                logging.info("Selected search zone periods are [%.2f, %.2f] days", min_and_max_period[0],
                             min_and_max_period[1])
                if min_and_max_period[0] > lightcurve_timespan or min_and_max_period[1] > lightcurve_timespan:
                    logging.info("Selected search zone period values are greater than lightcurve dataset. " +
                                 "Defaulting to minimum and maximum input periods.")
                else:
                    sherlock_target.period_min = min_and_max_period[0]
                    sherlock_target.period_max = min_and_max_period[1]
        elif sherlock_target.search_zone is not None:
            logging.info("Selected search zone was %s but star catalog info was not found or wasn't complete. " +
                         "Defaulting to minimum and maximum input periods.", sherlock_target.search_zone)
        if lc_build.sectors is not None:
            sectors_count = len(lc_build.sectors)
            logging.info('================================================')
            logging.info('SECTORS/QUARTERS/CAMPAIGNS INFO')
            logging.info('================================================')
            logging.info('Sectors/Quarters/Campaigns : %s', lc_build.sectors)
            if sectors_count < sherlock_target.min_sectors or sectors_count > sherlock_target.max_sectors:
                raise InvalidNumberOfSectorsError("The object " + sherlock_id + " contains " + str(sectors_count) +
                                                  " sectors/quarters/campaigns and the min and max selected are [" +
                                                  str(sherlock_target.min_sectors) + ", " + str(
                    sherlock_target.max_sectors) + "].")
        logging.info('================================================')
        logging.info("Detrend Window length / Kernel size")
        logging.info('================================================')
        if sherlock_target.detrend_l_min is not None and sherlock_target.detrend_l_max is not None:
            logging.info("Using user input WL / KS")
            self.wl_min[sherlock_id] = sherlock_target.detrend_l_min
            self.wl_max[sherlock_id] = sherlock_target.detrend_l_max
        else:
            logging.info("Using transit duration based WL or fixed KS")
            min_transit_duration = wotan.t14(R_s=lc_build.star_info.radius, M_s=lc_build.star_info.mass,
                                             P=sherlock_target.period_min, small_planet=True)
            max_transit_duration = wotan.t14(R_s=lc_build.star_info.radius, M_s=lc_build.star_info.mass,
                                             P=sherlock_target.period_max, small_planet=False)
            self.wl_min[sherlock_id] = 4 * min_transit_duration
            self.wl_max[sherlock_id] = 8 * max_transit_duration
        logging.info("wl/ks_min: %.2f", self.wl_min[sherlock_id])
        logging.info("wl/ks_max: %.2f", self.wl_max[sherlock_id])
        logging.info('================================================')
        if not sherlock_target.period_min:
            sherlock_target.period_min = lc_build.detrend_period * 4
            logging.info("Setting Min Period to %.3f due to auto_detrend period", sherlock_target.period_min)
        if lc_build.sectors is not None:
            logging.info("======================================")
            logging.info("Field Of View Plots")
            logging.info("======================================")
            fov_dir = object_dir + "/fov"
            Watson.vetting_field_of_view(fov_dir, mission, object_num, lc_build.cadence, lc_build.star_info.ra,
                                         lc_build.star_info.dec, lc_build.sectors if isinstance(lc_build.sectors, list)
                                         else lc_build.sectors.tolist(), lc_build.tpf_source, lc_build.tpf_apertures)
        if sherlock_target.ois_mask and self.ois is not None:
            logging.info("Masking OIS")
            target_ois = self.ois[self.ois["Object Id"] == object_info.mission_id()]
            target_ois = target_ois[target_ois["OI"].notnull()]
            transits_dict = []
            for index, oi_row in target_ois.iterrows():
                epoch = oi_row['Epoch (BJD)']
                epoch = LcbuilderHelper.correct_epoch(mission, epoch)
                transits_dict.append({'P': oi_row["Period (days)"], 'T0': epoch,
                                      'D': oi_row['Duration (hours)'] * 60 * 2})
            lc_build.lc = LcbuilderHelper.mask_transits_dict(lc_build.lc, transits_dict)
        return lc_build

    def __analyse(self, sherlock_target, time, lcs, flux_err, star_info, id_run, transits_min_count, cadence, report,
                  wl, period_grid, detrend_source_period):
        logging.info('=================================')
        logging.info('SEARCH OF SIGNALS - Run %s', id_run)
        logging.info('=================================')
        transit_results = self.__identify_signals(sherlock_target, time, lcs, flux_err, star_info, transits_min_count,
                                                  wl, id_run, cadence, report, period_grid, detrend_source_period)
        run_dir = self.__init_object_dir(star_info.object_id) + '/' + str(id_run) + '/'
        signal_selection = sherlock_target.signal_score_selectors[sherlock_target.best_signal_algorithm]\
            .select(id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence)
        title = 'Run ' + str(id_run) + '# curve: SELECTED' + \
                ' # P=' + format(signal_selection.transit_result.period, '.2f') + 'd # T0=' + \
                format(signal_selection.transit_result.t0, '.2f') + ' # Depth=' + \
                format(signal_selection.transit_result.depth, '.4f') + "ppt # Dur=" + \
                format(signal_selection.transit_result.duration * 24 * 60, '.0f') + 'm # SNR:' + \
                str(format(signal_selection.transit_result.snr, '.2f')) + ' # SDE:' + \
                str(format(signal_selection.transit_result.sde, '.2f')) + ' # BS:' + \
                str(format(signal_selection.transit_result.border_score, '.2f'))
        file = 'Run_' + str(id_run) + '_SELECTED_' + str(signal_selection.curve_index) + '_' + str(star_info.object_id) + '.png'
        save_transit_plot(star_info.object_id, title, run_dir, file, time, lcs[signal_selection.curve_index],
                          signal_selection.transit_result, cadence, id_run, sherlock_target.use_harmonics_spectra)
        if sherlock_target.pickle_mode == 'all':
            with open(run_dir + 'search_results.pickle', 'wb') as search_results_file:
                pickle.dump(transit_results, search_results_file, protocol=pickle.HIGHEST_PROTOCOL)
        elif sherlock_target.pickle_mode == 'selected':
            with open(run_dir + 'search_results.pickle', 'wb') as search_results_file:
                pickle.dump(transit_results[signal_selection.curve_index], search_results_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(signal_selection.get_message())
        return transit_results, signal_selection

    def __detrend(self, sherlock_target, time, lc, flux_err, star_info):
        wl_min = self.wl_min[star_info.object_id]
        wl_max = self.wl_max[star_info.object_id]
        bins = len(time) * 2 / sherlock_target.bin_minutes
        bin_means, bin_edges, binnumber = stats.binned_statistic(time, lc, statistic='mean', bins=bins)
        logging.info('=================================')
        logging.info('MODELS IN THE DETRENDING')
        logging.info('=================================')
        logging.info("%-25s%-17s%-15s%-11s%-15s", "light_curve", "Detrend_method", "win/ker_size", "RMS (ppm)",
                     "RMS_10min (ppm)")
        logging.info("%-25s%-17s%-15s%-11.2f%-15.2f", "PDCSAP_FLUX", "---", "---", np.std(lc) * 1e6,
                     np.std(bin_means[~np.isnan(bin_means)]) * 1e6)
        wl_step = (wl_max - wl_min) / sherlock_target.detrends_number
        wl = np.arange(wl_min, wl_max, wl_step)  # we define all the posibles window_length that we apply
        final_lcs = np.zeros((len(wl), len(lc)))
        flatten_inputs = []
        flattener = Flattener()
        if sherlock_target.detrend_cores > 1:
            for i in range(0, len(wl)):
                flatten_inputs.append(FlattenInput(time, lc, wl[i], sherlock_target.bin_minutes))
            if sherlock_target.detrend_method == 'gp':
                flatten_results = self.run_multiprocessing(sherlock_target.cpu_cores, flattener.flatten_gp, flatten_inputs)
            else:
                flatten_results = self.run_multiprocessing(sherlock_target.cpu_cores, flattener.flatten_bw, flatten_inputs)
        else:
            flatten_results = []
            for i in range(0, len(wl)):
                if sherlock_target.detrend_method == 'gp':
                    flatten_results.append(flattener.flatten_gp(FlattenInput(time, lc, wl[i], sherlock_target.bin_minutes)))
                else:
                    flatten_results.append(flattener.flatten_bw(FlattenInput(time, lc, wl[i], sherlock_target.bin_minutes)))
        i = 0
        # Plot entire curve detrends
        for flatten_lc_detrended, lc_trend, bin_centers, bin_means, flatten_wl in flatten_results:
            final_lcs[i] = flatten_lc_detrended
            logging.info("%-25s%-17s%-15.4f%-11.2f%-15.2f", 'flatten_lc & trend_lc ' + str(i),
                         sherlock_target.detrend_method,
                         flatten_wl, np.std(flatten_lc_detrended) * 1e6, np.std(bin_means[~np.isnan(bin_means)]) * 1e6)
            self.__plot_detrends(star_info.object_id, sherlock_target.detrends_number,
                                 sherlock_target.detrend_method, wl, time, lc, flatten_results, 0, len(time) - 1,
                                 'Detrends_' + str(star_info.object_id) + '.png')
            i = i + 1
        # Plot tokenized curve detrends
        dif = time[1:] - time[:-1]
        jumps = np.where(dif > 3)[0]
        jumps = np.append(jumps, len(time))
        previous_jump_index = 0
        for jumpIndex in jumps:
            self.__plot_detrends(star_info.object_id, sherlock_target.detrends_number, sherlock_target.detrend_method,
                                 wl, time, lc, flatten_results, previous_jump_index, jumpIndex)
            previous_jump_index = jumpIndex
        return final_lcs, wl

    def __plot_detrends(self, object_id, detrends_number, detrend_method, window_lengths, time, original_lc,
                        flatten_results, low_index, high_index, filename=None):
        plot_dir = self.__init_object_dir(object_id) + "/detrends/"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        figsize = (12, 4 * detrends_number)  # x,y
        rows = detrends_number
        cols = 1
        min_original = np.min(original_lc[low_index:high_index])
        shift = 2 * (1.0 - (np.min(original_lc)))  # shift in the between the raw and detrended data
        fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        if detrends_number > 1:
            axs = self.__trim_axs(axs, len(window_lengths))
        plot_axs = axs
        i = 0
        time_partial = time[low_index:high_index]
        original_lc_partial = original_lc[low_index:high_index]
        for flatten_lc_detrended, lc_trend, bin_centers, bin_means, flatten_wl in flatten_results:
            lc_trend_partial = lc_trend[low_index:high_index]
            dist = 1.3 * ((1 - min_original) + (np.max(flatten_lc_detrended) - 1))
            if detrends_number > 1:
                plot_axs = axs[i]
            if detrend_method == 'gp':
                plot_axs.set_title('ks=%s' % str(np.around(flatten_wl, decimals=4)))
            else:
                plot_axs.set_title('ws=%s d' % str(np.around(flatten_wl, decimals=4)))
            plot_axs.scatter(time_partial[1:], original_lc_partial[1:], color='black', s=1, alpha=0.25, rasterized=True)
            plot_axs.plot(time_partial[1:], lc_trend_partial[1:], linewidth=2, color='firebrick', alpha=1.)
            plot_axs.scatter(time_partial[1:], flatten_lc_detrended[low_index:high_index][1:] - dist, color='black', s=1, alpha=0.75, rasterized=True)
            plot_axs.set_ylabel("Flux norm.")
            plot_axs.set_xlabel("Time (d)")
            i = i + 1
        filename = filename if filename is not None else 'Detrends_' + str(object_id) + '_time_' + \
                                                         str(time_partial[1]) + '_' + str(time_partial[-1]) + '.png'
        plt.savefig(plot_dir + filename, dpi=200)
        fig.clf()
        plt.close(fig)

    def __identify_signals(self, sherlock_target, time, lcs, flux_err, star_info, transits_min_count, wl, id_run,
                           cadence, report, period_grid, detrend_source_period):
        object_info = sherlock_target.object_info
        detrend_logging_customs = 'ker_size' if sherlock_target.detrend_method == 'gp' else "win_size"
        logging.info("%-12s%-12s%-10s%-8s%-18s%-12s%-12s%-14s%-14s%-12s%-12s%-16s%-14s%-12s%-25s%-10s%-18s%-20s",
                     detrend_logging_customs, "Period", "Per_err", "N.Tran", "Mean Depth (ppt)", 'Depth_err',
                     'Depth_sig', "T. dur (min)", "T0", "SNR", "SDE", "Border_score", "Matching OI", "Harmonic",
                     "Planet radius (R_Earth)", "Rp/Rs", "Semi-major axis", "Habitability Zone")
        transit_results = {}
        plot_dir = self.__init_object_run_dir(star_info.object_id, id_run)
        if not sherlock_target.ignore_original:
            transit_result = self.__adjust_transit(sherlock_target, time, lcs[0], star_info, transits_min_count,
                                                   transit_results, report, cadence, period_grid, detrend_source_period)
            r_planet = self.__calculate_planet_radius(star_info, transit_result.depth)
            rp_rs = transit_result.results.rp_rs
            a, habitability_zone = self.habitability_calculator \
                .calculate_hz_score(star_info.teff, star_info.mass, star_info.lum, transit_result.period)
            oi = self.__find_matching_oi(object_info, transit_result.period, transit_result.t0)
            logging.info('%-12s%-12.5f%-10.6f%-8s%-18.3f%-12.3f%-12.3f%-14.1f%-14.4f%-12.3f%-12.3f%-16.2f%-14s%-12s%-25.5f%-10.5f%-18.5f%-20s',
                         "PDCSAP_FLUX", transit_result.period,
                         transit_result.per_err, transit_result.count, transit_result.depth, transit_result.depth_err,
                         transit_result.depth / transit_result.depth_err,
                         transit_result.duration * 24 * 60, transit_result.t0, transit_result.snr, transit_result.sde,
                         transit_result.border_score, oi, transit_result.harmonic, r_planet, rp_rs, a,
                         habitability_zone)
            plot_title = 'Run ' + str(id_run) + 'PDCSAP_FLUX # P=' + \
                         format(transit_result.period, '.2f') + 'd # T0=' + format(transit_result.t0, '.2f') + \
                         ' # Depth=' + format(transit_result.depth, '.4f') + 'ppt # Dur=' + \
                         format(transit_result.duration * 24 * 60, '.0f') + 'm # SNR:' + \
                         str(format(transit_result.snr, '.2f')) + ' # SDE:' + str(format(transit_result.sde, '.2f')) + \
                         ' # BS:' + str(format(transit_result.border_score, '.2f'))
            plot_file = 'Run_' + str(id_run) + '_PDCSAP-FLUX_' + str(star_info.object_id) + '.png'
            save_transit_plot(star_info.object_id, plot_title, plot_dir, plot_file, time, lcs[0], transit_result,
                              cadence, id_run, sherlock_target.use_harmonics_spectra)
        else:
            transit_result = TransitResult(None, None, 0, 0, 0, 0, [], [], 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, [])
        transit_results[0] = transit_result
        for i in range(1, len(wl)):
            transit_result = self.__adjust_transit(sherlock_target, time, lcs[i], star_info, transits_min_count,
                                                   transit_results, report, cadence, period_grid, detrend_source_period)
            transit_results[i] = transit_result
            r_planet = self.__calculate_planet_radius(star_info, transit_result.depth)
            rp_rs = transit_result.results.rp_rs
            a, habitability_zone = self.habitability_calculator \
                .calculate_hz_score(star_info.teff, star_info.mass, star_info.lum, transit_result.period)
            oi = self.__find_matching_oi(object_info, transit_result.period, transit_result.t0)
            logging.info('%-12.4f%-12.5f%-10.6f%-8s%-18.3f%-12.3f%-12.3f%-14.1f%-14.4f%-12.3f%-12.3f%-16.2f%-14s%-12s%-25.5f%-10.5f%-18.5f%-20s',
                         wl[i], transit_result.period,
                     transit_result.per_err, transit_result.count, transit_result.depth, transit_result.depth_err,
                         transit_result.depth / transit_result.depth_err,
                     transit_result.duration * 24 * 60, transit_result.t0, transit_result.snr, transit_result.sde,
                         transit_result.border_score, oi, transit_result.harmonic, r_planet, rp_rs, a,
                     habitability_zone)
            detrend_file_title_customs = 'ker_size' if sherlock_target.detrend_method == 'gp' else 'win_size'
            detrend_file_name_customs = 'ks' if sherlock_target.detrend_method == 'gp' else 'ws'
            title = 'Run ' + str(id_run) + '# ' + detrend_file_title_customs + ':' + str(format(wl[i], '.4f')) + \
                    ' # P=' + format(transit_result.period, '.2f') + 'd # T0=' + \
                    format(transit_result.t0, '.2f') + ' # Depth=' + format(transit_result.depth, '.4f') + "ppt # Dur=" + \
                    format(transit_result.duration * 24 * 60, '.0f') + 'm # SNR:' + \
                    str(format(transit_result.snr, '.2f')) + ' # SDE:' + str(format(transit_result.sde, '.2f')) + \
                    ' # BS:' + str(format(transit_result.border_score, '.2f'))
            file = 'Run_' + str(id_run) + '_' + detrend_file_name_customs + '=' + str(format(wl[i], '.4f')) + '_' + \
                   str(star_info.object_id) + '.png'
            save_transit_plot(star_info.object_id, title, plot_dir, file, time, lcs[i], transit_result, cadence,
                              id_run, sherlock_target.use_harmonics_spectra)
        return transit_results

    def __find_matching_oi(self, object_info, period, epoch):
        oi = ""
        if self.ois is not None:
            mission, mission_id, target_id = LcBuilder().parse_object_info(object_info.mission_id())
            corrected_epoch = LcbuilderHelper.normalize_mission_epoch(mission, epoch)
            existing_period_in_object = self.ois[(self.ois["Object Id"] == object_info.mission_id())]
            exists_oi = False
            for index, row in existing_period_in_object.iterrows():
                if HarmonicSelector.is_harmonic(0, (np.abs(corrected_epoch - row['Epoch (BJD)']) % period), period, row["Period (days)"]):
                    oi = row["OI"]
                    break
        return oi

    def __adjust_transit(self, sherlock_target, time, lc, star_info, transits_min_count, run_results, report, cadence,
                         period_grid, detrend_source_period):
        oversampling = sherlock_target.oversampling
        model = tls.transitleastsquares(time, lc)
        power_args = {"transit_template": sherlock_target.fit_method, "period_min": sherlock_target.period_min,
                      "period_max": sherlock_target.period_max, "n_transits_min": transits_min_count,
                      "T0_fit_margin": sherlock_target.t0_fit_margin, "show_progress_bar": False,
                      "use_threads": sherlock_target.cpu_cores, "oversampling_factor": oversampling,
                      "duration_grid_step": sherlock_target.duration_grid_step,
                      "period_grid": period_grid}
        if star_info.ld_coefficients is not None:
            power_args["u"] = star_info.ld_coefficients
        power_args["R_star"] = star_info.radius
        power_args["R_star_min"] = star_info.radius_min
        power_args["R_star_max"] = star_info.radius_max
        power_args["M_star"] = star_info.mass
        power_args["M_star_min"] = star_info.mass_min
        power_args["M_star_max"] = star_info.mass_max
        if sherlock_target.custom_transit_template is not None:
            power_args["transit_template_generator"] = sherlock_target.custom_transit_template
        results = model.power(**power_args)
        depths = (1 - results.transit_depths) * 1000
        depths_err = results.transit_depths_uncertainties * 1000
        if results.T0 != 0:
            depths_calc = results.transit_depths[~np.isnan(results.transit_depths)]
            depth = (1. - np.mean(depths_calc)) * 1000
            depth_err = np.sqrt(np.nansum([depth_err ** 2 for depth_err in depths_err])) / len(depths_err)
        else:
            t0s = results.transit_times
            depth = results.transit_depths
            depth_err = np.nan
        t0s = np.array(results.transit_times)
        in_transit = tls.transit_mask(time, results.period, results.duration, results.T0)
        transit_count = results.distinct_transit_count
        border_score = compute_border_score(time, results, in_transit, cadence)
        # Recalculating duration because of tls issue https://github.com/hippke/tls/issues/83
        intransit_folded_model = np.where(results['model_folded_model'] < 1.)[0]
        if len(intransit_folded_model) > 0:
            duration = results['period'] * (results['model_folded_phase'][intransit_folded_model[-1]]
                                            - results['model_folded_phase'][intransit_folded_model[0]])
        else:
            duration = results['duration']
        harmonic = self.__is_harmonic(results, run_results, report, detrend_source_period)
        harmonic_power = harmonic_spectrum(results['periods'], results.power)
        return TransitResult(power_args, results, results.period, results.period_uncertainty, duration,
                             results.T0, t0s, depths, depths_err, depth, depth_err, results.odd_even_mismatch,
                             (1 - results.depth_mean_even[0]) * 1000, (1 - results.depth_mean_odd[0]) * 1000, transit_count,
                             results.snr, results.SDE, results.FAP, border_score, in_transit, harmonic,
                             harmonic_power)

    def __calculate_planet_radius(self, star_info, depth):
        rp = np.nan
        try:
            rp = star_info.radius * math.sqrt(depth / 1000) / 0.0091577
        except ValueError as e:
            logging.error("Planet radius could not be calculated: depth=%s, star_radius=%s", depth, star_info.radius)
            logging.exception("Planet radius could not be calculated")
            print(e)
        return rp

    def __is_harmonic(self, tls_results, run_results, report, detrend_source_period):
        scales = [0.25, 0.5, 1, 2, 4]
        if detrend_source_period is not None:
            rotator_scale = round(tls_results.period / detrend_source_period, 2)
            rotator_harmonic = np.array(np.argwhere((np.array(scales) > rotator_scale - 0.02) & (np.array(scales) < rotator_scale + 0.02))).flatten()
            if len(rotator_harmonic) > 0:
                return str(scales[rotator_harmonic[0]]) + "*source"
        period_scales = [tls_results.period / round(item["period"], 2) for item in report]
        for key, period_scale in enumerate(period_scales):
            period_harmonic = np.array(np.argwhere((np.array(scales) > period_scale - 0.02) & (np.array(scales) < period_scale + 0.02))).flatten()
            if len(period_harmonic) > 0:
                period_harmonic = scales[period_harmonic[0]]
                return str(period_harmonic) + "*SOI" + str(key + 1)
        period_scales = [round(tls_results.period / run_results[key].period, 2) if run_results[key].period > 0 else 0 for key in run_results]
        for key, period_scale in enumerate(period_scales):
            period_harmonic = np.array(np.argwhere(
                (np.array(scales) > period_scale - 0.02) & (np.array(scales) < period_scale + 0.02))).flatten()
            if len(period_harmonic) > 0 and period_harmonic[0] != 2:
                period_harmonic = scales[period_harmonic[0]]
                return str(period_harmonic) + "*this(" + str(key) + ")"
        return "-"

    def __trim_axs(self, axs, N):
        [axis.remove() for axis in axs.flat[N:]]
        return axs.flat[:N]

    def __apply_mask_from_transit_results(self, sherlock_target, time, lcs, transit_results, best_signal_index):
        intransit = tls.transit_mask(time, transit_results[best_signal_index].period,
                                     2 * transit_results[best_signal_index].duration, transit_results[best_signal_index].t0)
        if sherlock_target.mask_mode == 'subtract':
            model_flux, model_flux_edges, model_flux_binnumber = stats.binned_statistic(
                transit_results[best_signal_index].results.model_lightcurve_time,
                transit_results[best_signal_index].results.model_lightcurve_model, statistic='mean', bins=len(intransit))
            for flux in lcs:
                flux[intransit] = flux[intransit] + np.full(len(flux[intransit]), 1) - model_flux[intransit]
                flux[intransit] = np.full(len(flux[intransit]), 1)
            clean_time = time
            clean_lcs = lcs
        else:
            clean_lcs = []
            for key, flux in enumerate(lcs):
                flux[intransit] = np.nan
                clean_time = time
                clean_lcs.append(flux)
        return clean_time, np.array(clean_lcs)

    def run_multiprocessing(self, n_processors, func, func_input):
        with Pool(processes=n_processors) as pool:
            return pool.map(func, func_input)

    class KoiInput:
        def __init__(self, star_id, kic_id):
            self.star_id = star_id
            self.kic_id = kic_id
