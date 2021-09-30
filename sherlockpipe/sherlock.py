import logging
import math
import multiprocessing
import shutil
import pandas
import wotan
import matplotlib.pyplot as plt
import foldedleastsquares as tls
from foldedleastsquares.template_generator.default_transit_template_generator import DefaultTransitTemplateGenerator
import numpy as np
import os
import sys

from lcbuilder.helper import LcbuilderHelper
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.curve_preparer.Flattener import Flattener
from lcbuilder.curve_preparer.Flattener import FlattenInput
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
from lcbuilder.objectinfo.InvalidNumberOfSectorsError import InvalidNumberOfSectorsError
from sherlockpipe.ois.OisManager import OisManager
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from sherlockpipe.transitresult import TransitResult
from multiprocessing import Pool
from scipy.ndimage.interpolation import shift
from scipy import stats

from sherlockpipe.update import Updater
from sherlockpipe.vet import Vetter


class Sherlock:
    """
    Main SHERLOCK PIPEline class to be used for loading input, setting up the running parameters and launch the
    analysis of the desired TESS, Kepler, K2 or csv objects light curves.
    """
    TOIS_CSV_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    CTOIS_CSV_URL = 'https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv'
    KOIS_LIST_URL = 'https://exofop.ipac.caltech.edu/kepler/targets.php?sort=num-pc&page1=1&ipp1=100000&koi1=&koi2='
    KOI_TARGET_URL = 'https://exofop.ipac.caltech.edu/kepler/edit_target.php?id={$id}'
    KOI_TARGET_URL_NEW = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative'
    EPIC_TARGET_URL_NEW = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2candidates'
    MASK_MODES = ['mask', 'subtract']
    RESULTS_DIR = './'
    VALID_DETREND_METHODS = ["biweight", "gp"]
    VALID_PERIODIC_DETREND_METHODS = ["biweight", "gp", "cosine", "cofiam"]
    VALID_SIGNAL_SELECTORS = ["basic", "border-correct", "quorum"]
    NUM_CORES = multiprocessing.cpu_count() - 1
    TOIS_CSV = RESULTS_DIR + 'tois.csv'
    CTOIS_CSV = RESULTS_DIR + 'ctois.csv'
    KOIS_CSV = RESULTS_DIR + 'kois.csv'
    OBJECT_ID_REGEX = "^(KIC|TIC|EPIC)[-_ ]([0-9]+)$"
    NUMBERS_REGEX = "[0-9]+$"
    MISSION_ID_KEPLER = "KIC"
    MISSION_ID_KEPLER_2 = "EPIC"
    MISSION_ID_TESS = "TIC"
    transits_min_count = {}
    wl_min = {}
    wl_max = {}
    report = {}
    ois = None
    config_step = 0
    use_ois = False

    def __init__(self, sherlock_targets: list, explore=False, update_ois=False, update_force=False, update_clean=False,
                 cache_dir=os.path.expanduser('~') + "/"):
        """
        Initializes a Sherlock object, loading the OIs from the csvs, setting up the detrend and transit configurations,
        storing the provided object_infos list and initializing the builders to be used to prepare the light curves for
        the provided object_infos.
        @param update_ois: Flag to signal SHERLOCK for updating the TOIs, KOIs and EPICs
        @param sherlock_targets: a list of objects information to be analysed
        @param explore: whether to only run the prepare stage for all objects
        @param cache_dir: directory to store caches for sherlock.
        """
        self.explore = explore
        self.cache_dir = cache_dir
        self.ois_manager = OisManager(self.cache_dir)
        self.setup_files(update_ois, update_force, update_clean)
        self.sherlock_targets = sherlock_targets
        self.habitability_calculator = HabitabilityCalculator()
        self.detrend_plot_axis = [[1, 1], [2, 1], [3, 1], [2, 2], [3, 2], [3, 2], [3, 3], [3, 3], [3, 3], [4, 3],
                                  [4, 3], [4, 3], [4, 4], [4, 4], [4, 4], [4, 4], [5, 4], [5, 4], [5, 4], [5, 4],
                                  [6, 4], [6, 4], [6, 4], [6, 4]]
        self.detrend_plot_axis.append([1, 1])
        self.detrend_plot_axis.append([2, 1])
        self.detrend_plot_axis.append([3, 1])
        self.lcbuilder = LcBuilder()

    def setup_files(self, refresh_ois, refresh_force, refresh_clean, results_dir=RESULTS_DIR):
        """
        Loads the objects of interest data from the downloaded CSVs.
        @param refresh_ois: Flag update the TOIs, KOIs and EPICs
        @param results_dir: Stores the directory to be used for the execution.
        @return: the Sherlock object itself
        """
        self.results_dir = results_dir
        self.load_ois(refresh_ois, refresh_force, refresh_clean)
        return self

    def refresh_ois(self):
        """
        Downloads the TOIs, KOIs and EPIC OIs into csv files.
        @return: the Sherlock object itself
        @rtype: Sherlock
        """
        self.ois_manager.update_tic_csvs()
        self.ois_manager.update_kic_csvs()
        self.ois_manager.update_epic_csvs()
        return self

    def load_ois(self, refresh_ois, refresh_force, refresh_clean):
        """
        Loads the csv OIs files into memory
        @return: the Sherlock object itself
        @rtype: Sherlock
        """
        Updater(self.cache_dir).update(refresh_clean, refresh_ois, refresh_force)
        self.ois = self.ois_manager.load_ois()
        return self

    def filter_hj_ois(self):
        """
        Filters the in-memory OIs given some basic filters associated to hot jupiters properties. This method is added
        as an example
        @return: the Sherlock object itself
        @rtype: Sherlock
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
        @return: the Sherlock object itself
        @rtype: Sherlock
        """
        self.use_ois = True
        self.ois = self.ois[self.ois["Disposition"].notnull()]
        self.ois = self.ois[self.ois["Period (days)"].notnull()]
        self.ois = self.ois[self.ois["Depth (ppm)"].notnull()]
        self.ois = self.ois[(self.ois["Disposition"] == "KP") | (self.ois["Disposition"] == "CP") | (self.ois["Disposition"] == "PC")]
        self.ois = self.ois[self.ois.duplicated(subset=['Object Id'], keep=False)]
        self.ois.sort_values(by=['Object Id', 'OI'])
        return self

    def filter_ois(self, function):
        """
        Applies a function accepting the Sherlock objects of interests dataframe and stores the result into the
        Sherlock same ois dataframe.
        @param function: the function to be applied to filter the Sherlock OIs.
        @return: the Sherlock object itself
        @rtype: Sherlock
        """
        self.use_ois = True
        self.ois = function(self.ois)
        return self

    def limit_ois(self, offset=0, limit=0):
        """
        Limits the in-memory loaded OIs given an offset and a limit (like a pagination)
        @param offset:
        @param limit:
        @return: the Sherlock object itself
        @rtype: Sherlock
        """
        if limit == 0:
            limit = len(self.ois.index)
        self.ois = self.ois[offset:limit]
        return self

    def run(self):
        """
        Entrypoint of Sherlock which launches the main execution for all the input object_infos
        """
        self.__setup_logging()
        logging.info('SHERLOCK (Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-base seeKers)')
        logging.info('Version %s', sys.modules["sherlockpipe"].__version__)
        logging.info('MODE: %s', "ANALYSE" if not self.explore else "EXPLORE")
        if len(self.sherlock_targets) == 0 and self.use_ois:
            self.sherlock_targets = [MissionObjectInfo(object_id, 'all')
                                     for object_id in self.ois["Object Id"].astype('string').unique()]
        for sherlock_target in self.sherlock_targets:
            self.__run_object(sherlock_target)

    def __run_object(self, sherlock_target):
        """
        Performs the analysis for one object_info
        @param sherlock_target: The object to be analysed.
        @type sherlock_target: ObjectInfo
        """
        sherlock_id = sherlock_target.object_info.sherlock_id()
        mission_id = sherlock_target.object_info.mission_id()
        try:
            lc_build = self.__prepare(sherlock_target)
            time = lc_build.lc.time.value
            flux = lc_build.lc.flux.value
            flux_err = lc_build.lc.flux_err.value
            period_grid = LcbuilderHelper.calculate_period_grid(time, sherlock_target.period_min,
                                                                sherlock_target.period_max,
                                                                sherlock_target.oversampling,
                                                                lc_build.star_info, lc_build.transits_min_count)
            id_run = 1
            best_signal_score = 1
            self.report[sherlock_id] = []
            logging.info('================================================')
            logging.info('SEARCH RUNS with period grid: [%.2f - %.2f] and length %.0f', np.min(period_grid),
                         np.max(period_grid), len(period_grid))
            logging.info('================================================')
            lcs, wl = self.__detrend(sherlock_target, time, flux, lc_build.star_info)
            lcs = np.concatenate(([flux], lcs), axis=0)
            wl = np.concatenate(([0], wl), axis=0)
            while not self.explore and best_signal_score == 1 and id_run <= sherlock_target.max_runs:
                self.__setup_object_logging(sherlock_id, False)
                object_report = {}
                logging.info("________________________________ run %s________________________________", id_run)
                transit_results, signal_selection = \
                    self.__analyse(sherlock_target, time, lcs, flux_err, lc_build.star_info, id_run,
                                   lc_build.transits_min_count, lc_build.cadence, self.report[sherlock_id], wl,
                                   period_grid, lc_build.detrend_period)
                best_signal_score = signal_selection.score
                object_report["Object Id"] = mission_id
                object_report["run"] = id_run
                object_report["score"] = best_signal_score
                object_report["curve"] = str(signal_selection.curve_index)
                object_report["snr"] = transit_results[signal_selection.curve_index].snr
                object_report["sde"] = transit_results[signal_selection.curve_index].sde
                object_report["fap"] = transit_results[signal_selection.curve_index].fap
                object_report["border_score"] = transit_results[signal_selection.curve_index].border_score
                object_report["harmonic"] = transit_results[signal_selection.curve_index].harmonic
                object_report["period"] = transit_results[signal_selection.curve_index].period
                object_report["per_err"] = transit_results[signal_selection.curve_index].per_err
                object_report["duration"] = transit_results[signal_selection.curve_index].duration * 60 * 24
                object_report["t0"] = transit_results[signal_selection.curve_index].t0
                object_report["depth"] = transit_results[signal_selection.curve_index].depth
                object_report['rp_rs'] = transit_results[signal_selection.curve_index].results.rp_rs
                real_transit_args = np.argwhere(~np.isnan(transit_results[signal_selection.curve_index]
                                                          .results.transit_depths))
                object_report["transits"] = np.array(transit_results[signal_selection.curve_index]
                                                          .results.transit_times)[real_transit_args.flatten()]
                object_report["transits"] = ','.join(map(str, object_report["transits"]))
                object_report["sectors"] = ','.join(map(str, lc_build.sectors)) \
                    if lc_build.sectors is not None and len(lc_build.sectors) > 0 else None
                object_report["ffi"] = isinstance(sherlock_target, MissionFfiIdObjectInfo) or \
                                       isinstance(sherlock_target, MissionFfiCoordsObjectInfo)

                object_report["oi"] = self.__find_matching_oi(sherlock_target.object_info, object_report["period"])
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
                object_dir = self.__init_object_dir(sherlock_id)
                logging.info("Listing most promising candidates for ID %s:", sherlock_id)
                logging.info("%-12s%-10s%-10s%-10s%-8s%-8s%-8s%-8s%-10s%-14s%-14s%-12s%-25s%-10s%-18s%-20s", "Detrend no.", "Period",
                             "Per_err", "Duration", "T0", "Depth", "SNR", "SDE", "FAP", "Border_score", "Matching OI", "Harmonic",
                             "Planet radius (R_Earth)", "Rp/Rs", "Semi-major axis", "Habitability Zone")
                if sherlock_id in self.report:
                    candidates_df = pandas.DataFrame(columns=['curve', 'period', 'per_err', 'duration', 't0', 'depth',
                                                              'snr', 'sde', 'fap', 'border_score', 'oi', 'rad_p', 'rp_rs',
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
                        logging.info("%-12s%-10.4f%-10.5f%-10.2f%-8.2f%-8.3f%-8.2f%-8.2f%-10.6f%-14.2f%-14s%-12s%-25.5f%-10.5f%-18.5f%-20s",
                                     report["curve"], report["period"], report["per_err"],
                                     report["duration"], report["t0"], report["depth"], report["snr"], report["sde"],
                                     report["fap"], report["border_score"], report["oi"], report["harmonic"],
                                     report['rad_p'], report['rp_rs'], a, habitability_zone)
                        candidates_df = candidates_df.append(report, ignore_index=True)
                        i = i + 1
                    candidates_df.to_csv(object_dir + "candidates.csv", index=False)
        except InvalidNumberOfSectorsError as e:
            logging.exception(str(e))
            print(e)
            self.__remove_object_dir(sherlock_id)
        except Exception as e:
            logging.exception(str(e))
            print(e)

    def __init_object_dir(self, object_id, clean_dir=False):
        dir = self.results_dir + str(object_id)
        dir = dir.replace(" ", "_")
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
        formatter = logging.Formatter('%(message)s')
        logger = logging.getLogger()
        while len(logger.handlers) > 0:
            logger.handlers.pop()
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def __setup_object_logging(self, object_id, clean_dir=True):
        object_dir = self.__init_object_dir(object_id, clean_dir)
        logger = logging.getLogger()
        while len(logger.handlers) > 1:
            logger.handlers.pop()
        formatter = logging.Formatter('%(message)s')
        handler = logging.FileHandler(object_dir + str(object_id) + "_report.log")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return object_dir

    def __setup_object_report_logging(self, object_id, clean=False):
        object_dir = self.__setup_object_logging(object_id, False)
        logger = logging.getLogger()
        logger.handlers.pop()
        formatter = logging.Formatter('%(message)s')
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
        logging.info('Sigma threshold for outliers clipping: %.1f', object_info.outliers_sigma)
        logging.info("Fit method: %s", sherlock_target.fit_method)
        if sherlock_target.oversampling is not None:
            logging.info('Oversampling: %.3f', sherlock_target.oversampling)
        if sherlock_target.duration_grid_step is not None:
            logging.info('Duration step: %.3f', sherlock_target.duration_grid_step)
        if sherlock_target.t0_fit_margin is not None:
            logging.info('T0 Fit Margin: %.3f', sherlock_target.t0_fit_margin)
        logging.info('Signal scoring algorithm: %s', sherlock_target.best_signal_algorithm)
        if sherlock_target.best_signal_algorithm == self.VALID_SIGNAL_SELECTORS[2]:
            logging.info('Quorum algorithm vote strength: %.2f',
                         sherlock_target.signal_score_selectors[self.VALID_SIGNAL_SELECTORS[2]].strength)
        mission, mission_prefix, object_num = self.lcbuilder.parse_object_info(object_info.mission_id())
        if object_info.reduce_simple_oscillations and \
                object_info.oscillation_max_period < object_info.oscillation_min_period:
            logging.info("Stellar oscillation period has been set to empty. Defaulting to 1/3 the minimum search period")
            object_info.oscillation_max_period = sherlock_target.period_min / 3
        lc_build = self.lcbuilder.build(object_info, object_dir)
        logging.info('Minimum number of transits: %s', lc_build.transits_min_count)
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
            if sherlock_target.detrend_method == 'gp':
                self.wl_min[sherlock_id] = sherlock_target.detrend_l_min
                self.wl_max[sherlock_id] = sherlock_target.detrend_l_max
            else:
                self.wl_min[sherlock_id] = sherlock_target.detrend_l_min
                self.wl_max[sherlock_id] = sherlock_target.detrend_l_max
        else:
            logging.info("Using transit duration based WL or fixed KS")
            transit_duration = wotan.t14(R_s=lc_build.star_info.radius, M_s=lc_build.star_info.mass,
                                         P=sherlock_target.period_protect, small_planet=True)
            if sherlock_target.detrend_method == 'gp':
                self.wl_min[sherlock_id] = 1
                self.wl_max[sherlock_id] = 12
            else:
                self.wl_min[sherlock_id] = 3 * transit_duration  # minimum transit duration
                self.wl_max[sherlock_id] = 20 * transit_duration  # maximum transit duration
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
            Vetter.vetting_field_of_view(fov_dir, mission, object_num, lc_build.cadence, lc_build.star_info.ra,
                                         lc_build.star_info.dec, lc_build.sectors if isinstance(lc_build.sectors, list)
                                         else lc_build.sectors.tolist(), lc_build.tpf_source, lc_build.tpf_apertures)
        return lc_build

    def __analyse(self, sherlock_target, time, lcs, flux_err, star_info, id_run, transits_min_count, cadence, report,
                  wl, period_grid, detrend_source_period):
        logging.info('=================================')
        logging.info('SEARCH OF SIGNALS - Run %s', id_run)
        logging.info('=================================')
        transit_results = self.__identify_signals(sherlock_target, time, lcs, flux_err, star_info, transits_min_count,
                                                  wl, id_run, cadence, report, period_grid, detrend_source_period)
        signal_selection = sherlock_target.signal_score_selectors[sherlock_target.best_signal_algorithm]\
            .select(transit_results, sherlock_target.snr_min, sherlock_target.detrend_method, wl)
        logging.info(signal_selection.get_message())
        return transit_results, signal_selection

    def __detrend(self, sherlock_target, time, lc, star_info):
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
        figsize = (8, 8)  # x,y
        rows = self.detrend_plot_axis[detrends_number - 1][0]
        cols = self.detrend_plot_axis[detrends_number - 1][1]
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
            if detrends_number > 1:
                plot_axs = axs[i]
            if detrend_method == 'gp':
                plot_axs.set_title('ks=%s' % str(np.around(flatten_wl, decimals=4)))
            else:
                plot_axs.set_title('ws=%s d' % str(np.around(flatten_wl, decimals=4)))
            plot_axs.plot(time_partial[1:], original_lc_partial[1:], linewidth=0.05, color='black', alpha=0.75,
                          rasterized=True)
            plot_axs.plot(time_partial[1:], lc_trend_partial[1:], linewidth=1, color='orange', alpha=1.0)
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
        logging.info("%-12s%-12s%-10s%-8s%-18s%-14s%-14s%-12s%-12s%-14s%-16s%-14s%-12s%-25s%-10s%-18s%-20s",
                     detrend_logging_customs, "Period", "Per_err", "N.Tran", "Mean Depth (ppt)", "T. dur (min)", "T0",
                     "SNR", "SDE", "FAP", "Border_score", "Matching OI", "Harmonic", "Planet radius (R_Earth)", "Rp/Rs",
                     "Semi-major axis", "Habitability Zone")
        transit_results = {}
        run_dir = self.__init_object_run_dir(object_info.sherlock_id(), id_run)
        lc_df = pandas.DataFrame(columns=['#time', 'flux', 'flux_err'])
        args = np.argwhere(~np.isnan(lcs[0])).flatten()
        lc_df['#time'] = time[args]
        lc_df['flux'] = lcs[0][args]
        lc_df['flux_err'] = flux_err[args]
        lc_df.to_csv(run_dir + "/lc_0.csv", index=False)
        transit_result = self.__adjust_transit(sherlock_target, time, lcs[0], star_info, transits_min_count,
                                               transit_results, report, cadence, period_grid, detrend_source_period)
        transit_results[0] = transit_result
        r_planet = self.__calculate_planet_radius(star_info, transit_result.depth)
        rp_rs = transit_result.results.rp_rs
        a, habitability_zone = self.habitability_calculator \
            .calculate_hz_score(star_info.teff, star_info.mass, star_info.lum, transit_result.period)
        oi = self.__find_matching_oi(object_info, transit_result.period)
        logging.info('%-12s%-12.5f%-10.6f%-8s%-18.3f%-14.1f%-14.4f%-12.3f%-12.3f%-14s%-16.2f%-14s%-12s%-25.5f%-10.5f%-18.5f%-20s',
                     "PDCSAP_FLUX", transit_result.period,
                     transit_result.per_err, transit_result.count, transit_result.depth,
                     transit_result.duration * 24 * 60, transit_result.t0, transit_result.snr, transit_result.sde,
                     transit_result.fap, transit_result.border_score, oi, transit_result.harmonic, r_planet, rp_rs, a,
                     habitability_zone)
        plot_title = 'Run ' + str(id_run) + 'PDCSAP_FLUX # P=' + \
                     format(transit_result.period, '.2f') + 'd # T0=' + format(transit_result.t0, '.2f') + \
                     ' # Depth=' + format(transit_result.depth, '.4f') + 'ppt # Dur=' + \
                     format(transit_result.duration * 24 * 60, '.0f') + 'm # SNR:' + \
                     str(format(transit_result.snr, '.2f')) + ' # SDE:' + str(format(transit_result.sde, '.2f')) + \
                     ' # FAP:' + format(transit_result.fap, '.6f')
        plot_file = 'Run_' + str(id_run) + '_PDCSAP-FLUX_' + str(star_info.object_id) + '.png'
        self.__save_transit_plot(star_info.object_id, plot_title, plot_file, time, lcs[0], transit_result, cadence,
                                 id_run)
        for i in range(1, len(wl)):
            lc_df = pandas.DataFrame(columns=['#time', 'flux', 'flux_err'])
            args = np.argwhere(~np.isnan(lcs[i])).flatten()
            lc_df['#time'] = time[args]
            lc_df['flux'] = lcs[i][args]
            lc_df['flux_err'] = flux_err[args]
            lc_df.to_csv(run_dir + "/lc_" + str(i) + ".csv", index=False)
            transit_result = self.__adjust_transit(sherlock_target, time, lcs[i], star_info, transits_min_count,
                                                   transit_results, report, cadence, period_grid, detrend_source_period)
            transit_results[i] = transit_result
            r_planet = self.__calculate_planet_radius(star_info, transit_result.depth)
            rp_rs = transit_result.results.rp_rs
            a, habitability_zone = self.habitability_calculator \
                .calculate_hz_score(star_info.teff, star_info.mass, star_info.lum, transit_result.period)
            oi = self.__find_matching_oi(object_info, transit_result.period)
            logging.info('%-12.4f%-12.5f%-10.6f%-8s%-18.3f%-14.1f%-14.4f%-12.3f%-12.3f%-14s%-16.2f%-14s%-12s%-25.5f%-10.5f%-18.5f%-20s',
                         wl[i], transit_result.period,
                     transit_result.per_err, transit_result.count, transit_result.depth,
                     transit_result.duration * 24 * 60, transit_result.t0, transit_result.snr, transit_result.sde,
                     transit_result.fap, transit_result.border_score, oi, transit_result.harmonic, r_planet, rp_rs, a,
                     habitability_zone)
            detrend_file_title_customs = 'ker_size' if sherlock_target.detrend_method == 'gp' else 'win_size'
            detrend_file_name_customs = 'ks' if sherlock_target.detrend_method == 'gp' else 'ws'
            title = 'Run ' + str(id_run) + '# ' + detrend_file_title_customs + ':' + str(format(wl[i], '.4f')) + \
                    ' # P=' + format(transit_result.period, '.2f') + 'd # T0=' + \
                    format(transit_result.t0, '.2f') + ' # Depth=' + format(transit_result.depth, '.4f') + "ppt # Dur=" + \
                    format(transit_result.duration * 24 * 60, '.0f') + 'm # SNR:' + \
                    str(format(transit_result.snr, '.2f')) + ' # SDE:' + str(format(transit_result.sde, '.2f')) + \
                    ' # FAP:' + format(transit_result.fap, '.6f')
            file = 'Run_' + str(id_run) + '_' + detrend_file_name_customs + '=' + str(format(wl[i], '.4f')) + '_' + \
                   str(star_info.object_id) + '.png'
            self.__save_transit_plot(star_info.object_id, title, file, time, lcs[i], transit_result, cadence, id_run)
        return transit_results

    def __find_matching_oi(self, object_info, period):
        if self.ois is not None:
            existing_period_in_object = self.ois[(self.ois["Object Id"] == object_info.mission_id()) &
                                                 (0.95 < self.ois["Period (days)"] / period) &
                                                 (self.ois["Period (days)"] / period < 1.05)]
            existing_period_in_oi = existing_period_in_object[existing_period_in_object["OI"].notnull()]
            oi = existing_period_in_oi["OI"].iloc[0] if len(
                existing_period_in_oi.index) > 0 else np.nan
        else:
            oi = ""
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
        if results.T0 != 0:
            depths = results.transit_depths[~np.isnan(results.transit_depths)]
            depth = (1. - np.mean(depths)) * 100 / 0.1  # change to ppt units
        else:
            depths = results.transit_depths
            depth = results.transit_depths
        in_transit = tls.transit_mask(time, results.period, results.duration, results.T0)
        transit_count = results.distinct_transit_count
        border_score = self.__compute_border_score(time, results, in_transit, cadence)
        # Recalculating duration because of tls issue https://github.com/hippke/tls/issues/83
        intransit_folded_model = np.where(results['model_folded_model'] < 1.)[0]
        if len(intransit_folded_model) > 0:
            duration = results['period'] * (results['model_folded_phase'][intransit_folded_model[-1]]
                                            - results['model_folded_phase'][intransit_folded_model[0]])
        else:
            duration = results['duration']
        harmonic = self.__is_harmonic(results, run_results, report, detrend_source_period)
        return TransitResult(results, results.period, results.period_uncertainty, duration,
                             results.T0, depths, depth, transit_count, results.snr,
                             results.SDE, results.FAP, border_score, in_transit, harmonic)

    def __calculate_planet_radius(self, star_info, depth):
        rp = np.nan
        try:
            rp = star_info.radius * math.sqrt(depth / 1000) / 0.0091577
        except ValueError as e:
            logging.error("Planet radius could not be calculated: depth=%s, star_radius=%s", depth, star_info.radius)
            logging.exception(str(e))
            print(e)
        return rp

    def __compute_border_score(self, time, result, intransit, cadence):
        shift_cadences = 3600 / cadence
        edge_limit_days = 0.05
        transit_depths = np.nan_to_num(result.transit_depths)
        transit_depths = np.zeros(1) if type(transit_depths) is not np.ndarray else transit_depths
        transit_depths = transit_depths[transit_depths > 0] if len(transit_depths) > 0 else []
        # a=a[np.where([i for i, j in groupby(intransit)])]
        border_score = 0
        if len(transit_depths) > 0:
            shifted_transit_points = shift(intransit, shift_cadences, cval=np.nan)
            inverse_shifted_transit_points = shift(intransit, -shift_cadences, cval=np.nan)
            intransit_shifted = intransit | shifted_transit_points | inverse_shifted_transit_points
            time_edge_indexes = np.where(abs(time[:-1] - time[1:]) > edge_limit_days)[0]
            time_edge = np.full(len(time), False)
            time_edge[time_edge_indexes] = True
            time_edge[0] = True
            time_edge[len(time_edge) - 1] = True
            transits_in_edge = intransit_shifted & time_edge
            transits_in_edge_count = len(transits_in_edge[transits_in_edge])
            border_score = 1 - transits_in_edge_count / len(transit_depths)
        return border_score if border_score >= 0 else 0

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
        period_scales = [round(tls_results.period / run_results[key].period, 2) for key in run_results]
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

    def __save_transit_plot(self, object_id, title, file, time, lc, transit_result, cadence, run_no):
        # start the plotting
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), constrained_layout=True)
        fig.suptitle(title)
        # 1-Plot all the transits
        in_transit = transit_result.in_transit
        tls_results = transit_result.results
        ax1.scatter(time[in_transit], lc[in_transit], color='red', s=2, zorder=0)
        ax1.scatter(time[~in_transit], lc[~in_transit], color='black', alpha=0.05, s=2, zorder=0)
        ax1.plot(tls_results.model_lightcurve_time, tls_results.model_lightcurve_model, alpha=1, color='red', zorder=1)
        # plt.scatter(time_n, flux_new_n, color='orange', alpha=0.3, s=20, zorder=3)
        plt.xlim(time.min(), time.max())
        # plt.xlim(1362.0,1364.0)
        ax1.set(xlabel='Time (days)', ylabel='Relative flux')
        # phase folded plus binning
        bins_per_transit = 8
        half_duration_phase = transit_result.duration / 2 / transit_result.period
        if np.isnan(transit_result.period) or np.isnan(transit_result.duration):
            bins = 200
            folded_plot_range = 0.05
        else:
            bins = transit_result.period / transit_result.duration * bins_per_transit
            folded_plot_range = half_duration_phase * 10
        binning_enabled = round(cadence) <= 300
        ax2.plot(tls_results.model_folded_phase, tls_results.model_folded_model, color='red')
        scatter_measurements_alpha = 0.05 if binning_enabled else 0.8
        ax2.scatter(tls_results.folded_phase, tls_results.folded_y, color='black', s=10,
                    alpha=scatter_measurements_alpha, zorder=2)
        ax2.set_xlim(0.5 - folded_plot_range, 0.5 + folded_plot_range)
        ax2.set(xlabel='Phase', ylabel='Relative flux')
        plt.ticklabel_format(useOffset=False)
        bins = 80
        if binning_enabled and tls_results.SDE != 0:
            folded_phase_zoom_mask = np.where((tls_results.folded_phase > 0.5 - folded_plot_range) &
                                              (tls_results.folded_phase < 0.5 + folded_plot_range))
            folded_phase = tls_results.folded_phase[folded_phase_zoom_mask]
            folded_y = tls_results.folded_y[folded_phase_zoom_mask]
            bin_means, bin_edges, binnumber = stats.binned_statistic(folded_phase, folded_y, statistic='mean',
                                                                     bins=bins)
            bin_stds, _, _ = stats.binned_statistic(folded_phase, folded_y, statistic='std', bins=bins)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            bin_size = int(folded_plot_range * 2 / bins * transit_result.period * 24 * 60)
            ax2.errorbar(bin_centers, bin_means, yerr=bin_stds / 2, xerr=bin_width / 2, marker='o', markersize=4,
                         color='darkorange', alpha=1, linestyle='none', label='Bin size: ' + str(bin_size) + "m")
            ax2.legend(loc="upper right")
        ax3 = plt.gca()
        ax3.axvline(transit_result.period, alpha=0.4, lw=3)
        plt.xlim(np.min(tls_results.periods), np.max(tls_results.periods))
        for n in range(2, 10):
            ax3.axvline(n * tls_results.period, alpha=0.4, lw=1, linestyle="dashed")
            ax3.axvline(tls_results.period / n, alpha=0.4, lw=1, linestyle="dashed")
        ax3.set(xlabel='Period (days)', ylabel='SDE')
        ax3.plot(tls_results.periods, tls_results.power, color='black', lw=0.5)
        ax3.set_xlim(0., max(tls_results.periods))
        plot_dir = self.__init_object_run_dir(object_id, run_no)
        plt.savefig(plot_dir + file, bbox_inches='tight', dpi=200)
        fig.clf()
        plt.close(fig)

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
                clean_time, clean_flux = tls.cleaned_array(time, flux)
                clean_lcs.append(clean_flux)
        return clean_time, np.array(clean_lcs)

    def run_multiprocessing(self, n_processors, func, func_input):
        with Pool(processes=n_processors) as pool:
            return pool.map(func, func_input)

    class KoiInput:
        def __init__(self, star_id, kic_id):
            self.star_id = star_id
            self.kic_id = kic_id
