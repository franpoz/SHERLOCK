import multiprocessing
import os

from lcbuilder.objectinfo.ObjectInfo import ObjectInfo

from sherlockpipe.scoring.AverageSpectrumSignalSelector import AverageSpectrumSignalSelector
from sherlockpipe.scoring.BasicSdeSignalSelector import BasicSdeSignalSelector
from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector
from sherlockpipe.scoring.QuorumSdeBorderCorrectedSignalSelector import QuorumSdeBorderCorrectedSignalSelector
from sherlockpipe.scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector
from sherlockpipe.scoring.SdeBorderCorrectedSignalSelector import SdeBorderCorrectedSignalSelector
from sherlockpipe.scoring.SnrBorderCorrectedSignalSelector import SnrBorderCorrectedSignalSelector
from sherlockpipe.search_zones.HabitableSearchZone import HabitableSearchZone
from sherlockpipe.search_zones.OptimisticHabitableSearchZone import OptimisticHabitableSearchZone


class SherlockTarget:
    """
    Used as input for :class:`sherlockpipe.sherlock.Sherlock`:
    """
    MASK_MODES = ['mask', 'subtract']
    VALID_SIGNAL_SELECTORS = ["basic", "border-correct", "quorum", "basic-snr", "border-correct-snr", "quorum-snr",
                              'average-spectrum']

    def __init__(self, object_info,
                 detrend_method='biweight', detrend_l_min=None, detrend_l_max=None, detrends_number=10,
                 detrend_cores=multiprocessing.cpu_count() - 1,
                 custom_selection_algorithm=None, custom_transit_template=None,
                 search_zone=None, custom_search_zone=None,
                 snr_min=5, sde_min=5,
                 min_sectors=0, max_sectors=99999,
                 bin_minutes=10,
                 mask_mode='mask',
                 cpu_cores=multiprocessing.cpu_count() - 1, max_runs=10, period_min=0.5,
                 period_max=33, period_protect=10, best_signal_algorithm='border-correct', quorum_strength=1,
                 min_quorum=0, fit_method='tls', oversampling=10,
                 t0_fit_margin=0.05, duration_grid_step=1.1,
                 source_properties_file=None,
                 cache_dir=os.path.expanduser('~') + "/",
                 ignore_original=False, pickle_mode='none', use_harmonics_spectra=False,
                 ois_mask=False, min_transits_count=2):
        self.min_sectors = min_sectors
        self.max_sectors = max_sectors
        self.bin_minutes = bin_minutes
        self.mask_mode = mask_mode
        self.cpu_cores = cpu_cores if cpu_cores <= os.cpu_count() else os.cpu_count()
        self.max_runs = max_runs
        self.period_min = period_min
        self.period_max = period_max
        self.period_protect = period_protect
        self.best_signal_algorithm = best_signal_algorithm
        self.quorum_strength = quorum_strength
        self.min_quorum = min_quorum
        self.fit_method = fit_method
        self.oversampling = oversampling
        self.t0_fit_margin = t0_fit_margin
        self.duration_grid_step = duration_grid_step
        self.sde_min = sde_min
        self.snr_min = snr_min
        self.custom_search_zone = custom_search_zone
        self.search_zone = search_zone
        self.custom_transit_template = custom_transit_template
        self.custom_selection_algorithm = custom_selection_algorithm
        self.detrend_cores = detrend_cores if detrend_cores <= os.cpu_count() else os.cpu_count()
        self.detrends_number = detrends_number
        self.detrend_l_max = detrend_l_max
        self.detrend_l_min = detrend_l_min
        self.detrend_method = detrend_method
        self.pickle_mode = pickle_mode
        if mask_mode not in self.MASK_MODES:
            raise ValueError("Provided mask mode '" + mask_mode + "' is not allowed.")
        if best_signal_algorithm not in self.VALID_SIGNAL_SELECTORS:
            raise ValueError("Provided best signal algorithm '" + best_signal_algorithm + "' is not allowed.")
        self.search_zones_resolvers = {'hz': HabitableSearchZone(),
                                       'ohz': OptimisticHabitableSearchZone()}
        self.search_zone = search_zone if custom_search_zone is None else "user"
        if custom_search_zone is not None:
            self.search_zones_resolvers["user"] = custom_search_zone
        self.signal_score_selectors = {self.VALID_SIGNAL_SELECTORS[0]: BasicSdeSignalSelector(),
                                       self.VALID_SIGNAL_SELECTORS[1]: SdeBorderCorrectedSignalSelector(),
                                       self.VALID_SIGNAL_SELECTORS[2]: QuorumSdeBorderCorrectedSignalSelector(
                                           quorum_strength, min_quorum),
                                       self.VALID_SIGNAL_SELECTORS[3]: BasicSignalSelector(),
                                       self.VALID_SIGNAL_SELECTORS[4]: SnrBorderCorrectedSignalSelector(),
                                       self.VALID_SIGNAL_SELECTORS[5]: QuorumSnrBorderCorrectedSignalSelector(
                                           quorum_strength, min_quorum),
                                       self.VALID_SIGNAL_SELECTORS[6]: AverageSpectrumSignalSelector(),
                                       "user": custom_selection_algorithm}
        self.best_signal_algorithm = best_signal_algorithm if custom_selection_algorithm is None else "user"
        self.fit_method = "default"
        if fit_method is not None and fit_method.lower() == 'bls':
            self.fit_method = "box"
        elif fit_method is not None and fit_method.lower() == 'grazing':
            self.fit_method = "grazing"
        elif fit_method is not None and fit_method.lower() == 'tailed':
            self.fit_method = "tailed"
        self.oversampling = oversampling
        if self.oversampling is not None:
            self.oversampling = self.oversampling
        if custom_transit_template is not None:
            self.fit_method = "custom"
            self.user_transit_template = custom_transit_template
        self.object_info = object_info
        self.source_properties_file = source_properties_file
        self.cache_dir = cache_dir
        self.ignore_original = ignore_original
        self.use_harmonics_spectra = use_harmonics_spectra
        self.ois_mask = ois_mask
        self.min_transits_count = min_transits_count
