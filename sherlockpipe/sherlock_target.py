import multiprocessing

from lcbuilder.objectinfo.ObjectInfo import ObjectInfo

from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector
from sherlockpipe.scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector
from sherlockpipe.scoring.SnrBorderCorrectedSignalSelector import SnrBorderCorrectedSignalSelector
from sherlockpipe.search_zones.HabitableSearchZone import HabitableSearchZone
from sherlockpipe.search_zones.OptimisticHabitableSearchZone import OptimisticHabitableSearchZone


class SherlockTarget:
    MASK_MODES = ['mask', 'subtract']
    VALID_SIGNAL_SELECTORS = ["basic", "border-correct", "quorum"]

    def __init__(self, object_info, high_rms_enabled=True, high_rms_threshold=2.5,
                 high_rms_bin_hours=4, smooth_enabled=False,
                 auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                 auto_detrend_period=None,
                 detrend_method='biweight', detrend_l_min=None, detrend_l_max=None, detrends_number=10,
                 detrend_cores=multiprocessing.cpu_count() - 1,
                 prepare_algorithm=None, custom_selection_algorithm=None, custom_transit_template=None,
                 search_zone=None, custom_search_zone=None,
                 snr_min=5, sde_min=5,
                 min_sectors=0, max_sectors=99999,
                 bin_minutes=10,
                 mask_mode='mask',
                 cpu_cores=multiprocessing.cpu_count() - 1, max_runs=10, period_min=0.5,
                 period_max=33, period_protect=10, best_signal_algorithm='border-correct', quorum_strength=1,
                 min_quorum=0, fit_method='tls', oversampling=None,
                 t0_fit_margin=0.05, duration_grid_step=1.1,
                 outliers_sigma=3):
        self.min_sectors = min_sectors
        self.max_sectors = max_sectors
        self.bin_minutes = bin_minutes
        self.mask_mode = mask_mode
        self.cpu_cores = cpu_cores
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
        self.prepare_algorithm = prepare_algorithm
        self.detrend_cores = detrend_cores
        self.detrends_number = detrends_number
        self.detrend_l_max = detrend_l_max
        self.detrend_l_min = detrend_l_min
        self.detrend_method = detrend_method
        self.auto_detrend_period = auto_detrend_period
        self.auto_detrend_ratio = auto_detrend_ratio
        self.auto_detrend_method = auto_detrend_method
        self.auto_detrend_enabled = auto_detrend_enabled
        self.smooth_enabled = smooth_enabled
        self.high_rms_bin_hours = high_rms_bin_hours
        self.high_rms_threshold = high_rms_threshold
        self.high_rms_enabled = high_rms_enabled
        self.outliers_sigma = outliers_sigma
        if mask_mode not in self.MASK_MODES:
            raise ValueError("Provided mask mode '" + mask_mode + "' is not allowed.")
        if best_signal_algorithm not in self.VALID_SIGNAL_SELECTORS:
            raise ValueError("Provided best signal algorithm '" + best_signal_algorithm + "' is not allowed.")
        self.search_zones_resolvers = {'hz': HabitableSearchZone(),
                                       'ohz': OptimisticHabitableSearchZone()}
        self.search_zone = search_zone if custom_search_zone is None else "user"
        if custom_search_zone is not None:
            self.search_zones_resolvers["user"] = custom_search_zone
        self.signal_score_selectors = {self.VALID_SIGNAL_SELECTORS[0]: BasicSignalSelector(),
                                       self.VALID_SIGNAL_SELECTORS[1]: SnrBorderCorrectedSignalSelector(),
                                       self.VALID_SIGNAL_SELECTORS[2]: QuorumSnrBorderCorrectedSignalSelector(
                                           quorum_strength, min_quorum),
                                       "user": custom_selection_algorithm}
        self.best_signal_algorithm = best_signal_algorithm if custom_selection_algorithm is None else "user"
        self.fit_method = "default"
        if fit_method is not None and fit_method.lower() == 'bls':
            self.fit_method = "box"
        elif fit_method is not None and fit_method.lower() == 'grazing':
            self.fit_method = "grazing"
        elif fit_method is not None and fit_method.lower() == 'comet':
            self.fit_method = "comet"
        self.oversampling = oversampling
        if self.oversampling is not None:
            self.oversampling = int(self.oversampling)
        if custom_transit_template is not None:
            self.fit_method = "custom"
            self.user_transit_template = custom_transit_template
        self.object_info = object_info