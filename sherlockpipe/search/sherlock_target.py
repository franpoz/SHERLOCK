import multiprocessing
import os
from typing import Dict

from lcbuilder.objectinfo.ObjectInfo import ObjectInfo

from sherlockpipe.scoring.AverageSpectrumSignalSelector import AverageSpectrumSignalSelector
from sherlockpipe.scoring.BasicSdeSignalSelector import BasicSdeSignalSelector
from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector
from sherlockpipe.scoring.QuorumSdeBorderCorrectedSignalSelector import QuorumSdeBorderCorrectedSignalSelector
from sherlockpipe.scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector
from sherlockpipe.scoring.SdeBorderCorrectedSignalSelector import SdeBorderCorrectedSignalSelector
from sherlockpipe.scoring.SnrBorderCorrectedSignalSelector import SnrBorderCorrectedSignalSelector
from sherlockpipe.search.BlsSearcher import BlsSearcher
from sherlockpipe.search.Searcher import Searcher
from sherlockpipe.search.TlsSearcher import TlsSearcher
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
                 ois_mask=False, min_transits_count=2, compute_phase_coverage=False, search_engine='cpu'):
        """
        Configuration container for a single transiting planet search target.

        Parameters
        ----------
        object_info : lcbuilder.objectinfo.ObjectInfo.ObjectInfo
            The mission object information (TIC, KIC, EPIC, etc.).
        detrend_method : str
            Detrending method ('biweight' or 'gp').
        detrend_l_min : float or None
            Minimum detrend window length / kernel size in days.
        detrend_l_max : float or None
            Maximum detrend window length / kernel size in days.
        detrends_number : int
            Number of detrend models to apply.
        detrend_cores : int
            Number of CPU cores for detrending.
        custom_selection_algorithm : callable or None
            Custom signal selection algorithm.
        custom_transit_template : callable or None
            Custom transit template generator for TLS.
        search_zone : str or None
            Predefined search zone ('hz', 'ohz', or None).
        custom_search_zone : object or None
            Custom search zone resolver.
        snr_min : float
            Minimum SNR threshold for a signal to be considered.
        sde_min : float
            Minimum SDE threshold for a signal to be considered.
        min_sectors : int
            Minimum number of sectors required.
        max_sectors : int
            Maximum number of sectors allowed.
        bin_minutes : int
            Binning size in minutes for detrend plots.
        mask_mode : str
            Signal masking mode ('mask' or 'subtract').
        cpu_cores : int
            Number of CPU cores for the search and fit.
        max_runs : int
            Maximum number of iterative search runs.
        period_min : float
            Minimum search period in days.
        period_max : float
            Maximum search period in days.
        period_protect : float
            Period protection window in days for detrending.
        best_signal_algorithm : str
            Algorithm used to select the best signal among detrended curves.
        quorum_strength : float
            Voting strength for quorum-based signal selectors.
        min_quorum : int
            Minimum votes for quorum-based signal selectors.
        fit_method : str
            Transit fit method ('tls', 'bls', 'bls-periodogram', 'grazing', 'tailed', etc.).
        oversampling : float
            Oversampling factor for the period grid.
        t0_fit_margin : float
            Margin in days for T0 fitting.
        duration_grid_step : float
            Step size for the duration grid.
        source_properties_file : str or None
            Path to the source YAML properties file.
        cache_dir : str
            Directory for caching downloaded data.
        ignore_original : bool
            If True, skip searching the original (undetrended) light curve.
        pickle_mode : str
            Pickle output mode ('none', 'all', or 'selected').
        use_harmonics_spectra : bool
            If True, compute harmonic spectra for detected signals.
        ois_mask : bool
            If True, mask known Objects of Interest in the light curve.
        min_transits_count : int
            Minimum number of transits required for a valid signal.
        compute_phase_coverage : bool
            If True, compute the phase coverage over the period grid.
        search_engine : str
            Search engine backend ('cpu', 'gpu', or 'gpu_approximate').
        """
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
        self.searchers: Dict[str, Searcher] = {'bls-periodogram': BlsSearcher(), 'default': TlsSearcher()}
        self.best_signal_algorithm = best_signal_algorithm if custom_selection_algorithm is None else "user"
        self.fit_method = "default"
        if fit_method is not None and fit_method.lower() == 'bls':
            self.fit_method = "box"
        elif fit_method is not None and fit_method.lower() == 'grazing':
            self.fit_method = "grazing"
        elif fit_method is not None and fit_method.lower() == 'tailed':
            self.fit_method = "tailed"
        elif fit_method is not None and fit_method.lower() == 'bls-periodogram':
            self.fit_method = 'bls-periodogram'
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
        self.compute_phase_coverage = compute_phase_coverage
        self.search_engine = search_engine
