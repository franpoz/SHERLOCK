import numpy as np

from foldedleastsquares.stats import spectra, period_uncertainty
from sherlockpipe.scoring.SignalSelector import SignalSelector
from sherlockpipe.scoring.helper import harmonic_spectrum
from sherlockpipe.search.transitresult import TransitResult


class AverageSpectrumSignalSelector(SignalSelector):
    """
    Selects the signal with best SNR by summing all the residuals from each detrend search and then computing the SDE.
    """
    def __init__(self):
        """Initialize the average-spectrum signal selector."""
        super().__init__()

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        """Select the best signal by averaging the residuals of all detrended searches.

        The chi2 residuals from each detrended curve are averaged into a single
        spectrum, from which the SDE and maximum power period are computed.

        Parameters
        ----------
        id_run : int
            The current SHERLOCK run number.
        sherlock_target : SherlockTarget
            The target configuration.
        star_info : StarInfo
            The star information.
        transits_min_count : int
            The minimum number of transits required.
        time : ndarray
            The time array of the light curve.
        lcs : dict
            Dictionary of detrended light curve flux arrays.
        transit_results : dict
            Dictionary mapping curve indices to TransitResult objects.
        wl : float
            The window length used for detrending.
        cadence : float
            The cadence of the observations.

        Returns
        -------
        AvgSpectrumSignalSelection
            The selection result with the averaged spectrum information.
        """
        non_nan_result_args = np.argwhere(np.array([1 if result.results is not None else 0 for result in transit_results.values()]) == 1).flatten()
        non_nan_results_count = len(non_nan_result_args)
        if non_nan_results_count == 0:
            return AvgSpectrumSignalSelection(0, 0, TransitResult(None, None, 0, 0, 0, 0, [], [], 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, [], False))
        chi2_sum = np.nansum([result.results.chi2 if result.results is not None else np.zeros(len(transit_results[non_nan_result_args[0]].results.chi2)) for result in transit_results.values()], axis=0) / non_nan_results_count
        if transit_results[len(transit_results) - 1].mode == 'tls':
            SR, power_raw, power, SDE_raw, SDE = spectra(chi2_sum, sherlock_target.oversampling, len(time))
        elif transit_results[len(transit_results) - 1].mode == 'bls':
            power = chi2_sum / np.nanmedian(chi2_sum)
            SDE = np.nanmax(power)
        index_highest_power = np.nanargmax(power)
        signals_powers_for_period = [result.results.power[index_highest_power] if result.results is not None else np.nan for result in
                                     transit_results.values()]
        best_curve_for_signal = np.nanargmax(signals_powers_for_period)
        period = transit_results[non_nan_result_args[0]].results.periods[index_highest_power]
        period_grid = transit_results[non_nan_result_args[0]].results.periods
        if sherlock_target.fit_method in sherlock_target.searchers:
            searcher = sherlock_target.searchers[sherlock_target.fit_method]
        else:
            searcher = sherlock_target.searchers['default']
        result: TransitResult = searcher.search(sherlock_target, time, lcs[best_curve_for_signal], star_info, transits_min_count,
                                 None, None, cadence, [period, period + 1e-6], None)
        result.results.periods = transit_results[best_curve_for_signal].results.periods
        result.results.period_uncertainty = period_uncertainty(period_grid, power)
        result.results.power = power
        result.sde = SDE
        result.results.SDE = SDE
        harmonic_power = harmonic_spectrum(result.results.periods, power)
        result.harmonic_spectrum = harmonic_power
        if result.snr > sherlock_target.snr_min and result.sde > sherlock_target.sde_min:  # and SDE[a] > SDE_min and FAP[a] < FAP_max):
            best_signal_score = 1
        else:
            best_signal_score = 0
        return AvgSpectrumSignalSelection(best_signal_score, best_curve_for_signal, result)


class AvgSpectrumSignalSelection:
    """Container for the result of an average-spectrum signal selection."""

    def __init__(self, score, curve_index, transit_result):
        """Initialize the average-spectrum signal selection.

        Parameters
        ----------
        score : float
            The selection score.
        curve_index : int
            The index of the detrended curve with the best signal.
        transit_result : TransitResult
            The transit result object for the selected signal.
        """
        self.score = score
        self.curve_index = curve_index
        self.transit_result = transit_result

    def get_message(self):
        """Return a human-readable summary of the average-spectrum signal selection.

        Returns
        -------
        str
            A string describing the chosen signal with its name, period,
            SNR, SDE, FAP, and border score.
        """
        curve_name = "PDCSAP_FLUX" if self.curve_index == 0 else str(self.curve_index - 1)
        return "Chosen signal with AVERAGE-SPECTRUM algorithm --> NAME: " + curve_name + \
               "\tPeriod:" + str(self.transit_result.period) + \
               "\tSNR: " + str(self.transit_result.snr) + \
               "\tSDE: " + str(self.transit_result.sde) + \
               "\tFAP: " + str(self.transit_result.fap) + \
               "\tBORDER_SCORE: " + str(self.transit_result.border_score)
