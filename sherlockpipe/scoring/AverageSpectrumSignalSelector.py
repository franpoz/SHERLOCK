import numpy as np

from foldedleastsquares.stats import spectra
from sherlockpipe.scoring.SignalSelector import SignalSelector
from sherlockpipe.scoring.helper import compute_border_score, harmonic_spectrum
from sherlockpipe.search.transitresult import TransitResult
import foldedleastsquares as tls


class AverageSpectrumSignalSelector(SignalSelector):
    """
    Selects the signal with best SNR by summing all the residuals from each detrend search and then computing the SDE.
    """
    def __init__(self):
        super().__init__()

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        non_nan_result_args = np.argwhere(np.array([1 if result.results is not None else 0 for result in transit_results.values()]) == 1).flatten()
        non_nan_results_count = len(non_nan_result_args)
        if non_nan_results_count == 0:
            return AvgSpectrumSignalSelection(0, 0, TransitResult(None, None, 0, 0, 0, 0, [], [], 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, [], False))
        chi2_sum = np.nansum([result.results['chi2'] if result.results is not None else np.zeros(len(transit_results[non_nan_result_args[0]].results.chi2)) for result in transit_results.values()], axis=0) / non_nan_results_count
        SR, power_raw, power, SDE_raw, SDE = spectra(chi2_sum, sherlock_target.oversampling, len(time))
        index_highest_power = np.nanargmax(power)
        signals_powers_for_period = [result.results['power'][index_highest_power] if result.results is not None else np.nan for result in
                                     transit_results.values()]
        best_curve_for_signal = np.nanargmax(signals_powers_for_period)
        period = transit_results[non_nan_result_args[0]].results['periods'][index_highest_power]
        oversampling = sherlock_target.oversampling
        model = tls.transitleastsquares(time, lcs[best_curve_for_signal])
        power_args = {"transit_template": sherlock_target.fit_method, "n_transits_min": transits_min_count,
                      "T0_fit_margin": sherlock_target.t0_fit_margin, "show_progress_bar": False,
                      "use_threads": sherlock_target.cpu_cores, "oversampling_factor": oversampling,
                      "duration_grid_step": sherlock_target.duration_grid_step,
                      "period_grid": np.array([period])}
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
        t0s = np.array(results.transit_times)
        in_transit = tls.transit_mask(time, period, results.duration, results.T0)
        transit_count = results.distinct_transit_count
        border_score = compute_border_score(time, results, in_transit, cadence)
        results.periods = transit_results[best_curve_for_signal].results['periods']
        results.power = power
        harmonic_power = harmonic_spectrum(results.periods, power)
        depth_err = np.sqrt(np.nansum([depth_err ** 2 for depth_err in depths_err])) / len(depths_err)
        result = TransitResult(power_args, results, period, period / 100, results['duration'],
                               results['T0'], t0s, depths,
                               depths_err, results['depth'], depth_err, results['odd_even_mismatch'],
                               (1 - results.depth_mean_even[0]) * 1000, (1 - results.depth_mean_odd[0]) * 1000,
                               transit_count, results.snr, SDE, None, border_score, in_transit, False,
                               harmonic_power)
        if result.snr > sherlock_target.snr_min and result.sde > sherlock_target.sde_min:  # and SDE[a] > SDE_min and FAP[a] < FAP_max):
            best_signal_score = 1
        else:
            best_signal_score = 0
        return AvgSpectrumSignalSelection(best_signal_score, best_curve_for_signal, result)

class AvgSpectrumSignalSelection:
    def __init__(self, score, curve_index, transit_result):
        self.score = score
        self.curve_index = curve_index
        self.transit_result = transit_result

    def get_message(self):
        curve_name = "PDCSAP_FLUX" if self.curve_index == 0 else str(self.curve_index - 1)
        return "Chosen signal with AVERAGE-SPECTRUM algorithm --> NAME: " + curve_name + \
               "\tPeriod:" + str(self.transit_result.period) + \
               "\tSNR: " + str(self.transit_result.snr) + \
               "\tSDE: " + str(self.transit_result.sde) + \
               "\tFAP: " + str(self.transit_result.fap) + \
               "\tBORDER_SCORE: " + str(self.transit_result.border_score)
