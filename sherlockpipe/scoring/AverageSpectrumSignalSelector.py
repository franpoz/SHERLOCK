import numpy as np

from foldedleastsquares.stats import spectra
from sherlockpipe.scoring.SignalSelector import SignalSelector, SignalSelection
from sherlockpipe.scoring.helper import compute_border_score
from sherlockpipe.transitresult import TransitResult
import foldedleastsquares as tls


class AverageSpectrumSignalSelector(SignalSelector):
    """
    Selects the signal with best SNR
    """
    def __init__(self):
        super().__init__()

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        chi2_sum = np.sum([result.results['chi2'] for result in transit_results.values()], axis=0) / len(
            transit_results)
        SR, power_raw, power, SDE_raw, SDE = spectra(chi2_sum, sherlock_target.oversampling, len(time))
        index_highest_power = np.argmax(power)
        period = transit_results[0].results['periods'][index_highest_power]
        oversampling = sherlock_target.oversampling
        model = tls.transitleastsquares(time, lcs[0])
        power_args = {"transit_template": sherlock_target.fit_method, "n_transits_min": transits_min_count,
                      "T0_fit_margin": sherlock_target.t0_fit_margin, "show_progress_bar": False,
                      "use_threads": 1, "oversampling_factor": oversampling,
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
        results.periods = transit_results[0].results['periods']
        results.power = power
        result = TransitResult(power_args, results, period, period / 100, results['duration'],
                               results['T0'], t0s, depths,
                               depths_err, results['depth'], results['odd_even_mismatch'],
                               (1 - results.depth_mean_even[0]) * 1000, (1 - results.depth_mean_odd[0]) * 1000,
                               transit_count, results.snr, SDE, None, border_score, in_transit, False)
        if result.snr > sherlock_target.snr_min and result.sde > sherlock_target.sde_min:  # and SDE[a] > SDE_min and FAP[a] < FAP_max):
            best_signal_score = 1
        else:
            best_signal_score = 0
        return SignalSelection(best_signal_score, 0, result)
