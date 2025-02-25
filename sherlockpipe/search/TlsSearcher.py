from lcbuilder.star.starinfo import StarInfo

from sherlockpipe.scoring.helper import compute_border_score, harmonic_spectrum
from sherlockpipe.search.Searcher import Searcher
import numpy as np
import foldedleastsquares as tls

from sherlockpipe.search.transitresult import TransitResult


class TlsSearcher(Searcher):
    def search(self, sherlock_target, time, lc, star_info: StarInfo, transits_min_count: int,
               run_results, report, cadence, period_grid, detrend_source_period):
        model = tls.transitleastsquares(time, lc)
        power_args = {"transit_template": sherlock_target.fit_method, "period_min": sherlock_target.period_min,
                      "period_max": sherlock_target.period_max, "n_transits_min": transits_min_count,
                      "T0_fit_margin": sherlock_target.t0_fit_margin, "show_progress_bar": False,
                      "use_threads": sherlock_target.cpu_cores, "oversampling_factor": sherlock_target.oversampling,
                      "duration_grid_step": sherlock_target.duration_grid_step,
                      "period_grid": np.array(period_grid)}
        if star_info.ld_coefficients is not None:
            power_args["u"] = star_info.ld_coefficients
        power_args["R_star"] = star_info.radius
        power_args["R_star_min"] = star_info.radius_min
        power_args["R_star_max"] = star_info.radius_max
        power_args["M_star"] = star_info.mass
        power_args["M_star_min"] = star_info.mass_min
        power_args["M_star_max"] = star_info.mass_max
        power_args[
            'use_gpu'] = sherlock_target.search_engine == 'gpu' or sherlock_target.search_engine == 'gpu_approximate'
        power_args['gpu_approximate'] = sherlock_target.search_engine == 'gpu_approximate'
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
        # Recalculating duration because of tls issue https://github.com/hippke/tls/issues/83
        intransit_folded_model = np.where(results['model_folded_model'] < 1.)[0]
        if len(intransit_folded_model) > 1:
            duration = results['period'] * (results['model_folded_phase'][intransit_folded_model[-1]]
                                            - results['model_folded_phase'][intransit_folded_model[0]])
        else:
            duration = results['duration']
        border_score = compute_border_score(time, results, in_transit, cadence)
        harmonic = None
        if run_results is not None:
            harmonic = self._is_harmonic(results, run_results, report, detrend_source_period)
        harmonic_power = harmonic_spectrum(results.periods, results.power)
        return TransitResult(power_args, results, results.period, results.period_uncertainty, duration,
                             results.T0, t0s, depths, depths_err, depth, depth_err, results.odd_even_mismatch,
                             (1 - results.depth_mean_even[0]) * 1000, (1 - results.depth_mean_odd[0]) * 1000,
                             transit_count,
                             results.snr, results.SDE, results.FAP, border_score, in_transit, harmonic,
                             harmonic_power, mode='tls')
