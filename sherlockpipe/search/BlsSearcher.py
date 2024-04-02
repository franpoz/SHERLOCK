import lightkurve
from lcbuilder.star.starinfo import StarInfo

from sherlockpipe.scoring.helper import compute_border_score, harmonic_spectrum
from sherlockpipe.search.Searcher import Searcher
import pandas as pd
import numpy as np
import foldedleastsquares as tls

from sherlockpipe.search.transitresult import TransitResult


class BlsSearcher(Searcher):
    def search(self, sherlock_target, time, lc, star_info: StarInfo, transits_min_count: int,
               run_results, report, cadence, period_grid, detrend_source_period):
        bls_results = lightkurve.LightCurve(time=time, flux=lc).to_periodogram(method='bls', period=period_grid)
        max_power_index = np.argmax(bls_results.power)
        results = type('', (object,), {"foo": 1})()
        results.SDE = bls_results.power[max_power_index].value / np.nanmedian(bls_results.power).value
        results.power = bls_results.power
        results.FAP = 0
        results.duration = bls_results.duration_at_max_power.value
        duration = results.duration
        results.period = bls_results.period_at_max_power.value
        results.T0 = bls_results.transit_time_at_max_power.value
        in_transit_double = tls.transit_mask(time, results.period, 2 * results.duration, results.T0)
        in_transit = tls.transit_mask(time, results.period, results.duration, results.T0)
        transit_count = sum(
            [value != in_transit[index + 1] if index < len(in_transit) - 1 else False for index, value in
             enumerate(in_transit)]) // 2
        oot_flux = lc[~in_transit]
        it_flux = lc[in_transit]
        results.snr = np.abs(np.mean(1 - it_flux)) / np.nanstd(oot_flux) * (len(it_flux) ** 0.5)
        depth = bls_results.depth[max_power_index].value * 1000
        depth_err = depth / results.snr
        results.rp_rs = (depth * (star_info.radius ** 2)) ** 2
        results.odd_even_mismatch = 0
        results.periods = bls_results.period.value
        results.period_uncertainty = np.abs(results.period - bls_results.period[max_power_index - 1].value) \
            if max_power_index > 0 else np.abs(results.period - bls_results.period[max_power_index + 1].value)
        t0s = []
        depths = [depth for value in range(0, transit_count)]
        results.transit_depths = depths
        results.depth_mean_even = [depth]
        results.depth_mean_odd = [depth]
        results.odd_even_mismatch = 0
        depths_err = [0 for value in range(0, transit_count)]
        results.model_lightcurve_time = time
        results.model_lightcurve_model = np.ones(len(time))
        results.model_lightcurve_model[in_transit] = 1 - depth / 1000
        lc_model_df = pd.DataFrame(columns=['time', 'flux'])
        lc_model_df['time'] = tls.fold(time, results.period, results.T0 + results.period / 2)
        lc_model_df['flux'] = lc
        lc_model_df.sort_values(by=['time'], ascending=True, inplace=True)
        results.model_folded_phase = lc_model_df['time'].to_numpy()
        intransit_folded_indexes = (
            np.argwhere((results.model_folded_phase >= 0.5 - 0.5 * duration / results.period) &
                        (results.model_folded_phase <= 0.5 + 0.5 * duration / results.period)).flatten())
        results.model_folded_model = np.ones(len(time))
        results.model_folded_model[intransit_folded_indexes] = 1 - depth / 1000
        results.folded_phase = results.model_folded_phase
        results.folded_y = lc_model_df['flux'].to_numpy()
        results.chi2 = bls_results.power.value
        border_score = compute_border_score(time, results, in_transit, cadence)
        harmonic = None
        if run_results is not None:
            harmonic = self._is_harmonic(results, run_results, report, detrend_source_period)
        harmonic_power = harmonic_spectrum(results.periods, results.power)
        return TransitResult(None, results, results.period, results.period_uncertainty, duration,
                             results.T0, t0s, depths, depths_err, depth, depth_err, results.odd_even_mismatch,
                             (1 - results.depth_mean_even[0]) * 1000, (1 - results.depth_mean_odd[0]) * 1000,
                             transit_count,
                             results.snr, results.SDE, results.FAP, border_score, in_transit, harmonic,
                             harmonic_power, mode='bls')