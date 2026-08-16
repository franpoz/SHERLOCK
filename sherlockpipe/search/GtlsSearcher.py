from types import SimpleNamespace

import numpy as np

from lcbuilder.star.starinfo import StarInfo

from sherlockpipe.scoring.helper import compute_border_score, harmonic_spectrum
from sherlockpipe.search.Searcher import Searcher
from sherlockpipe.search.transitresult import TransitResult


class GtlsSearcher(Searcher):
    """GPU TLS searcher backed by the PyPI ``gputls`` package."""

    def search(self, sherlock_target, time, lc, star_info: StarInfo, transits_min_count: int,
               run_results, report, cadence, period_grid, detrend_source_period):
        if sherlock_target.search_engine != "gpu":
            raise ValueError("GTLS is only available for SEARCH_ENGINE='gpu'.")
        if sherlock_target.custom_transit_template is not None:
            raise ValueError("GTLS does not support custom transit templates.")

        try:
            from gputls import gtls
        except ImportError as error:
            raise ImportError(
                "SEARCH_ENGINE='gpu' requires the PyPI package gputls==0.4.4 "
                "and a compatible CuPy/CUDA installation."
            ) from error

        template = self._template(sherlock_target.fit_method)
        periods = np.asarray(period_grid, dtype=float)
        if periods.size == 0:
            raise ValueError("GTLS requires a non-empty period grid.")
        search_periods = self._gtls_period_grid(periods)

        model = gtls(t=np.asarray(time, dtype=float), y=np.asarray(lc, dtype=float), verbose=False)
        power_args = {
            "periods": search_periods,
            "period_min": sherlock_target.period_min,
            "period_max": sherlock_target.period_max,
            "n_transits_min": transits_min_count,
            "T0_fit_margin": sherlock_target.t0_fit_margin,
            "oversampling_factor": sherlock_target.oversampling,
            "duration_grid_step": sherlock_target.duration_grid_step,
            "transit_template": template,
            "R_star": star_info.radius,
            "R_star_min": star_info.radius_min,
            "R_star_max": star_info.radius_max,
            "M_star": star_info.mass,
            "M_star_min": star_info.mass_min,
            "M_star_max": star_info.mass_max,
            "GPUDeviceID": sherlock_target.gpu_device_id,
            "bar_location": 0,
        }
        if star_info.ld_coefficients is not None:
            power_args["u"] = star_info.ld_coefficients

        gtls_results = model.power(**power_args)
        results = self._normalize_results(gtls_results, time, lc, star_info)
        in_transit = self._transit_mask(time, results.period, results.duration, results.T0)
        border_score = compute_border_score(time, results, in_transit, cadence)
        harmonic = None
        if run_results is not None:
            harmonic = self._is_harmonic(results, run_results, report, detrend_source_period)
        harmonic_power = harmonic_spectrum(results.periods, results.power)

        return TransitResult(
            power_args, results, results.period, results.period_uncertainty,
            results.duration, results.T0, results.transit_times,
            results.transit_depths, results.transit_depths_uncertainties,
            (1 - np.nanmean(results.transit_depths)) * 1000,
            np.sqrt(np.nansum(results.transit_depths_uncertainties ** 2)) /
            max(len(results.transit_depths_uncertainties), 1),
            results.odd_even_mismatch,
            (1 - results.depth_mean_even[0]) * 1000,
            (1 - results.depth_mean_odd[0]) * 1000,
            results.distinct_transit_count, results.snr, results.SDE,
            results.FAP, border_score, in_transit, harmonic, harmonic_power,
            mode="tls"
        )

    @staticmethod
    def _template(fit_method):
        if fit_method in ("default", "box", "grazing"):
            return fit_method
        raise ValueError("GTLS supports only the default, box, and grazing templates.")

    @staticmethod
    def _gtls_period_grid(periods):
        """Avoid GTLS 0.4.4's zero batch-size bug for small grids."""
        if len(periods) >= 100:
            return periods
        lower = float(np.min(periods))
        upper = float(np.max(periods))
        if lower == upper:
            upper = lower + max(abs(lower) * 1e-6, 1e-6)
        return np.linspace(lower, upper, 100)

    @staticmethod
    def _transit_mask(time, period, duration, t0):
        return np.abs((time - t0 + 0.5 * period) % period - 0.5 * period) < 0.5 * duration

    def _normalize_results(self, source, time, lc, star_info):
        periods = np.asarray(source.periods, dtype=float)
        power = np.asarray(source.power, dtype=float)
        period = float(np.asarray(source.period).item())
        duration = float(np.asarray(source.duration).item())
        t0 = float(np.asarray(source.T0).item())
        transit_times = np.asarray(source.transitTimes, dtype=float)
        transit_depths, depth_errors = self._transit_depths(time, lc, transit_times, period, duration)
        depth_even = np.nanmean(transit_depths[::2]) if len(transit_depths[::2]) else np.nan
        depth_odd = np.nanmean(transit_depths[1::2]) if len(transit_depths[1::2]) else np.nan
        mismatch = 0.0 if np.isnan(depth_even) or np.isnan(depth_odd) else abs(depth_even - depth_odd)
        model = np.ones(len(time))
        model[self._transit_mask(time, period, duration, t0)] -= float(source.depth)
        phase = ((time - (t0 + period / 2)) / period) % 1
        order = np.argsort(phase)
        model_phase = np.linspace(0, 1, max(len(time), 2))
        model_folded = np.ones(len(model_phase))
        model_folded[np.abs(model_phase - 0.5) < duration / (2 * period)] -= float(source.depth)
        return SimpleNamespace(
            periods=periods, power=power, chi2=np.asarray(source.chi2), period=period,
            period_uncertainty=self._period_uncertainty(periods, power), T0=t0,
            duration=duration, transit_times=transit_times,
            transit_depths=transit_depths, transit_depths_uncertainties=depth_errors,
            depth_mean_even=np.asarray([depth_even]), depth_mean_odd=np.asarray([depth_odd]),
            odd_even_mismatch=mismatch, distinct_transit_count=len(transit_times),
            snr=float(source.snr), SDE=float(source.SDE), FAP=np.nan,
            rp_rs=np.sqrt(max(float(source.depth), 0)),
            model_lightcurve_time=np.asarray(time), model_lightcurve_model=model,
            model_folded_phase=model_phase, model_folded_model=model_folded,
            folded_phase=phase[order], folded_y=np.asarray(lc)[order]
        )

    @staticmethod
    def _period_uncertainty(periods, power):
        index = int(np.nanargmax(power))
        neighbors = []
        if index:
            neighbors.append(abs(periods[index] - periods[index - 1]))
        if index + 1 < len(periods):
            neighbors.append(abs(periods[index + 1] - periods[index]))
        return min(neighbors) if neighbors else np.nan

    def _transit_depths(self, time, lc, transit_times, period, duration):
        depths = []
        errors = []
        for transit_time in transit_times:
            mask = self._transit_mask(time, period, duration, transit_time)
            values = np.asarray(lc)[mask]
            depths.append(np.nanmean(values) if len(values) else np.nan)
            errors.append(np.nanstd(values) / np.sqrt(len(values)) if len(values) else np.nan)
        return np.asarray(depths), np.asarray(errors)
