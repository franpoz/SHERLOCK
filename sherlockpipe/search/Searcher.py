from abc import abstractmethod, ABC

from lcbuilder.star.starinfo import StarInfo

import numpy as np

class Searcher(ABC):
    @abstractmethod
    def search(self, sherlock_target, time, lc, star_info: StarInfo, transits_min_count: int,
               run_results, report, cadence, period_grid, detrend_source_period):
        pass

    def _is_harmonic(self, tls_results, run_results, report, detrend_source_period):
        scales = [0.25, 0.5, 1, 2, 4]
        if detrend_source_period is not None:
            rotator_scale = round(tls_results.period / detrend_source_period, 2)
            rotator_harmonic = np.array(np.argwhere(
                (np.array(scales) > rotator_scale - 0.02) & (np.array(scales) < rotator_scale + 0.02))).flatten()
            if len(rotator_harmonic) > 0:
                return str(scales[rotator_harmonic[0]]) + "*source"
        period_scales = [tls_results.period / round(item["period"], 2) for item in report]
        for key, period_scale in enumerate(period_scales):
            period_harmonic = np.array(np.argwhere(
                (np.array(scales) > period_scale - 0.02) & (np.array(scales) < period_scale + 0.02))).flatten()
            if len(period_harmonic) > 0:
                period_harmonic = scales[period_harmonic[0]]
                return str(period_harmonic) + "*SOI" + str(key + 1)
        period_scales = [round(tls_results.period / run_results[key].period, 2) if run_results[key].period > 0 else 0
                         for key in run_results]
        for key, period_scale in enumerate(period_scales):
            period_harmonic = np.array(np.argwhere(
                (np.array(scales) > period_scale - 0.02) & (np.array(scales) < period_scale + 0.02))).flatten()
            if len(period_harmonic) > 0 and period_harmonic[0] != 2:
                period_harmonic = scales[period_harmonic[0]]
                return str(period_harmonic) + "*this(" + str(key) + ")"
        return "-"
