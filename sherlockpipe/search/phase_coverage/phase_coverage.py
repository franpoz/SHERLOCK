import dataclasses
import logging
from multiprocessing import Pool
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


@dataclasses.dataclass
class PhaseCoverageInput:
    """
    Used as input for the multiprocessing task to compute coverage
    """
    time: List[float]
    period: float


class PhaseCoverage:
    @staticmethod
    def phase_coverage(phase_coverage_input: PhaseCoverageInput):
        if phase_coverage_input.period == 0:
            return 1
        else:
            ph = ((phase_coverage_input.time + 0.5 * phase_coverage_input.period) % phase_coverage_input.period -
                  (0.5 * phase_coverage_input.period))
            sph_in = np.sort(ph)
            sph_out = sph_in[::-1]
            sph_in_diff = np.abs(np.diff(sph_in))
            sph_out_diff = np.abs(np.diff(sph_out))
            df = np.min(np.diff(phase_coverage_input.time))
            spaces_in = np.sort(sph_in[np.hstack([*np.argwhere(sph_in_diff > 4 * df).T, len(sph_in) - 1])])
            spaces_out = np.sort(sph_out[np.hstack([*np.argwhere(sph_out_diff > 4 * df).T, len(sph_in) - 1])])
            return np.sum(spaces_in - spaces_out) / phase_coverage_input.period

    @staticmethod
    def compute_phase_coverage(dir: str, time: List[float], period_grid: List[float], cpus=1):
        logging.info("Computing period grid phase coverage")
        period_grid = np.sort(period_grid)
        phase_coverage_inputs = []
        for period in period_grid:
            phase_coverage_inputs.append(PhaseCoverageInput(time, period))
        with Pool(processes=cpus) as pool:
            phase_coverage = np.array(pool.map(PhaseCoverage.phase_coverage, phase_coverage_inputs))
        fig1, (ax1) = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
        fig1.patch.set_facecolor('xkcd:white'),
        bins = len(period_grid) / 100
        phase_coverage = np.array(phase_coverage) * 100
        bin_means, bin_edges, binnumber = stats.binned_statistic(period_grid, phase_coverage, statistic='mean',
                                                                 bins=bins)
        bin_stds, _, _ = stats.binned_statistic(period_grid, phase_coverage, statistic='std', bins=bins)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2
        ax1.plot(period_grid, phase_coverage, color='teal', linestyle='-', alpha=0.15)
        ax1.errorbar(bin_centers, bin_means, yerr=bin_stds / 2, color='peru', marker='.', alpha=0.99, label='-')
        ax1.set_ylabel("Phase Coverage ($\\%$)", fontsize=12)
        ax1.set_xlabel("Period (days)", fontsize=12)
        plt.grid(True)
        plt.savefig(dir + '/phase_coverage.png', dpi=200)


