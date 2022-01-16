import gzip
import logging
import math
import pickle
import sys
from fractions import Fraction

import allesfitter
import lightkurve
import numpy
import pandas
from astroplan import Observer
import astropy.units as u
from astropy.time import Time

from lcbuilder.eleanor import TargetData
from sherlockpipe import constants as const
from sherlockpipe import tpfplotter
from sherlockpipe.nbodies.megno import MegnoStabilityCalculator
from sherlockpipe.nbodies.stability_calculator import StabilityCalculator
from sherlockpipe.sherlock import Sherlock

# sherlock = Sherlock(None)
# sherlock = sherlock.filter_multiplanet_ois()
# sherlock.ois.to_csv("multiplanet_ois.csv")

#Allesfitter stuff
alles = allesfitter.allesclass("/mnt/0242522242521AAD/dev/workspaces/git_repositories/sherlockpipe/TIC305048087_[2]_bck/fit_2/")
alles.posterior_samples("lc", "SOI_2_period")
allesfitter.ns_output("/mnt/0242522242521AAD/dev/workspaces/git_repositories/sherlockpipe/run_tests/analysis/TIC142748283_all/fit_0/ttvs_0")
results = pickle.load(open('/mnt/0242522242521AAD/dev/workspaces/git_repositories/sherlockpipe/run_tests/analysis/dietrich/TIC467179528_all/fit_2/results/ns_derived_samples.pickle', 'rb'))
logging.info(results)

#Stability plots
# stability_dir = "/mnt/0242522242521AAD/dev/workspaces/git_repositories/sherlockpipe/run_tests/analysis/dietrich/TIC467179528_all/fit_2/stability_0/"
# df = pandas.read_csv(stability_dir + "stability_megno.csv")
# df = df[(df["megno"] < 3)]
# stability_calc = MegnoStabilityCalculator(5e2)
# for key, row in df.iterrows():
#     stability_calc.plot_simulation(row, stability_dir, str(key))

# filename = "/home/martin/Downloads/resimplepulsations/P2.5_R1.28_1354.197892531283.csv"
# df = pandas.read_csv(filename, float_precision='round_trip', sep=',',
#                                      usecols=['#time', 'flux', 'flux_err'])
# df["flux"] = df["flux"] + 1
# df.to_csv(filename, index=False)

# periods = [5.43, 1.75, 2.62, 6.17]
# p1 = 1.75
# p2 = 5.43
# fraction = Fraction(1.75/5.43).limit_denominator(max_denominator=int(9))
# print(p2 * fraction.numerator % fraction.denominator)

# def compute_resonance(periods, tolerance=0.03):
#     max_number = 9
#     result = []
#     periods = numpy.sort(periods)
#     for i in range(0, len(periods)):
#         result.append([])
#         for j in range(0, len(periods)):
#             if j <= i:
#                 result[i].append("-")
#                 continue
#             fraction = Fraction(periods[i] / periods[j]).limit_denominator(max_denominator=int(max_number))
#             model = periods[j] * fraction.numerator / fraction.denominator
#             result[i].append("-" if numpy.sqrt((model - periods[i]) ** 2) > tolerance \
#                 else str(fraction.numerator) + "/" + str(fraction.denominator))
#     return periods, result
#
# formatter = logging.Formatter('%(message)s')
# logger = logging.getLogger()
# while len(logger.handlers) > 0:
#     logger.handlers.pop()
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# handler.setFormatter(formatter)
# sorted_periods, resonances = compute_resonance([1.75, 5.43, 2.62, 6.17])
# header = "   "
# for period in sorted_periods:
#     header = header + "   " + str(sorted_periods)
# print(sorted_periods)
# for row_label, row in zip(sorted_periods, resonances):
#     print('%s [%s]' % (row_label, ' '.join('%03s' % i for i in row)))

# import psfmachine as psf
# import lightkurve as lk
# tpfs = lk.search_targetpixelfile('TIC 166184428', mission='TESS', sector=11, radius=100, limit=200, cadence='short')[0].download_all(quality_bitmask=None)
# machine = psf.TPFMachine.from_TPFs(tpfs, n_r_knots=10, n_phi_knots=12)
# try:
#     machine.fit_lightcurves(plot=True, fit_va=True)
#     for lc in machine.lcs:
#         lc.to_csv(lc.meta["LABEL"] + ".csv")
# finally:
#     print("FINISHED")

print(str(math.gcd(10, 10.945, 1.14)))

import foldedleastsquares


end = False
count = 0
while count < 1:
    result = transit_template_generator.calculate_results(no_transits_were_fit, chi2, chi2red, chi2_min,
                                                          chi2red_min, test_statistic_periods, test_statistic_depths,
                                                          self, lc_arr, best_row, periods,
                                                          durations, duration, maxwidth_in_samples)
    mask = foldedleastsquares.transit_mask(self.t, result.period, result.duration, result.T0)
    self.t = self.t[~mask]
    self.y = self.y[~mask]
    self.dy = self.dy[~mask]
    residuals_sort_index = numpy.argsort(test_statistic_residuals)
    min_residual_period = test_statistic_periods[residuals_sort_index][0]
    min_residual_period_pos = numpy.argwhere(test_statistic_periods == min_residual_period)
    data = search_period(
        transit_template_generator,
        period=min_residual_period,
        t=self.t,
        y=self.y,
        dy=self.dy,
        transit_depth_min=self.transit_depth_min,
        R_star_min=self.R_star_min,
        R_star_max=self.R_star_max,
        M_star_min=self.M_star_min,
        M_star_max=self.M_star_max,
        lc_arr=lc_arr,
        lc_cache_overview=lc_cache_overview,
        T0_fit_margin=self.T0_fit_margin,
    )
    period_arg = numpy.argwhere(test_statistic_periods == min_residual_period)
    test_statistic_periods[period_arg] = data[0]
    test_statistic_residuals[period_arg] = data[1]
    test_statistic_rows[period_arg] = data[2]
    test_statistic_depths[period_arg] = data[3]
    chi2 = test_statistic_residuals
    chi2red = test_statistic_residuals / (len(self.t) - degrees_of_freedom)
    chi2_min = numpy.min(chi2)
    chi2red_min = numpy.min(chi2red)
    idx_best = numpy.argmin(chi2)
    best_row = test_statistic_rows[idx_best]
    duration = lc_cache_overview["duration"][best_row]
    maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.t))
    result1 = transit_template_generator.calculate_results(no_transits_were_fit, chi2, chi2red, chi2_min,
                                                           chi2red_min, test_statistic_periods, test_statistic_depths,
                                                           self, lc_arr, best_row, periods,
                                                           durations, duration, maxwidth_in_samples)
    if numpy.abs(result.T0 - result1.T0) > 0.01:
        test_statistic_residuals = numpy.delete(test_statistic_residuals, min_residual_period_pos)
        test_statistic_periods = numpy.delete(test_statistic_periods, min_residual_period_pos)
        test_statistic_rows = numpy.delete(test_statistic_rows, min_residual_period_pos)
        test_statistic_depths = numpy.delete(test_statistic_depths, min_residual_period_pos)
        chi2 = numpy.delete(chi2, min_residual_period_pos)
        chi2red = numpy.delete(chi2red, min_residual_period_pos)
        residuals_sort_index = numpy.argsort(test_statistic_residuals)
        min_residual_period = test_statistic_periods[residuals_sort_index][0]
    else:
        count = count + 1

