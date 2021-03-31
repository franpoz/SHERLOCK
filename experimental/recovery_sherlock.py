#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
from sherlockpipe.scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector
from transitleastsquares import catalog_info
import astropy.units as u
import lightkurve as lk
import os
import re
import pandas as pd
from sherlockpipe import sherlock
from lcbuilder.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo


class QuorumSnrBorderCorrectedStopWhenMatchSignalSelector(QuorumSnrBorderCorrectedSignalSelector):
    def __init__(self, strength=1, min_quorum=0, per=None, t0=None):
        super().__init__()
        self.strength = strength
        self.min_quorum = min_quorum
        self.per = per
        self.t0 = t0

    def select(self, transit_results, snr_min, detrend_method, wl):
        signal_selection = super(QuorumSnrBorderCorrectedStopWhenMatchSignalSelector, self) \
            .select(transit_results, snr_min, detrend_method, wl)
        if signal_selection.score == 0 or (
                self.is_harmonic(signal_selection.transit_result.period, self.per) and
                self.isRightEpoch(signal_selection.transit_result.t0, self.t0, self.per)):
            signal_selection.score = 0
        return signal_selection

    def is_harmonic(self, a, b, tolerance=0.05):
        a = np.float(a)
        b = np.float(b)
        mod_ab = a % b
        mod_ba = b % a
        return (a > b and a < b * 3 + tolerance * 3 and (
                    abs(mod_ab % 1) <= tolerance or abs((b - mod_ab) % 1) <= tolerance)) or \
               (b > a and a > b / 3 - tolerance / 3 and (
                           abs(mod_ba % 1) <= tolerance or abs((a - mod_ba) % 1) <= tolerance))

    def isRightEpoch(self, t0, known_epoch, known_period):
        right_epoch = False
        for i in range(-5, 5):
            right_epoch = right_epoch or (np.abs(t0 - known_epoch + i * known_period) < (
                    1. / 24.))
        return right_epoch


def IsMultipleOf(a, b, tolerance=0.05):
    a = np.float(a)
    b = np.float(b)
    mod_ab = a % b
    mod_ba = b % a
    return (a > b and a < b * 3 + tolerance * 3 and (abs(mod_ab % 1) <= tolerance or abs((b - mod_ab) % 1) <= tolerance)) or \
           (b > a and a > b / 3 - tolerance / 3 and (abs(mod_ba % 1) <= tolerance or abs((a - mod_ba) % 1) <= tolerance))


# print(IsMultipleOf(0.5 / 2 + 0.005, 0.5))
# print(IsMultipleOf(0.5 / 3 - 0.005, 0.5))
# print(IsMultipleOf(0.5 / 4 + 0.01, 0.5))
# print(IsMultipleOf(0.5 + 0.01, 0.5))
# print(IsMultipleOf(0.5 * 2 + 0.01, 0.5))
# print(IsMultipleOf(0.5 * 3 + 0.01, 0.5))
# print(IsMultipleOf(0.5 * 4 + 0.01, 0.5))
# print(IsMultipleOf(0.75, 0.5))
# print(IsMultipleOf(1.25, 0.5))
# print(IsMultipleOf(5 + 0.01, 0.5))

#::: load data and set the units correctly
TIC_ID = 85400193  # TIC_ID of our candidate
lcf = lk.search_lightcurvefile('TIC ' + str(TIC_ID), mission="tess").download_all()
ab, mass, massmin, massmax, radius, radiusmin, radiusmax = catalog_info(TIC_ID=TIC_ID)
# units for ellc
rstar = radius * u.R_sun
# mass and radius for the TLS
# rstar=radius
# mstar=mass
mstar_min = mass - massmin
mstar_max = mass + massmax
rstar_min = radius - radiusmin
rstar_max = radius + radiusmax
dir = "/home/mdevora/ir/toi2096/curves/"
report = {}
reports_df = pd.DataFrame(columns=['period', 'radius', 'epoch', 'found', 'snr', 'sde', 'run'])
a = False
tls_report_df = pd.read_csv(dir + "a_tls_report.csv", float_precision='round_trip', sep=',', usecols=['period', 'radius',
                                                                                            'epoch', 'found',
                                                                                            'snr', 'run'])
samples = 4
samples_analysed = 4
for index, row in tls_report_df.iterrows():
    file = os.path.join('P' + str(row['period']) + '_R' + str(row['radius']) + '_' + str(row['epoch']) + '.csv')
    last_true = index < len(tls_report_df) - 1 and tls_report_df.iloc[index]['found'] and \
                not tls_report_df.iloc[index + 1]['found']
    if last_true:
        samples_analysed = 0
    if samples_analysed < 4:
        try:
            samples_analysed = samples_analysed + 1
            period = float(re.search("P([0-9]+\\.[0-9]+)", file)[1])
            r_planet = float(re.search("R([0-9]+\\.[0-9]+)", file)[1])
            epoch = float(re.search("_([0-9]+\\.[0-9]+)\\.csv", file)[1])
            signal_selection_algorithm = QuorumSnrBorderCorrectedStopWhenMatchSignalSelector(1, 0, period, epoch)
            df = pd.read_csv(dir + file, float_precision='round_trip', sep=',', usecols=['#time', 'flux', 'flux_err'])
            if len(df) == 0:
                found = True
                snr = 20
                sde = 20
                run = 1
            else:
                sherlock.Sherlock(False, sherlock_targets=[MissionInputObjectInfo("TIC 85400193", dir + file)]) \
                    .setup_detrend(True, True, 1.5, 4, 12, "biweight", 0.2, 1.0, 20, False,
                                   0.25, "cosine", None) \
                    .setup_transit_adjust_params(5, None, None, 10, None, None, 0.4, 14, 10,
                                                 20, 5, 5.5, 0.05, "mask", "quorum", 1, 0,
                                                 signal_selection_algorithm)\
                    .run()
                df = pd.read_csv("TIC85400193_INP/candidates.csv", float_precision='round_trip', sep=',',
                                 usecols=['curve', 'period', 't0', 'run', 'snr', 'sde', 'rad_p', 'transits'])
                snr = df["snr"].iloc[len(df) - 1]
                run = df["run"].iloc[len(df) - 1]
                sde = df["sde"].iloc[len(df) - 1]
                per_run = 0
                found_period = False
                j = 0
                for per in df["period"]:
                    if signal_selection_algorithm.is_harmonic(per, period / 2.):
                        found_period = True
                        t0 = df["t0"].iloc[j]
                        break
                    j = j + 1
                right_epoch = False
                if found_period:
                    for i in range(-5, 5):
                        right_epoch = right_epoch or (np.abs(t0 - epoch + i * period) < (
                                1. / 24.))
                        if right_epoch:
                            snr = df["snr"].iloc[j]
                            run = df["run"].iloc[j]
                            sde = df["sde"].iloc[j]
                            break
                found = right_epoch
            new_report = {"period": period, "radius": r_planet, "epoch": epoch, "found": found, "sde": sde, "snr": snr,
                          "run": int(run)}
            reports_df = reports_df.append(new_report, ignore_index=True)
            reports_df.to_csv(dir + "a_sherlock_report.csv", index=False)
            print("P=" + str(period) + ", R=" + str(r_planet) + ", T0=" + str(epoch) + ", FOUND WAS " + str(found) +
                  " WITH SNR " + str(snr) + "and SDE " + str(sde))
        except Exception as e:
            print(e)
            print("File not valid: " + file)

