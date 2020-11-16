#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

#::: modules
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import ellc
from dask import multiprocessing
from sherlockpipe.scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector

from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector

from sherlockpipe.scoring.SnrBorderCorrectedSignalSelector import CorrectedBorderSignalSelection
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask, cleaned_array
from transitleastsquares import catalog_info
import astropy.constants as ac
import astropy.units as u
import lightkurve as lk
from lightkurve import search_lightcurvefile
from scipy import stats
from wotan import t14
from wotan import flatten
import os
import re
import pandas as pd
from sherlockpipe import sherlock
from sherlockpipe.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
import multiprocessing


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
                self.IsMultipleOf(signal_selection.transit_result.period, self.per / 2.) and
                self.isRightEpoch(signal_selection.transit_result.t0, self.t0, self.per)):
            signal_selection.score = 0
        return signal_selection

    def IsMultipleOf(self, a, b, tolerance=0.05):
        a = np.float(a)
        b = np.float(b)
        result = a % b
        return (abs(result / b) <= tolerance) or (abs((b - result) / b) <= tolerance)

    def isRightEpoch(self, t0, known_epoch, known_period):
        right_epoch = False
        for i in range(-5, 5):
            right_epoch = right_epoch or (np.abs(t0 - known_epoch + i * known_period) < (
                    1. / 24.))
        return right_epoch


def IsMultipleOf(a, b, tolerance=0.05):
    a = np.float(a)
    b = np.float(b)
    result = a % b
    return (abs(result / b) <= tolerance) or (abs((b - result) / b) <= tolerance)


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
dir = "/home/pozuelos/martin/curves"
report = {}
reports_df = pd.DataFrame(columns=['period', 'radius', 'epoch', 'found', 'snr', 'run'])
a = False
for file in os.listdir(dir):
    if file.endswith(".csv"):
        try:
            period = float(re.search("P([0-9]+\\.[0-9]+)", file)[1])
            r_planet = float(re.search("R([0-9]+\\.[0-9]+)", file)[1])
            epoch = float(re.search("_([0-9]+\\.[0-9]+)\\.csv", file)[1])
            signal_selection_algorithm = QuorumSnrBorderCorrectedStopWhenMatchSignalSelector(1, 0, period, epoch)
            df = pd.read_csv(dir + file, float_precision='round_trip', sep=',', usecols=['#time', 'flux', 'flux_err'])
            if len(df) == 0:
                found = True
                snr = 20
                run = 1
            else:
                sherlock.Sherlock(False, object_infos=[MissionInputObjectInfo("TIC 85400193", dir + file)]) \
                    .setup_detrend(True, True, 1.5, 4, 6, "biweight", None, None, multiprocessing.cpu_count() - 1, False,
                                   0.25, "cosine", None) \
                    .setup_transit_adjust_params(15, None, None, 10, None, None, 0.4, 14, 10,
                                                 multiprocessing.cpu_count() - 1, 7, 5.5, 0.05, "mask", "quorum", 0.2, 0,
                                                 signal_selection_algorithm)\
                    .run()
                df = pd.read_csv("TIC85400193_INP/candidates.csv", float_precision='round_trip', sep=',',
                                 usecols=['curve', 'period', 't0', 'run', 'snr', 'rad_p', 'transits'])
                snr = df["snr"].iloc[len(df) - 1]
                run = df["run"].iloc[len(df) - 1]
                per_run = 0
                found_period = False
                j = 0
                for per in df["period"]:
                    if signal_selection_algorithm.IsMultipleOf(per, period / 2.):
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
                            break
                found = right_epoch
            new_report = {"period": period, "radius": r_planet, "epoch": epoch, "found": found, "snr": snr,
                          "run": int(run)}
            reports_df = reports_df.append(new_report, ignore_index=True)
            reports_df.to_csv(dir + "a_sherlock_report.csv", index=False)
            print("P=" + str(period) + ", R=" + str(r_planet) + ", T0=" + str(epoch) + ", FOUND WAS " + str(found) +
                  " WITH SNR " + str(snr))
        except Exception as e:
            print(e)
            print("File not valid: " + file)

