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


def IsMultipleOf(a, b, tolerance=0.05):
    a = np.float(a)
    b = np.float(b)
    result = a % b
    return (abs(result / b) <= tolerance) or (abs((b - result) / b) <= tolerance)


#::: tls search
def tls_search(time, flux, epoch, period, rplanet):
    SNR = 1e12
    SNR_threshold = 5.
    FOUND_SIGNAL = False

    #::: mask out the first detection at 6.2 days, with a duration of 2.082h, and T0=1712.54922
    # intransit = transit_mask(time, 6.26391, 2*2.082/24., 1712.54922)
    # time = time[~intransit]
    # flux = flux[~intransit]
    time, flux = cleaned_array(time, flux)
    run = 0
    #::: search for the rest
    while (SNR >= SNR_threshold) and (not FOUND_SIGNAL):
        model = transitleastsquares(time, flux)
        R_starx = rstar / u.R_sun
        results = model.power(u=ab,
                              R_star=radius,  # rstar/u.R_sun,
                              R_star_min=rstar_min,  # rstar_min/u.R_sun,
                              R_star_max=rstar_max,  # rstar_max/u.R_sun,
                              M_star=mass,  # mstar/u.M_sun,
                              M_star_min=mstar_min,  # mstar_min/u.M_sun,
                              M_star_max=mstar_max,  # mstar_max/u.M_sun,
                              period_min=0.5,
                              period_max=14,
                              n_transits_min=2,
                              show_progress_bar=False
                              )

        # mass and radius for the TLS
        # rstar=radius
        ##mstar=mass
        # mstar_min = mass-massmin
        # mstar_max = mass+massmax
        # rstar_min = radius-radiusmin
        # rstar_max = radius+raduismax
        SNR = results.snr
        if results.snr >= SNR_threshold:
            intransit = transit_mask(time, results.period, 2 * results.duration, results.T0)
            time = time[~intransit]
            flux = flux[~intransit]
            time, flux = cleaned_array(time, flux)

            #::: check if it found the right signal
            right_period = IsMultipleOf(results.period,
                                        period / 2.)  # check if it is a multiple of half the period to within 5%

            right_epoch = False
            for tt in results.transit_times:
                for i in range(-5, 5):
                    right_epoch = right_epoch or (np.abs(tt - epoch + i * period) < (
                                1. / 24.))  # check if any epochs matches to within 1 hour

            #            right_depth   = (np.abs(np.sqrt(1.-results.depth)*rstar - rplanet)/rplanet < 0.05) #check if the depth matches

            if right_period and right_epoch:
                FOUND_SIGNAL = True
                break
        run = run + 1
    return FOUND_SIGNAL, results.snr, run

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
dir = "../run_tests/experiment/"
report = {}
reports_df = pd.DataFrame(columns=['period', 'radius', 'epoch', 'found', 'snr', 'run'])
for file in os.listdir(dir):
    if file.endswith(".csv"):
        try:
            period = float(re.search("P([0-9]+\\.[0-9]+)", file)[1])
            r_planet = float(re.search("R([0-9]+\\.[0-9]+)", file)[1])
            epoch = float(re.search("_([0-9]+\\.[0-9]+)\\.csv", file)[1])
            sherlock.Sherlock(False, object_infos=[MissionInputObjectInfo("TIC 85400193", dir + file)]) \
                .setup_detrend(True, True, 1.5, 4, 6, "biweight", None, None, multiprocessing.cpu_count() // 2, False, 0.25,
                               "cosine", None) \
                .setup_transit_adjust_params(3, None, None, 10, None, None, 0.4, 20, 10,  multiprocessing.cpu_count() // 2,
                                             8, 5.5, 0.05, "mask", "quorum", 0.2, 0.24, None).run()
            df = pd.read_csv("TIC85400193_INP/candidates.csv", float_precision='round_trip', sep=',',
                             usecols=['curve', 'period', 't0', 'run', 'snr', 'rad_p', 'transits'])
            snr = df["snr"].iloc[len(df) - 1]
            run = df["run"].iloc[len(df) - 1]
            df = df[[IsMultipleOf(df["period"], period / 2.)]]
            right_epoch = False
            if len(df) > 0:
                j = 0
                for tt in df["t0"]:
                    for i in range(-5, 5):
                        right_epoch = right_epoch or (np.abs(tt - epoch + i * period) < (
                                1. / 24.))
                        if right_epoch:
                            snr = df["snr"].iloc[j]
                            run = df["run"].iloc[j]
                        j = j + 1
            found = right_epoch
            new_report = {"period": period, "radius": r_planet, "epoch": epoch, "found": found, "snr": df["snr"], "run": df["run"]}
            reports_df = reports_df.append(new_report, ignore_index=True)
            print("P=" + str(period) + ", R=" + str(r_planet) + ", T0=" + str(epoch) + ", FOUND WAS " + str(found) + " WITH SNR " + str(df["snr"]))
        except Exception as e:
            print(e)
            print("File not valid: " + file)
reports_df.to_csv(dir + "sherlock_report.csv", index=False)
