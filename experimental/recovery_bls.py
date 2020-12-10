#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import ellc
import wotan
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


def equal(a, b, tolerance=0.01):
    return np.abs(a - b) < tolerance

def IsMultipleOf(a, b, tolerance=0.05):
    a = np.float(a)
    b = np.float(b)
    result = a % b
    return (abs(result / b) <= tolerance) or (abs((b - result) / b) <= tolerance)


#::: tls search
def tls_search(time, flux, epoch, period, rplanet, min_snr, transit_template):
    SNR = 1e12
    FOUND_SIGNAL = False

    #::: mask out the first detection at 6.2 days, with a duration of 2.082h, and T0=1712.54922
    # intransit = transit_mask(time, 6.26391, 2*2.082/24., 1712.54922)
    # time = time[~intransit]
    # flux = flux[~intransit]
    time, flux = cleaned_array(time, flux)
    run = 0
    flux = wotan.flatten(time, flux, window_length=0.25, return_trend=False, method='biweight',
                  break_tolerance=0.5)
    #::: search for the rest
    while (SNR >= min_snr) and (not FOUND_SIGNAL):
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
                              show_progress_bar=False,
                              use_threads=20,
                              transit_template=transit_template
                              )

        # mass and radius for the TLS
        # rstar=radius
        ##mstar=mass
        # mstar_min = mass-massmin
        # mstar_max = mass+massmax
        # rstar_min = radius-radiusmin
        # rstar_max = radius+raduismax
        SNR = results.snr
        if results.snr >= min_snr:
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
dir = "../run_tests/experiment/curves/"
results_dir = "../run_tests/experiment/curves/"
report = {}
reports_df = pd.DataFrame(columns=['period', 'radius', 'epoch', 'found', 'snr', 'run'])
for file in os.listdir(dir):
    if file.endswith(".csv"):
        try:
            period = float(re.search("P([0-9]+\\.[0-9]+)", file)[1])
            r_planet = float(re.search("R([0-9]+\\.[0-9]+)", file)[1])
            epoch = float(re.search("_([0-9]+\\.[0-9]+)\\.csv", file)[1])
            df = pd.read_csv(dir + file, float_precision='round_trip', sep=',', usecols=['#time', 'flux', 'flux_err'])
            if len(df) == 0:
                found = True
                snr = 20
                run = 1
            else:
                lc = lk.LightCurve(time=df['#time'], flux=df['flux'], flux_err=df['flux_err'])
                clean = lc.remove_nans().remove_outliers(sigma_lower=float('inf'), sigma_upper=3)  # remove outliers over 3sigma
                flux = clean.flux
                time = clean.time
                found, snr, run = tls_search(time, flux, epoch, period, r_planet, 7, "box")
            new_report = {"period": period, "radius": r_planet, "epoch": epoch, "found": found, "snr": snr, "run": run}
            reports_df = reports_df.append(new_report, ignore_index=True)
            print("P=" + str(period) + ", R=" + str(r_planet) + ", T0=" + str(epoch) + ", FOUND WAS " + str(found) + " WITH SNR " + str(snr))
            reports_df.to_csv(dir + "a_bls_report.csv", index=False)
        except:
            print("File not valid: " + file)

df = pd.read_csv(dir + "a_bls_report.csv")
min_period = df["period"].min()
max_period = df["period"].max()
min_rad = df["radius"].min()
max_rad = 3.19 #df["radius"].max()
period_grid = np.around(np.arange(min_period, max_period + 0.1, 0.5), 1)
radius_grid = np.around(np.arange(min_rad, max_rad + 0.1, 0.1), 1)
result = np.zeros((len(period_grid), len(radius_grid)))
for i in period_grid:
    ipos = int(round((i - min_period) * 2))
    for j in radius_grid:
        jpos = int(round((j - min_rad) * 10))
        sel_df = df[equal(df["period"], i)]
        sel_df = sel_df[equal(sel_df["radius"], j)]
        found_count = len(sel_df[sel_df["found"]])
        result[ipos][jpos] = found_count
result = np.transpose(result)
fig, ax = plt.subplots()
im = ax.imshow(result)
ax.set_xticks(np.arange(len(period_grid)))
ax.set_yticks(np.arange(len(radius_grid)))
ax.set_xticklabels(period_grid)
ax.set_yticklabels(radius_grid)
ax.set_xlabel("Period")
ax.set_ylabel("Radius")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Found transits count", rotation=-90, va="bottom")
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.gca().invert_yaxis()
ax.set_title("BLS Period/radius recovery")
fig.tight_layout()
plt.savefig(dir + "a_bls_report.png")
plt.close()

