from __future__ import print_function, division, absolute_import

import sys

import lightkurve
import foldedleastsquares as tls
import multiprocessing
import numpy as np
import wotan
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter


# lcf = lightkurve.search_lightcurve("TIC 352315023", mission="TESS", cadence="short", sector=[13, 27], author="SPOC")\
#     .download_all()
# lc = lcf.stitch().remove_nans()
# lc = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper=3)
# model = tls.transitleastsquares(lc.time.value, lc.flux.value)
# results = model.power(period_min=0.45, period_max=40, use_threads=multiprocessing.cpu_count(),
#                       oversampling_factor=1.1119355997446583, T0_fit_margin=0.05, duration_grid_step=1.1)
# print(results)
from scipy import stats, signal
from scipy.interpolate import interp1d
import time
t0 = time.time()
lcf = lightkurve.search_lightcurve("TIC 251848941", mission="TESS", cadence="short", sector=[2], author="SPOC")\
    .download_all()
lc = lcf.stitch().remove_nans()
lc = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper=3)
lc_time = lc.time.value
flux = lc.flux.value
cadence = 2
window_length = 25 / cadence
flux = savgol_filter(flux, 11, 3)
R_s = 1.1
M_s = 1.3
P_min = 0.5
P_max = 22
ld_coefficients = [0.2, 0.1]
min_duration = wotan.t14(R_s, M_s, P_min, True)
max_duration = wotan.t14(R_s, M_s, P_max, True)
duration_grid = np.arange(min_duration * 24 * 60 // cadence, max_duration * 24 * 60 // cadence, 1)


def calculate_semi_major_axis(period, star_mass):
    G = 6.674e-11
    period_seconds = period * 24. * 3600.
    mass_kg = star_mass * 2.e30
    a1 = (G * mass_kg * period_seconds ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
    return a1 / 1.496e11

a_au = calculate_semi_major_axis(0.5, M_s)
a_Rs = a_au / (R_s * 0.00465047)
# TODO start with rp_rs that causes twice depth than curve RMS
curve_rms = np.std(flux)
min_depth = 2 * curve_rms
initial_rp = (min_depth * (R_s ** 2)) ** (1 / 2)
rp_rs = initial_rp / R_s
from pytransit import QuadraticModel
tm = QuadraticModel()
time_model = np.arange(0, 1, 0.0001)
tm.set_data(time_model)
# k is the radius ratio, ldc is the limb darkening coefficient vector, t0 the zero epoch, p the orbital period, a the
# semi-major axis divided by the stellar radius, i the inclination in radians, e the eccentricity, and w the argument
# of periastron. Eccentricity and argument of periastron are optional, and omitting them defaults to a circular orbit.
model = tm.evaluate(k=rp_rs, ldc=ld_coefficients, t0=0.5, p=1.0, a=a_Rs, i=0.5*np.pi)
model = model[model < 1]
baseline_model = np.full(len(model), 1)
model = np.append(baseline_model, model)
model = np.append(model, baseline_model)



def downsample(array, npts: int):
    interpolated = interp1d(np.arange(len(array)), array, axis=0, fill_value='extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled


model_samples = [downsample(model, int(duration) * 3) for duration in duration_grid]


def calculate_residuals(time, flux, model_sample, flux_index):
    flux_subset = flux[flux_index:flux_index + len(model_sample)]
    time_subset = time[flux_index:flux_index + len(model_sample)]
    # TODO adjusting model to minimum flux value this might get improved by several scalations of min_flux
    model_sample_scaled = np.copy(model_sample)
    flux_subset_len = len(flux_subset)
    max_flux = 1 - np.std(flux_subset) / 2
    flux_at_middle = np.mean(flux_subset[flux_subset_len // 3:flux_subset_len * 2 // 3])
    if flux_at_middle < max_flux:
        model_sample_scaled[model_sample_scaled < 1] = model_sample_scaled[model_sample_scaled < 1] * (flux_at_middle / np.min(model_sample))
    # fig_transit, axs = plt.subplots(1, 1, figsize=(8, 8))
    # clean_flux_subset = wotan.flatten(time_subset, flux_subset, len(model_sample))
    # axs.plot(time_subset, model_sample_scaled, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
    # axs.plot(time_subset, flux_subset, color='red', alpha=1, rasterized=True, label="Flux Transit ")
    # axs.plot(time_subset, clean_flux_subset, color='pink', alpha=1, rasterized=True, label="Flux Transit ")
    # axs.set_title("Residuals")
    # axs.set_xlabel('Time')
    # axs.set_ylabel('Flux')
    # fig_transit.show()
        depth = 1 - flux_at_middle
        return np.sum((flux_subset - model_sample_scaled) ** 2) ** 0.5 * depth
    return np.nan


cumulative_residuals = np.full(len(lc_time), np.nan)
residual_calculation = np.full((len(model_samples), len(lc_time)), np.nan)
for model_index, model_sample in enumerate(model_samples):
    model_duration_days = duration_grid[model_index] * cadence / 60 / 24
    first_valid_time = lc_time[lc_time > lc_time[0] + model_duration_days * 3][0]
    time_without_tail = lc_time[lc_time < lc_time[len(lc_time) - 1] - model_duration_days * 3]
    last_valid_time = time_without_tail[len(time_without_tail) - 1]
    first_valid_time = lc_time[0]
    last_valid_time = lc_time[len(lc_time) - 1 - len(model_sample)]
    dt_flux = wotan.flatten(lc_time, flux, model_duration_days * 4, method="biweight")
    dt_flux = flux
    for flux_index, flux_value in enumerate(lc_time[(lc_time >= first_valid_time) & (lc_time <= last_valid_time)]):
        residual_calculation[model_index][flux_index] = calculate_residuals(lc_time, dt_flux, model_sample, flux_index)
    local_residual_minima = argrelextrema(residual_calculation[model_index], np.less)[0]
    minima_mask = np.full(len(residual_calculation[model_index]), False)
    minima_mask[local_residual_minima] = True
    max_allowed_residual = np.nanmax(residual_calculation[model_index])
    residual_calculation[model_index][np.where(np.isnan(residual_calculation[model_index]))] = max_allowed_residual
    #residual_calculation[model_index][~minima_mask] = max_allowed_residual
    #time_plot = time[minima_mask]
    residual_plot = residual_calculation[model_index]
    fig_transit, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(lc_time, dt_flux, color='gray', alpha=1, rasterized=True, label="Flux")
    axs[0].set_title("Light curve" + str(model_duration_days * 24 * 60) + "m")
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Flux')
    axs[1].plot(lc_time, residual_plot, color='gray', alpha=1, rasterized=True, label="Residuals")
    axs[1].set_title("Residuals for transit duration " + str(model_duration_days * 24 * 60) + "m")
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Residuals')
    fig_transit.show()

cumulative_residuals = np.sum(residual_calculation, axis=0)
residual_plot = cumulative_residuals
fig_transit, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(lc_time, flux, color='gray', alpha=1, rasterized=True, label="Flux")
axs[0].set_title("Light curve")
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Flux')
axs[1].plot(lc_time, cumulative_residuals, color='gray', alpha=1, rasterized=True, label="Residuals")
axs[1].set_title("Cumulative Residuals")
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Residuals')
fig_transit.show()
pgram = signal.lombscargle(lc_time, cumulative_residuals, np.linspace(0.04347, 2, 100000), normalize=True)
fig_transit, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(lc_time, flux, color='gray', alpha=1, rasterized=True, label="Flux")
axs[0].set_title("Light curve")
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Flux')
axs[1].plot(np.linspace(0.5, 23, 100000), pgram, color='gray', alpha=1, rasterized=True, label="Periodogram")
axs[1].set_title("Periodogram")
axs[1].set_xlabel('Freq')
axs[1].set_ylabel('Power')
fig_transit.show()
print("END: Took " + str(time.time() - t0) + "s")
#tm.evaluate(k=[0.01, 0.12], ldc=[[0.2, 0.1, 0.5, 0.1]], t0=0.0, p=1.0, a=3.0, i=0.5*pi)


# import pandas
# file = "run_tests/experiment/ir/pulsator_TIC13145616/time-series_prewithened_SYNTHETIC2.out"
# df = pandas.read_csv(file)
# df["flux"] = df["flux"] + 1
# df.to_csv(file, index=False)


