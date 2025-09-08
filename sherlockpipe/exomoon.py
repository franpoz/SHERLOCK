import itertools
import logging
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool
import astropy.units as u
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from scipy.optimize import minimize
from batman import TransitModel
from numba import njit
from pytransit import QuadraticModel
import batman
from tqdm import tqdm
import pandoramoon as pandora
import ellc
import numpy as np
import astropy.constants as ac
import astropy.units as u
import wotan
from lcbuilder.lcbuilder_class import LcBuilder
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import pandas as pd
from sherlockpipe.search.sherlock import Sherlock

G = 6.674e-11  # m3 kg-1 s-2
AU_TO_RSUN = 215.032
Msolar_to_kg = 2.e30
Mearth_to_kg = 5.972e24
M_earth_to_M_sun = Mearth_to_kg / Msolar_to_kg
R_earth_to_R_sun = 0.009175


class ExoMoonLeastSquares:
    MAX_RESIDUAL = sys.maxsize

    def __init__(self, object_dir, cpus, star_mass, star_radius, ab, planet_radius, planet_radius_err,
                 planet_mass, planet_period,
                 planet_t0, planet_duration, planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc,
                 planet_ecc_err, planet_arg_periastron, planet_impact_param, min_radius, max_radius, t0s, time, flux,
                 flux_err, max_moon_density, period_grid_size=2000, radius_grid_size=10,
                 min_cadences_per_t0: bool = 20):
        self.object_dir = object_dir
        self.cpus = cpus
        self.star_mass = star_mass
        self.star_radius = star_radius
        self.ab = ab
        self.planet_radius = planet_radius
        self.planet_radius_err = planet_radius_err
        self.planet_mass = planet_mass
        self.planet_period = planet_period
        self.planet_t0 = planet_t0
        self.planet_duration = planet_duration
        self.planet_semimajor_axis = planet_semimajor_axis
        self.planet_inc = planet_inc
        self.planet_inc_err = planet_inc_err
        self.planet_ecc = planet_ecc
        self.planet_ecc_err = planet_ecc_err
        self.planet_arg_periastron = planet_arg_periastron
        self.planet_impact_param = planet_impact_param
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.max_moon_density = max_moon_density
        self.min_cadences_per_t0 = min_cadences_per_t0
        self.bary_t0s = []
        for i, bary_t0 in enumerate(t0s):
            mask = np.abs(time - bary_t0) < self.planet_duration * 2
            if not self.is_skip_t0(time[mask]):
                self.bary_t0s.append(bary_t0)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.period_grid_size = period_grid_size
        self.radius_grid_size = radius_grid_size

    @staticmethod
    def compute_semimajor_axis(major_mass, minor_period):
        period_seconds = minor_period * 24. * 3600.
        mass_kg = major_mass * Msolar_to_kg
        a1 = (G * mass_kg * period_seconds ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
        return a1 / 1.496e11

    @staticmethod
    def compute_roche_period(moon_density):
        return np.sqrt((3 * np.pi) / (G * moon_density * 1000)) * 2.44 ** (3 / 2) / 3600 / 24

    @staticmethod
    def compute_hill_radius(major_mass, minor_mass, semimajor_axis, eccentricity=0):
        """
        :param major_mass: The main body mass
        :param minor_mass: The minor body mass
        :param semimajor_axis: The minor body semimajor axis in AU.
        :param eccentricity: the planet eccentricity
        :return: the hill radius of the minor body in the same units than the semimajor_axis
        """
        return AU_TO_RSUN * semimajor_axis * (1 - eccentricity) * (minor_mass / (3 * major_mass) ** (1 / 3))

    @staticmethod
    def au_to_period(mass, au):
        """
        Calculates the orbital period for the semi-major axis assuming a circular orbit.
        :param mass: the stellar mass
        :param au: the semi-major axis in astronomical units.
        :return: the period in days
        """
        mass_kg = mass * 2.e30
        a = au * 1.496e11
        return ((a ** 3) * 4 * (np.pi ** 2) / G / mass_kg) ** (1. / 2.) / 3600 / 24

    @staticmethod
    def compute_transit_duration(star_radius,
                                 transiting_body_semimajor_axis, transit_period, transiting_body_radius,
                                 impact_parameter=0):
        """

        :param star_radius: star radius
        :param transiting_body_semimajor_axis: orbit semimajor axis
        :param transit_period: in days
        :param transiting_body_radius: transiting body radius
        :param impact_parameter:
        :return:
        """
        return transit_period / np.pi * np.arcsin(np.sqrt((star_radius + transiting_body_radius) ** 2 - (impact_parameter * star_radius) ** 2) / transiting_body_semimajor_axis)
        #return 2 * moon_semimajor_axis / (planet_semimajor_axis * 2 * np.pi) * planet_period

    @staticmethod
    def compute_moon_period_grid(min, max, mode="lin", samples=10000):
        if "log" == mode:
            return np.logspace(math.log(min, 10), math.log(max, 10), samples, base=10)
        else:
            return np.linspace(min, max, samples)

    @staticmethod
    def fit_single_transit(time, flux, flux_err, guess_params, guess_params_bounds, transit_params: batman.TransitParams):
        # params = global transit params from literature, except T0
        def model(theta, t):
            transit_params.t0 = theta[0]
            transit_params.a = theta[1]
            transit_params.inc = theta[2]
            transit_params.rp = theta[3]
            m = TransitModel(transit_params, t)
            return m.light_curve(transit_params)

        def neg_loglike(theta):
            m = model(theta, time)
            res = (flux - m) / flux_err
            # penaliza geometrías sin tránsito (b>1) para evitar modelo plano
            # b = (a * cos i) / R*  con i en radianes:
            _, a, inc, _ = theta
            b = a * np.cos(np.deg2rad(inc))
            penalty = 0.0
            if b > 1.0:
                penalty += 1e6 * (b - 1.0) ** 2  # empuja lejos del "no-tránsito"
            return 0.5 * np.sum(res ** 2) + penalty
        if len(time > 0):
            res = minimize(neg_loglike, guess_params, method='Powell', bounds=guess_params_bounds)
            best_theta = res.x
            best_model = model(best_theta, time)
        else:
            best_theta = guess_params
            best_model = model(guess_params, time)
        return best_theta, best_model

    def fit_single_transits(self, time, flux, flux_err, plot=False):
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        planet_duration_plot = planet_duration * 2
        params = batman.TransitParams()
        params.per = self.planet_period
        params.ecc = self.planet_ecc  # eccentricity
        params.w = self.planet_arg_periastron  # longitude of periastron (deg)
        params.rp = self.planet_radius / star_radius_rearth  # planet radius / stellar radius
        params.a = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun,
                                                                                u.au)  # semi-major axis / stellar radius
        params.inc = self.planet_inc  # orbital inclination [deg]
        params.u = self.ab  # limb darkening coefficients
        params.limb_dark = "quadratic"
        for i, guess_t0 in enumerate(self.bary_t0s):
            mask = np.abs(time - guess_t0) < self.planet_duration * 2
            if self.is_skip_t0(time[mask]):
                continue
            params.t0 = guess_t0  # this will be updated for each transit
            guess_params = [guess_t0, params.a, params.inc, params.rp]
            guess_params_bounds = [(time.min(), time.max()),
                      (guess_params[1] - guess_params[1] / 100, guess_params[1] + guess_params[1] / 100),
                      (guess_params[2] - self.planet_inc_err, guess_params[2] + self.planet_inc_err),
                                   (guess_params[3] - self.planet_radius_err / star_radius_rearth,
                                    guess_params[3] + self.planet_radius_err / star_radius_rearth)]
            best_theta, best_model = ExoMoonLeastSquares.fit_single_transit(time[mask], flux[mask], flux_err[mask],
                                                                            guess_params, guess_params_bounds, params)
            best_t0 = best_theta[0]
            if plot:
                times_in_t0 = times[(times > best_t0 - planet_duration_plot) & (times < best_t0 + planet_duration_plot)]
                if len(times_in_t0) > 20:
                    fig_transit, axs = plt.subplots(1, 1, figsize=(12, 12))
                    axs.scatter(time[mask],
                                flux[mask],
                                color='gray', alpha=1, rasterized=True, label="Flux")
                    axs.plot(time[mask], best_model,
                             color='red', alpha=1, rasterized=True, label="Flux")
                    axs.set_title("Planet transit at t0=" + str(best_t0))
                    axs.set_xlabel('Time')
                    axs.set_ylabel('Flux')
                    plt.savefig(object_dir + "/T0_" + str(round(best_t0, 3)) + ".png")
                    plt.close(fig_transit)

    def subtract_planet_transit(self, time, flux, flux_err, mode='subtract', plot=False):
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        params = batman.TransitParams()
        params.per = self.planet_period
        params.rp = self.planet_radius / star_radius_rearth  # planet radius / stellar radius
        params.a = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun,
                                                                                u.au)  # semi-major axis / stellar radius
        params.inc = self.planet_inc  # orbital inclination [deg]
        params.ecc = self.planet_ecc  # eccentricity
        params.w = self.planet_arg_periastron  # longitude of periastron (deg)
        params.u = self.ab  # limb darkening coefficients
        params.limb_dark = "quadratic"
        params.t0 = planet_t0
        duration_window = self.planet_duration * 2
        result_model = flux.copy()
        result_time = time.copy()
        if flux_err is not None:
            result_flux_err = flux_err.copy()
        else:
            result_flux_err = None
        params = batman.TransitParams()
        params.per = self.planet_period
        params.ecc = self.planet_ecc  # eccentricity
        params.w = self.planet_arg_periastron  # longitude of periastron (deg)
        params.rp = self.planet_radius / star_radius_rearth  # planet radius / stellar radius
        params.limb_dark = "quadratic"
        self.ttvs = []
        for t0 in self.bary_t0s:
            if mode == 'subtract':
                mask = np.abs(time - t0) < duration_window
                intransit_indexes = np.argwhere(np.abs(time - t0) < duration_window).flatten()
                params.t0 = t0  # this will be updated for each transit
                params.inc = self.planet_inc  # orbital inclination [deg]
                params.a = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun,
                                                                                        u.au)  # semi-major axis / stellar radius
                params.u = self.ab  # limb darkening coefficients
                #b = a * np.cos(np.deg2rad(inc)) ensure b = 1 to calculate minimum inclination
                min_inc = np.rad2deg(np.arccos(1 / params.a))
                guess_params = [t0, params.a, params.inc, params.rp]
                guess_params_bounds = [(time[mask].min(), time[mask].max()),
                                       (guess_params[1] - guess_params[1] / 100,
                                        guess_params[1] + guess_params[1] / 100),
                                       (90 - min_inc, 90),
                                       (guess_params[3] - self.planet_radius_err / star_radius_rearth,
                                        guess_params[3] + self.planet_radius_err / star_radius_rearth)]
                best_theta, best_model = ExoMoonLeastSquares.fit_single_transit(time[mask], flux[mask], flux_err[mask],
                                                                                guess_params, guess_params_bounds, params)
                best_t0 = best_theta[0]
                self.ttvs.append(t0 - best_t0)
                result_model[intransit_indexes] = result_model[intransit_indexes] - (best_model - 1)
                if plot:
                    fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
                    axs[0].scatter(time[intransit_indexes], flux[
                        intransit_indexes],
                                   color='gray', alpha=1, rasterized=True, label="Flux")
                    axs[0].plot(time[intransit_indexes], best_model)
                    axs[0].set_title("Subtracting transit in t0 " + str(best_t0))
                    axs[0].set_xlabel('Time')
                    axs[0].set_ylabel('Flux')
                    axs[1].scatter(time[intransit_indexes], result_model[intransit_indexes],
                                color='gray', alpha=1, rasterized=True, label="Flux")
                    axs[1].set_title(
                        "Subtracted transit in t0 " + str(best_t0))
                    axs[1].set_xlabel('Time')
                    axs[1].set_ylabel('Flux')
                    plt.savefig(self.object_dir + "/T0_" + str(round(best_t0, 3)) + "_sub.png")
                    plt.close()
            else:
                intransit_mask = np.abs(self.time - t0) <= duration_window / 2
                result_time = result_time[~intransit_mask]
                result_model = result_model[~intransit_mask]
                if result_flux_err is not None:
                    result_flux_err = result_flux_err[~intransit_mask]
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(self.ttvs)), self.ttvs, marker='o')
        plt.xticks(range(len(self.ttvs)), [f"{t0:.4f}" for t0 in self.bary_t0s], rotation=45)
        plt.xlabel("Barycentric T0 Values")
        plt.ylabel("TTV (days)")
        plt.title("Transit Timing Variations")
        plt.tight_layout()
        plt.savefig(self.object_dir + "/ttvs.png")
        plt.close()
        return time, result_model, result_flux_err

    @staticmethod
    def compute_moon_ts(planet_t0, half_moon_orbit_transit_duration, half_moon_transit_duration):
        min_t1 = planet_t0 - half_moon_orbit_transit_duration
        min_t0 = min_t1 + half_moon_transit_duration
        max_moon_t0_tau = np.abs(planet_t0 - min_t0)
        return max_moon_t0_tau

    @staticmethod
    def compute_moon_t0(planet_first_t0, planet_t0, moon_period, moon_initial_alpha, max_moon_t0_tau):
        moon_phase = moon_initial_alpha + (((planet_t0 - planet_first_t0) % moon_period) / moon_period) * 2 * np.pi
        moon_tau = np.cos(moon_phase)
        moon_t0 = planet_t0 + moon_tau * max_moon_t0_tau
        return moon_t0, moon_phase


    @staticmethod
    #@njit(fastmath=True, parallel=False)
    def compute_moon_transit_scenarios(time, flux, flux_err, planet_t0, moon_initial_alpha, moon_period, bary_t0s,
                                       half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration,
                                       half_moon_transit_duration):
        #TODO need to take into account "prograde" or "retrograde" orbit
        orbit_scenarios = None
        for t0 in bary_t0s:
            max_moon_t0_tau = ExoMoonLeastSquares.compute_moon_ts(t0, half_moon_orbit_transit_duration, half_moon_transit_duration)
            moon_t0, moon_alpha = ExoMoonLeastSquares.compute_moon_t0(planet_t0, t0, moon_period, moon_initial_alpha, max_moon_t0_tau)
            moon_t1 = moon_t0 - half_moon_transit_duration
            # t1 = moon_orbit_range[1]
            # phase_delta = (t0 - planet_t0) % moon_period * 2 * np.pi
            # alpha = (moon_initial_alpha + phase_delta) % (2 * np.pi)
            # time_alpha = np.cos(alpha) * moon_orbit_transit_length / 2
            # moon_t1 = t1 + time_alpha
            time_orbit_range_args = np.argwhere(np.abs(time - t0) <= half_moon_max_orbit_transit_duration).flatten()
            time_moon_transit = time[time_orbit_range_args]
            flux_moon_transit = flux[time_orbit_range_args]
            flux_err_moon_transit = flux_err[time_orbit_range_args]
            time_moon_transit = time_moon_transit - (moon_t1 + half_moon_transit_duration)
            # fig_transit, axs = plt.subplots(1, 1, figsize=(8, 8))
            # axs.scatter(time_moon_transit, flux_moon_transit, color='gray', alpha=1, rasterized=True,
            #             label="Flux Transit ")
            # axs.set_title("Residuals")
            # axs.set_xlabel('Time')
            # axs.set_ylabel('Flux')
            # fig_transit.show()
            if len(time_moon_transit) > 0:
                if orbit_scenarios is None:
                    orbit_scenarios = [[moon_alpha, time_moon_transit, flux_moon_transit, flux_err_moon_transit]]
                orbit_scenarios.append([moon_alpha, time_moon_transit, flux_moon_transit, flux_err_moon_transit])
        return orbit_scenarios

    def search(self, search_input, return_lc=False):
        transit_scenarios = ExoMoonLeastSquares.compute_moon_transit_scenarios(self.time, self.flux, self.flux_err,
                                                                               self.bary_t0s[0], search_input.moon_alpha,
                                                                search_input.moon_period, self.bary_t0s,
                                                                               search_input.half_moon_orbit_transit_duration,
                                                                search_input.half_moon_max_orbit_transit_duration,
                                                                               search_input.half_moon_transit_duration)
        scenario_time = []
        scenario_flux = []
        scenario_flux_err = []
        for normalized_moon_transit_scenario in transit_scenarios:
            scenario_time = np.concatenate((scenario_time, normalized_moon_transit_scenario[1].flatten()))
            scenario_flux = np.concatenate((scenario_flux, normalized_moon_transit_scenario[2].flatten()))
            scenario_flux_err = np.concatenate((scenario_flux_err, normalized_moon_transit_scenario[3].flatten()))
        sorted_time_args = np.argsort(scenario_time)
        scenario_time = scenario_time[sorted_time_args]
        scenario_flux = scenario_flux[sorted_time_args]
        scenario_flux_err = scenario_flux_err[sorted_time_args]
        outliers_args = ExoMoonLeastSquares.remove_outliers(scenario_flux, sigma_lower=float('inf'), sigma_upper=3)
        scenario_time = scenario_time[~outliers_args].flatten()
        scenario_flux = scenario_flux[~outliers_args].flatten()
        scenario_flux_err = scenario_flux_err[~outliers_args].flatten()
        #TODO check whether to use this mean or the real scenario_flux
        #scenario_flux = ExoMoonLeastSquares.running_mean_equal_length(scenario_flux, len(scenario_flux) // 100)
        it_args = np.argwhere(np.abs(scenario_time) <= search_input.half_moon_transit_duration).flatten()
        oot_args = np.argwhere(np.abs(scenario_time) > search_input.half_moon_transit_duration).flatten()
        it_time = scenario_time[it_args]
        oot_time = scenario_time[oot_args]
        it_flux = scenario_flux[it_args]
        oot_flux = scenario_flux[oot_args]
        it_flux_err = scenario_flux_err[it_args]
        oot_flux_err = scenario_flux_err[oot_args]
        interpolated = interp1d(np.arange(len(self.model)), self.model, axis=0, fill_value='extrapolate')
        model_sample = interpolated(np.linspace(0, len(self.model), len(it_flux)))
        # fig_transit, axs = plt.subplots(1, 1, figsize=(8, 8))
        # axs.scatter(scenario_time, scenario_flux, color='gray', alpha=0.4, rasterized=True, label="Flux Transit ")
        # axs.plot(scenario_time, model_sample, color='red', alpha=1, rasterized=True, label="Flux Transit ")
        # axs.set_title("Residuals")
        # axs.set_xlabel('Time')
        # axs.set_ylabel('Flux')
        # fig_transit.show()
        t0_flux_mean_value = np.mean(it_flux)
        residual_calculation, residual_baseline, residual_radius, residual_model = (
            self.calculate_residuals(it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                                     t0_flux_mean_value, len(scenario_time), model_sample, self.min_radius,
                                     self.max_radius, self.star_radius, self.radius_grid_size))
        #oot_noise = np.std(oot_flux)
        #oot_noise = np.sqrt(np.std(oot_flux) * (np.sqrt(np.sum(oot_flux_err ** 2) / len(oot_flux_err))))
        oot_noise = np.std(oot_flux)
        model_snr = (1 - t0_flux_mean_value) / oot_noise * np.sqrt(len(it_flux))
        if return_lc:
            return residual_calculation, residual_baseline, residual_radius, model_snr, it_time, it_flux, oot_time, oot_flux, residual_model
        else:
            return residual_calculation, residual_baseline, residual_radius, model_snr

    @staticmethod
    def running_mean_equal_length(data, width_signal):
        """Returns the running mean in a given window"""
        cumsum = np.cumsum(np.insert(data, 0, 0))
        med = (cumsum[width_signal:] - cumsum[:-width_signal]) / float(width_signal)
        # Append the first/last value at the beginning/end to match the length of
        # data and returned median
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med)
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front
        med = np.append(np.full(values_front, first_values), med)
        med = np.append(med, np.full(values_end, last_values))
        return med

    @staticmethod
    def spectra(chi2, oversampling_factor=1, kernel_size=30):
        SR = np.min(chi2) / chi2
        SDE_raw = (1 - np.mean(SR)) / np.std(SR)

        # Scale SDE_power from 0 to SDE_raw
        power_raw = SR - np.mean(SR)  # shift down to the mean being zero
        scale = SDE_raw / np.max(power_raw)  # scale factor to touch max=SDE_raw
        power_raw = power_raw * scale

        # Detrended SDE, named "power"
        kernel = oversampling_factor * kernel_size
        if kernel % 2 == 0:
            kernel = kernel + 1
        if len(power_raw) > 2 * kernel:
            my_median = ExoMoonLeastSquares.running_median(power_raw, kernel)
            power = power_raw - my_median
            # Re-normalize to range between median = 0 and peak = SDE
            # shift down to the mean being zero
            power = power - np.mean(power)
            SDE = np.max(power / np.std(power))
            # scale factor to touch max=SDE
            scale = SDE / np.max(power)
            power = power * scale
        else:
            power = power_raw
            SDE = SDE_raw

        return SR, power_raw, power, SDE_raw, SDE

    @staticmethod
    #@njit(fastmath=True, parallel=False)
    def calculate_residuals(it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                            t0_flux_mean_value, datapoints, model_sample, min_radius, max_radius,
                            star_radius, radius_grid_size, ootr_weight=0.5):
        # TODO adjusting model to minimum flux value this might get improved by several scalations of min_flux
        model_baseline = np.full(len(model_sample), 1)
        oot_radius_residuals = np.sum(((oot_flux - 1) / oot_flux_err) ** 2)
        residuals_baseline = np.sum(((it_flux - model_baseline) / it_flux_err) ** 2)  + ootr_weight * oot_radius_residuals
        depth_center = 1 - t0_flux_mean_value
        min_depth = ((min_radius * R_earth_to_R_sun) ** 2) / star_radius ** 2
        best_radius = min_radius
        best_model = model_sample
        best_residual = ExoMoonLeastSquares.MAX_RESIDUAL
        if t0_flux_mean_value < 1 - min_depth:
            scale_factor = (t0_flux_mean_value - 1) / (np.min(model_sample) - 1)
            model_sample_scaled = 1 + (model_sample - 1) * scale_factor
            it_radius_residuals = np.sum(((it_flux - model_sample_scaled) / it_flux_err) ** 2)
            radius_residuals = it_radius_residuals + oot_radius_residuals #/ residuals_baseline
            radius = np.sqrt(1 - t0_flux_mean_value) * star_radius / R_earth_to_R_sun
            best_residual = radius_residuals #/ datapoints
            best_radius = radius
            best_model = model_sample_scaled
        # Old algorithm trying several transits
        # for radius in np.linspace(min_radius, max_radius, radius_grid_size):
        #     depth = ((radius * R_earth_to_R_sun) ** 2) / star_radius ** 2
        #     flux_at_middle = 1 - depth
        #     flux_mean = np.nanmean(flux)
        #     if flux_mean > 1 - min_depth:
        #         radius_residuals = datapoints
        #     else:
        #         scale_factor = (flux_at_middle - 1) / (np.min(model_sample) - 1)
        #         model_sample_scaled = 1 + (model_sample - 1) * scale_factor
        #         radius_residuals = np.sum((flux - model_sample_scaled)) ** 2 * flux_err
        #         radius_residuals = radius_residuals * min_radius / radius
        #     if radius_residuals < best_residual:
        #         best_residual = radius_residuals
        #         best_radius = radius
        #         best_model = model_sample_scaled

        # fig_transit, axs = plt.subplots(1, 1, figsize=(8, 8))
        # axs.scatter(time, flux, color='gray', alpha=0.4, rasterized=True, label="Flux Transit ")
        # axs.plot(time, best_model, color='red', alpha=1, rasterized=True, label="Flux Transit ")
        # axs.set_title("Residuals")
        # axs.set_xlabel('Time')
        # axs.set_ylabel('Flux')
        # bin_means, bin_edges, binnumber = stats.binned_statistic(time,
        #                                                          flux,
        #                                                          statistic='mean', bins=25)
        # bin_stds, _, _ = stats.binned_statistic(time,
        #                                         flux, statistic='std', bins=25)
        # bin_width = (bin_edges[1] - bin_edges[0])
        # bin_centers = bin_edges[1:] - bin_width / 2
        # axs.errorbar(bin_centers, bin_means, yerr=bin_stds / 2, xerr=bin_width / 2, marker='o', markersize=4,
        #              color='darkorange', alpha=1, linestyle='none')
        # fig_transit.show()
        return best_residual, residuals_baseline, best_radius, best_model

    @staticmethod
    def running_median(data, kernel):
        """Returns sliding median of width 'kernel' and same length as data """
        idx = np.arange(kernel) + np.arange(len(data) - kernel + 1)[:, None]
        med = np.median(data[idx], axis=1)

        # Append the first/last value at the beginning/end to match the length of
        # data and returned median
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med)
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front
        med = np.append(np.full(values_front, first_values), med)
        med = np.append(med, np.full(values_end, last_values))
        return med

    @staticmethod
    def remove_outliers(
        flux, sigma=5.0, sigma_lower=None, sigma_upper=None, **kwargs
    ):
        # The import time for `sigma_clip` is somehow very slow, so we use
        # a local import here.
        from astropy.stats.sigma_clipping import sigma_clip

        # First, we create the outlier mask using AstroPy's sigma_clip function
        with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
            warnings.simplefilter("ignore")
            outlier_mask = sigma_clip(
                data=flux,
                sigma=sigma,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                **kwargs,
            ).mask
        # Second, we return the masked light curve and optionally the mask itself
        return outlier_mask

    def is_skip_t0(self, time):
        if len(time) < self.min_cadences_per_t0:
            return True
        return False

    def inject_moon(self, time, flux, flux_err, t0s, planet_semimajor_axis, planet_ecc, moon_radius,
                    moon_period, moon_mass, initial_alpha=0, plot=False, inject_planet=False, inject_moon=True):
        logging.info("Injecting moon with radius of  %.2fR_e, %.2fdays and %.2frad", moon_radius, moon_period, initial_alpha)
        star_radius_m = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.m)
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        params = pandora.model_params()
        params.R_star = star_radius_m  # [m]
        params.u1 = self.ab[0]
        params.u2 = self.ab[1]
        # planet parameters
        params.per_bary = self.planet_period  # [days]
        params.a_bary = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun,
                                                                                     u.au)  # [R_star]
        params.r_planet = self.planet_radius / star_radius_rearth  # [R_star]
        params.b_bary = self.planet_impact_param  # [0..1]
        params.t0_bary = self.planet_t0  # [days]
        params.t0_bary_offset = 0.001  # [days]
        params.M_planet = LcbuilderHelper.convert_from_to(self.planet_mass, u.M_earth, u.kg)  # [kg]
        params.w_bary = self.planet_arg_periastron  # [deg]
        params.ecc_bary = self.planet_ecc  # [0..1]
        # moon parameters
        params.r_moon = moon_radius / star_radius_rearth  # [R_star]
        params.per_moon = moon_period  # [days]
        params.tau_moon = 0  # [0..1]
        params.Omega_moon = 0  # [0..180]
        params.i_moon = self.planet_inc  # [0..180]
        params.e_moon = 0  # [0..1]
        params.w_moon = self.planet_arg_periastron  # [deg]
        params.M_moon = LcbuilderHelper.convert_from_to(moon_mass, u.M_earth, u.kg)  # [kg]
        # times params
        # params.epochs = 3  # [int]
        # params.epoch_duration = 2  # [days]
        # params.cadences_per_day = 48  # [int]
        params.epoch_distance = self.planet_period  # [days]
        params.supersampling_factor = 1  # [int]
        params.occult_small_threshold = 0.1  # [0..1]
        params.hill_sphere_threshold = 1
        model = pandora.moon_model(params)
        injected_flux_total, injected_flux_planet, injected_flux_moon = model.light_curve(time)
        # model.video(
        #     time=pandora_time,
        #     limb_darkening=True,
        #     teff=3200,
        #     planet_color="black",
        #     moon_color="black",
        #     ld_circles=100
        # )
        if inject_moon:
            flux = flux - (1 - injected_flux_moon)
        if inject_planet:
            flux = flux - (1 - injected_flux_planet)
        moon_semimajor_axis = self.compute_semimajor_axis(planet_mass * M_earth_to_M_sun, moon_period)
        moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                              planet_semimajor_axis * AU_TO_RSUN,
                                                              self.planet_period,
                                                              moon_radius * R_earth_to_R_sun, self.planet_impact_param)
        moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                    planet_semimajor_axis * AU_TO_RSUN,
                                                                    self.planet_period,
                                                                    moon_semimajor_axis * AU_TO_RSUN, self.planet_impact_param)
        _, subtracted_planet_flux, _ = self.subtract_planet_transit(time, flux, flux_err)
        for t0 in t0s:
            max_moon_t0_tau = ExoMoonLeastSquares.compute_moon_ts(t0, moon_orbit_transit_duration / 2, moon_transit_duration / 2)
            moon_t0, moon_phase = ExoMoonLeastSquares.compute_moon_t0(self.planet_t0, t0, moon_period, initial_alpha, max_moon_t0_tau)
            #time_transit_mask = (moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)
            time_transit_mask = (t0 - moon_orbit_transit_duration < time) & (time < t0 + moon_orbit_transit_duration)
            time_transit = time[time_transit_mask]
            ma = batman.TransitParams()
            ma.t0 = moon_t0  # time of inferior conjunction
            ma.per = self.planet_period  # orbital period, use Earth as a reference
            ma.rp = moon_radius * R_earth_to_R_sun / self.star_radius  # planet radius (in units of stellar radii)
            ma.a = planet_semimajor_axis * AU_TO_RSUN / self.star_radius  # semi-major axis (in units of stellar radii)
            ma.inc = self.planet_inc  # orbital inclination (in degrees)
            ma.ecc = planet_ecc  # eccentricity
            ma.w = 0  # longitude of periastron (in degrees)
            ma.u = self.ab  # limb darkening coefficients
            ma.limb_dark = "quadratic"  # limb darkening model
            m = batman.TransitModel(ma, time_transit)  # initializes model
            model_moon = m.light_curve(ma)  # calculates light curve
            # flux[(moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)] = \
            #     flux[(moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)] - (1 - model_moon)
            subtracted_planet_flux[time_transit_mask] = \
                subtracted_planet_flux[time_transit_mask] - (1 - model_moon)
            if plot:
                fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
                axs[0].scatter(time_transit, flux[time_transit_mask],
                               color='gray', alpha=1, rasterized=True, label="Flux")
                axs[0].set_title(
                    "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
                        moon_phase))
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Flux')
                axs[0].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
                axs[0].plot(time_transit, injected_flux_moon[time_transit_mask],
                            color='orange', alpha=1, rasterized=True, label="Pandora Model")
                axs[1].scatter(time_transit, flux[time_transit_mask],
                               color='gray', alpha=1, rasterized=True, label="Flux")
                axs[1].set_title(
                    "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
                        moon_phase))
                axs[1].set_xlabel('Time')
                axs[1].set_ylabel('Flux')
                axs[1].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
                axs[1].plot(time_transit, injected_flux_moon[time_transit_mask],
                            color='orange', alpha=1, rasterized=True, label="Pandora Model")
                axs[2].scatter(time_transit, subtracted_planet_flux[time_transit_mask])
                axs[2].set_title(
                    "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
                        moon_phase))
                axs[2].set_xlabel('Time')
                axs[2].set_ylabel('Flux')
                axs[2].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
                axs[2].plot(time_transit, injected_flux_moon[time_transit_mask],
                            color='orange', alpha=1, rasterized=True, label="Pandora Model")
                plt.savefig(self.object_dir + "/T0_" + str(round(t0, 3)) + "_moon.png")
                plt.close()
                time_transit_mask = (moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)
                time_transit = time[time_transit_mask]
                # limb darkening model
                m = batman.TransitModel(ma, time_transit)  # initializes model
                model_moon = m.light_curve(ma)
                fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
                axs[0].scatter(time_transit, flux[time_transit_mask],
                               color='gray', alpha=1, rasterized=True, label="Flux")
                axs[0].set_title(
                    "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
                        moon_phase))
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Flux')
                axs[0].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
                axs[0].plot(time_transit, injected_flux_moon[time_transit_mask],
                            color='orange', alpha=1, rasterized=True, label="Pandora Model")
                axs[1].scatter(time_transit, flux[time_transit_mask],
                               color='gray', alpha=1, rasterized=True, label="Flux")
                axs[1].set_title(
                    "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
                        moon_phase))
                axs[1].set_xlabel('Time')
                axs[1].set_ylabel('Flux')
                axs[1].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
                axs[1].plot(time_transit, injected_flux_moon[time_transit_mask],
                            color='orange', alpha=1, rasterized=True, label="Pandora Model")
                axs[2].scatter(time_transit, subtracted_planet_flux[time_transit_mask])
                axs[2].set_title(
                    "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
                        moon_phase))
                axs[2].set_xlabel('Time')
                axs[2].set_ylabel('Flux')
                axs[2].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
                axs[2].plot(time_transit, injected_flux_moon[time_transit_mask],
                            color='orange', alpha=1, rasterized=True, label="Pandora Model")
                plt.savefig(self.object_dir + "/T0_" + str(round(t0, 3)) + "_moon_focus.png")
                plt.close()
        return flux

    def remove_non_transit_flux(self, time, flux, flux_err, t0s, max_planet_mass):
        max_period = self.au_to_period(max_planet_mass * M_earth_to_M_sun, self.compute_hill_radius(self.star_mass, max_planet_mass * M_earth_to_M_sun, self.planet_semimajor_axis))
        moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, max_period)
        moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                              self.planet_semimajor_axis * AU_TO_RSUN,
                                                              self.planet_period,
                                                              moon_semimajor_axis * AU_TO_RSUN)
        flux_mask = []
        for t0 in t0s:
            flux_mask = np.concatenate((flux_mask, np.argwhere(
                (time > t0 - moon_orbit_transit_duration) & (time < t0 + moon_orbit_transit_duration)).flatten()))
        flux_mask = flux_mask.astype(int)
        time = time[flux_mask]
        flux = flux[flux_mask]
        flux_err = flux_err[flux_mask]
        return time, flux, flux_err

    @staticmethod
    def depth_to_radius(depth, star_radius):
        return np.sqrt(depth * ((star_radius / R_earth_to_R_sun) ** 2))

    def plot_signal(self, axs, scenario_time, scenario_flux, it_time, oot_time, model):
        bin_means, bin_edges, binnumber = stats.binned_statistic(scenario_time,
                                                                 scenario_flux,
                                                                 statistic='mean', bins=60)
        bin_stds, _, _ = stats.binned_statistic(scenario_time,
                                                scenario_flux, statistic='std', bins=60)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2
        axs.scatter(scenario_time, scenario_flux,
                       color='gray', alpha=0.4, rasterized=True, label="Flux")
        axs.errorbar(bin_centers, bin_means, yerr=bin_stds / 2, xerr=bin_width / 2, marker='o', markersize=4,
                        color='darkorange', alpha=1, linestyle='none')
        axs.scatter(it_time, model,
                       color='red', alpha=1, rasterized=True, label="Model")
        axs.scatter(oot_time, np.ones(len(oot_time)),
                       color='red', alpha=1, rasterized=True, label="Model")
        axs.set_xlabel('Time')
        axs.set_ylabel('Flux')

    def run(self, plot=False):
        planet_mass_grid = self.planet_mass_grid
        moon_inc_grid = self.moon_inc_grid
        moon_ecc_grid = self.moon_ecc_grid
        moon_arg_periastron_grid = self.moon_arg_periastron_grid
        self.time, self.flux, self.flux_err = self.remove_non_transit_flux(self.time, self.flux, self.flux_err, self.bary_t0s, np.max(planet_mass_grid))
        rms = np.std(self.flux)
        self.time, self.flux, _ = self.subtract_planet_transit(self.time, self.flux, self.flux_err, plot=plot)
        ma = batman.TransitParams()
        ma.t0 = 10  # time of inferior conjunction
        ma.per = self.planet_period  # orbital period, use Earth as a reference
        ma.rp = self.min_radius * R_earth_to_R_sun / self.star_radius  # planet radius (in units of stellar radii)
        ma.a = self.planet_semimajor_axis * AU_TO_RSUN / self.star_radius  # semi-major axis (in units of stellar radii)
        ma.inc = self.planet_inc  # orbital inclination (in degrees)
        ma.ecc = planet_ecc  # eccentricity
        ma.w = 0  # longitude of periastron (in degrees)
        ma.u = self.ab  # limb darkening coefficients
        ma.limb_dark = "quadratic"  # limb darkening model
        time_model = np.linspace(ma.t0 - planet_duration * 0.7, ma.t0 + planet_duration * 0.7, 1000)
        m = batman.TransitModel(ma, time_model)  # initializes model
        self.model = m.light_curve(ma)
        self.model = self.model[self.model < 1]
        # baseline_model = np.full(len(model), 1)
        # model = np.append(baseline_model, model)
        # model = np.append(model, baseline_model)
        search_inputs = []
        alpha_grid_size = 25
        alpha_grid = np.linspace(0, np.pi * 2 - np.pi * 2 / 25, alpha_grid_size)
        #alpha_grid = [0.0]
        impact_params_grid = [self.planet_impact_param]
        for planet_mass in planet_mass_grid:
            # CITE Édouard Roche (1849, Académie des Sciences de Montpellier) for rigid body
            # planet_volume_m3 = (4/3) * np.pi * ((LcbuilderHelper.convert_from_to(self.planet_radius, u.R_earth, u.m)) ** 3)
            # planet_mass_kg = LcbuilderHelper.convert_from_to(self.planet_mass, u.M_earth, u.kg)
            # planet_density = planet_mass_kg / planet_volume_m3
            # roche_lobe = 2.44 * self.planet_radius * (planet_density / self.max_moon_density) ** (1/3)
            min_period = ExoMoonLeastSquares.compute_roche_period(self.max_moon_density)
            logging.info(f"Roche Limit Period: {min_period}")
            max_period = self.au_to_period(planet_mass * M_earth_to_M_sun, self.compute_hill_radius(self.star_mass, planet_mass * M_earth_to_M_sun, self.planet_semimajor_axis))
            logging.info(f"Hill Limit Period: {max_period}")
            period_grid = self.compute_moon_period_grid(min_period, max_period, samples=self.period_grid_size, mode="log")
            #period_grid = [2.5, 50]
            moon_max_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, max_period)
            for moon_inc in moon_inc_grid:
                for moon_ecc in moon_ecc_grid:
                    for moon_arg_periastron in moon_arg_periastron_grid:
                        for impact_param in impact_params_grid:
                            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                                self.planet_semimajor_axis * AU_TO_RSUN,
                                                                                self.planet_period,
                                                                                self.max_radius * R_earth_to_R_sun,
                                                                                impact_param)
                            half_moon_transit_duration = moon_transit_duration / 2
                            moon_max_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                                            self.planet_period,
                                                                                            moon_max_semimajor_axis * AU_TO_RSUN,
                                                                                            impact_param)
                            half_moon_max_orbit_transit_duration = moon_max_orbit_transit_duration / 2
                            for moon_period in period_grid:
                                moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun,
                                                                                  moon_period)
                                moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                                            self.planet_period,
                                                                                            moon_semimajor_axis * AU_TO_RSUN,
                                                                                            impact_param)
                                half_moon_orbit_transit_duration = moon_orbit_transit_duration / 2
                                for moon_initial_alpha in alpha_grid:
                                    #TODO moon_orbit_ranges should use moon_radius ?
                                    search_inputs.append(SearchInput(
                                        moon_period, moon_initial_alpha, moon_ecc, moon_inc, moon_arg_periastron,
                                        self.min_radius, impact_param, moon_max_orbit_transit_duration,
                                        moon_transit_duration, moon_orbit_transit_duration, half_moon_transit_duration,
                                        half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration))
        with Pool(processes=self.cpus) as pool:
            all_residuals = list(
                tqdm(pool.imap(self.search, search_inputs),
                     total=len(search_inputs))
            )
        best_residuals_per_scenarios = []
        best_residuals_spectra = []
        best_residuals_spectra_values = []
        best_residual_per_period = np.inf
        best_residual_per_period_values = []
        best_alpha_per_period = []
        best_snr_per_period = []
        residuals_matrix = np.full((alpha_grid_size, len(period_grid)), np.nan)
        max_residual = 0
        for i in np.arange(0, len(search_inputs)):
            if i % alpha_grid_size == 0 and i != 0:
                best_residuals_spectra = best_residuals_spectra + [best_residual_per_period]
                best_residuals_spectra_values = best_residuals_spectra_values + best_residual_per_period_values
                best_residual_per_period = np.inf
                best_residual_per_period_values = []
            all_residual = all_residuals[i]
            residuals = all_residual[0]
            residuals_baseline = all_residual[1]
            radius = all_residual[2]
            model_snr = all_residual[3]
            moon_period = search_inputs[i].moon_period
            moon_initial_alpha = search_inputs[i].moon_alpha
            corrected_residual = residuals #/ residuals_baseline * self.min_radius / radius
            if corrected_residual > max_residual and corrected_residual < ExoMoonLeastSquares.MAX_RESIDUAL / 2:
                max_residual = corrected_residual
            best_residuals_per_scenarios.append(
                [moon_period, moon_initial_alpha, corrected_residual, radius])
            residuals_matrix[i % alpha_grid_size][
                i // alpha_grid_size] = corrected_residual
            if i % alpha_grid_size == 0:
                best_alpha_per_period.append([moon_initial_alpha, corrected_residual])
            elif best_alpha_per_period[i // alpha_grid_size][1] > corrected_residual:
                best_alpha_per_period[i // alpha_grid_size] = [moon_initial_alpha, corrected_residual]
            if best_residual_per_period > corrected_residual:
                best_residual_per_period = corrected_residual
                best_residual_per_period_values = [residuals_baseline, radius, moon_period, moon_initial_alpha]
            if i % alpha_grid_size == 0:
                best_snr_per_period.append([moon_initial_alpha, model_snr])
            elif best_snr_per_period[i // alpha_grid_size][1] < model_snr:
                best_snr_per_period[i // alpha_grid_size] = [moon_initial_alpha, model_snr]
        best_residuals_spectra = best_residuals_spectra + [best_residual_per_period]
        best_residuals_spectra = np.array(best_residuals_spectra)
        best_residuals_spectra[np.isnan(best_residuals_spectra)] = np.nanmax(best_residuals_spectra)
        best_residuals_spectra_values = best_residuals_spectra_values + best_residual_per_period_values
        residuals_matrix[np.isnan(residuals_matrix)] = np.nanmax(residuals_matrix)
        residuals_matrix[residuals_matrix > ExoMoonLeastSquares.MAX_RESIDUAL / 2] = max_residual
        best_residuals_spectra[np.isnan(best_residuals_spectra)] = np.nanmax(best_residuals_spectra)
        best_residuals_per_scenarios = np.array(best_residuals_per_scenarios)
        transposed_best_residuals_per_scenarios = np.transpose(best_residuals_per_scenarios)
        residuals_per_scenario_baseline_args = np.argwhere(transposed_best_residuals_per_scenarios[2] > ExoMoonLeastSquares.MAX_RESIDUAL / 2 ).flatten()
        transposed_best_residuals_per_scenarios[2][residuals_per_scenario_baseline_args] = max_residual
        best_residuals_per_scenarios = np.transpose(transposed_best_residuals_per_scenarios)
        best_residuals_per_scenarios_plot = best_residuals_per_scenarios[np.argsort(np.array([best_residual_per_scenarios[2] for best_residual_per_scenarios in best_residuals_per_scenarios]).flatten())]
        fig_transit, axs = plt.subplots(5, 1, figsize=(20, 35))
        axs[0].plot(period_grid, best_residuals_spectra, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
        axs[0].set_title("Residuals Raw")
        axs[0].set_xlabel('P (days)')
        axs[0].set_ylabel('Residuals')
        residuals_per_scenario_baseline_args = np.argwhere(best_residuals_spectra > ExoMoonLeastSquares.MAX_RESIDUAL / 2).flatten()
        best_residuals_spectra[residuals_per_scenario_baseline_args] = max_residual
        axs[1].plot(period_grid, best_residuals_spectra, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
        axs[1].set_title("Residuals Reduced")
        axs[1].set_xlabel('P (days)')
        axs[1].set_ylabel('Residuals')
        axs[2].plot(period_grid, np.transpose(best_snr_per_period)[1])
        axs[2].set_title("SNRs")
        axs[2].set_xlabel('P (days)')
        axs[2].set_ylabel('SNR')
        kernel = int(np.sqrt(len(period_grid)))
        kernel = kernel if kernel % 2 != 0 else kernel + 1
        kernel = 31
        SR, power_raw, power, SDE_raw, SDE = ExoMoonLeastSquares.spectra(best_residuals_spectra, kernel)
        axs[3].plot(period_grid, power, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
        axs[3].set_title("SDE")
        axs[3].set_xlabel('P (days)')
        axs[3].set_ylabel('Power')
        plt.savefig(self.object_dir + "/stats.png")
        plt.close(fig_transit)
        residuals_score = np.nanmax(residuals_matrix) / residuals_matrix - 1
        plt.imshow(residuals_score, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.savefig(self.object_dir + "/residuals_map.png")
        plt.close()
        for i in np.arange(0, 15):
            moon_period = best_residuals_per_scenarios_plot[i][0]
            moon_initial_alpha = best_residuals_per_scenarios_plot[i][1]
            moon_max_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, period_grid[-1])
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.planet_semimajor_axis * AU_TO_RSUN,
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            moon_max_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_max_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, moon_period)
            moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                SearchInput(moon_period, moon_initial_alpha,
                            0, 90, 0, self.min_radius, self.planet_impact_param,
                            moon_max_orbit_transit_duration,
                            moon_transit_duration, moon_orbit_transit_duration, moon_transit_duration / 2,
                            moon_orbit_transit_duration / 2, moon_max_orbit_transit_duration / 2),
                return_lc=True)
            moon_depth = 1 - np.min(model)
            moon_radius = ExoMoonLeastSquares.depth_to_radius(moon_depth, self.star_radius)
            moon_mass = moon_radius ** 1.5
            if i == 0:
                self.inject_moon(self.time, self.flux, self.flux_err, self.bary_t0s, self.planet_semimajor_axis,
                                 self.planet_ecc, moon_radius,
                                 moon_period, moon_mass, initial_alpha=moon_initial_alpha, plot=True,
                                 inject_planet=False, inject_moon=False)
            logging.info("Best residual for period %s, alpha %s: Residual->%s, Radius->%s, Depth->%s, SNR->%s",
                         moon_period,
                         moon_initial_alpha, best_residuals_per_scenarios_plot[i][2],
                         moon_radius, moon_depth, model_snr)
            scenario_time = np.append(it_time, oot_time)
            scenario_flux = np.append(it_flux, oot_flux)
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            fig_transit.suptitle(f"Period {moon_period}, alpha {moon_initial_alpha}\n"
                f"Residual->{best_residuals_per_scenarios_plot[i][2]}, Radius->{moon_radius}, Depth->{moon_depth}, "
                f"SNR->{model_snr}")
            self.plot_signal(axs[0], scenario_time, scenario_flux, it_time, oot_time, model)
            args_in_moon_transit_duration = np.argwhere(np.abs(oot_time) < moon_transit_duration).flatten()
            scenario_time = np.append(it_time, oot_time[args_in_moon_transit_duration])
            scenario_flux = np.append(it_flux, oot_flux[args_in_moon_transit_duration])
            self.plot_signal(axs[1], scenario_time, scenario_flux, it_time, oot_time[args_in_moon_transit_duration], model)
            plt.savefig(self.object_dir + "/Residuals " + str(i) + ".png")
            plt.close(fig_transit)
        top_sde_args = np.flip(np.argsort(power))
        best_residuals_by_sde = best_residuals_spectra[top_sde_args]
        for i, top_sde_index in enumerate(top_sde_args[0:15]):
            moon_period = period_grid[top_sde_index]
            moon_initial_alpha = best_alpha_per_period[top_sde_index][0]
            sde = best_residuals_by_sde[top_sde_index]
            moon_max_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, period_grid[-1])
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.planet_semimajor_axis * AU_TO_RSUN,
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            moon_max_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_max_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, moon_period)
            moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                SearchInput(moon_period, moon_initial_alpha,
                            0, 90, 0, self.min_radius, self.planet_impact_param,
                            moon_max_orbit_transit_duration,
                            moon_transit_duration, moon_orbit_transit_duration, moon_transit_duration / 2,
                            moon_orbit_transit_duration / 2, moon_max_orbit_transit_duration / 2),
                return_lc=True)
            scenario_time = np.append(it_time, oot_time)
            scenario_flux = np.append(it_flux, oot_flux)
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            moon_depth = 1 - np.min(model)
            moon_radius = ExoMoonLeastSquares.depth_to_radius(moon_depth, self.star_radius)
            fig_transit.suptitle(f"Period {moon_period}, alpha {moon_initial_alpha}\n"
                f"SDE->{sde}, Radius->{moon_radius}, Depth->{moon_depth}, "
                f"SNR->{model_snr}")
            self.plot_signal(axs[0], scenario_time, scenario_flux, it_time, oot_time, model)
            args_in_moon_transit_duration = np.argwhere(np.abs(oot_time) < moon_transit_duration).flatten()
            scenario_time = np.append(it_time, oot_time[args_in_moon_transit_duration])
            scenario_flux = np.append(it_flux, oot_flux[args_in_moon_transit_duration])
            self.plot_signal(axs[1], scenario_time, scenario_flux, it_time, oot_time[args_in_moon_transit_duration], model)
            plt.savefig(self.object_dir + "/SDE " + str(i) + ".png")
            plt.close(fig_transit)
        best_scenario_per_snr_args = np.argsort(np.transpose(best_snr_per_period)[1])
        for i in np.arange(0, 15):
            index = best_scenario_per_snr_args[-1 -i]
            moon_period = period_grid[index]
            moon_initial_alpha = best_snr_per_period[index][0]
            moon_max_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, period_grid[-1])
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.planet_semimajor_axis * AU_TO_RSUN,
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            moon_max_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_max_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, moon_period)
            moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                SearchInput(moon_period, moon_initial_alpha,
                            0, 90, 0, self.min_radius, self.planet_impact_param,
                            moon_max_orbit_transit_duration,
                            moon_transit_duration, moon_orbit_transit_duration, moon_transit_duration / 2,
                            moon_orbit_transit_duration / 2, moon_max_orbit_transit_duration / 2),
                return_lc=True)
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            moon_depth = 1 - np.min(model)
            moon_radius = ExoMoonLeastSquares.depth_to_radius(moon_depth, self.star_radius)
            logging.info("Best SNR for period %s, alpha %s: Radius->%s, Depth->%s, SNR->%s",
                         moon_period,
                         moon_initial_alpha,
                         moon_radius, moon_depth, model_snr)
            scenario_time = np.append(it_time, oot_time)
            scenario_flux = np.append(it_flux, oot_flux)
            self.plot_signal(axs[0], scenario_time, scenario_flux, it_time, oot_time, model)
            fig_transit.suptitle(
                f"Period {moon_period}, alpha {moon_initial_alpha}\n"
                f"Radius->{moon_radius}, Depth->{moon_depth}, "
                f"SNR->{model_snr}")
            args_in_moon_transit_duration = np.argwhere(np.abs(oot_time) < moon_transit_duration).flatten()
            scenario_time = np.append(it_time, oot_time[args_in_moon_transit_duration])
            scenario_flux = np.append(it_flux, oot_flux[args_in_moon_transit_duration])
            self.plot_signal(axs[1], scenario_time, scenario_flux, it_time, oot_time[args_in_moon_transit_duration], model)
            plt.savefig(self.object_dir + "/SNR " + str(i) + ".png")
            plt.close(fig_transit)


class SearchInput:
    def __init__(self, moon_period, moon_alpha, moon_ecc, moon_inc, moon_arg_periastron, min_radius, impact_param,
                 moon_max_orbit_transit_duration, moon_transit_duration, moon_orbit_transit_duration,
                 half_moon_transit_duration, half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration) -> None:
        self.moon_period = moon_period
        self.moon_alpha = moon_alpha
        self.moon_ecc = moon_ecc
        self.moon_inc = moon_inc
        self.moon_arg_periastron = moon_arg_periastron
        self.min_radius = min_radius
        self.impact_param = impact_param
        self.moon_max_orbit_transit_duration = moon_max_orbit_transit_duration
        self.moon_transit_duration = moon_transit_duration
        self.moon_orbit_transit_duration = moon_orbit_transit_duration
        self.half_moon_transit_duration = half_moon_transit_duration
        self.half_moon_orbit_transit_duration = half_moon_orbit_transit_duration
        self.half_moon_max_orbit_transit_duration = half_moon_max_orbit_transit_duration


formatter = logging.Formatter('%(message)s')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


lc_builder = LcBuilder()
max_moon_density = 5.5
tois_df = pd.read_csv("/home/martin/.sherlockpipe/tois.csv")
confirmed_tois_df = tois_df.loc[((tois_df['Disposition'] == 'CP') | (tois_df['Disposition'] == 'KP')) & (tois_df['Period (days)'] > 50)]
confirmed_tois_df = confirmed_tois_df.sort_values(by='Planet SNR', ascending=False)
#confirmed_tois_df = confirmed_tois_df.loc[confirmed_tois_df['Object Id'] == 'TIC 309792357']
confirmed_tois_df = confirmed_tois_df.iloc[0:5]
for index, row in confirmed_tois_df.iterrows():
    target_name = row['Object Id']
    object_dir = target_name + "_EMLS/"
    if not os.path.exists(object_dir):
        os.mkdir(object_dir)
    object_info = lc_builder.build_object_info(target_name=target_name, author=['SPOC'], sectors="all", file=None, cadence=[120],
                                  initial_mask=[[2198, 2202]], initial_transit_mask=None, star_info=None, aperture=None,
                                  eleanor_corr_flux="pdcsap_flux", outliers_sigma=3, high_rms_enabled=False,
                                  high_rms_threshold=1.5, high_rms_bin_hours=4, smooth_enabled=False, binning=0,
                                  auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                                  auto_detrend_period=None, prepare_algorithm=None, reduce_simple_oscillations=False,
                                  oscillation_snr_threshold=4, oscillation_amplitude_threshold=0.1, oscillation_ws_scale=60,
                                  oscillation_min_period=0.002, oscillation_max_period=0.2)
    lc_build = lc_builder.build(object_info, object_dir)
    star_mass = lc_build.star_info.mass
    star_radius = lc_build.star_info.radius
    times = lc_build.lc.time.value
    flux = lc_build.lc.flux.value
    flux_err = lc_build.lc.flux_err.value
    ld_coeffs = lc_build.star_info.ld_coefficients

    companions_df = tois_df.loc[(tois_df['Object Id'] == target_name) & (tois_df['OI'] != row['OI'])]
    for companion_index, companion_row in companions_df.iterrows():
        logging.info(f"Masking companion {companion_row['OI']}")
        companion_t0 = companion_row['Epoch (BJD)'] - 2457000
        companion_period = companion_row['Period (days)']
        companion_duration = companion_row['Duration (hours)'] / 24 * 2
        times, flux, flux_err = LcbuilderHelper.mask_transits(times, flux, companion_period, companion_duration,
                                                             companion_t0, flux_err=flux_err)

    # star_mass = 0.88
    # star_radius = 0.878413
    # times = np.linspace(planet_t0 - 5, planet_t0 + 1000, 1005 * 60 * 24 // 2)
    # flux = np.random.normal(1, 0.001, len(times))
    # flux_err = np.random.normal(0.001, 0.0001, len(times))
    # ld_coeffs = [0.4136, 0.1999]

    planet_radius = row['Planet Radius (R_Earth)']
    planet_radius_err = row['Planet Radius (R_Earth) err']
    planet_period = row['Period (days)']
    planet_t0 = row['Epoch (BJD)'] - 2457000
    planet_t0 = times[0] + (planet_t0 - times[0]) % planet_period
    planet_duration = row['Duration (hours)'] / 24
    planet_inc = 90
    planet_inc_err = 3
    planet_ecc = 0.15
    planet_ecc_err = 0.15
    planet_arg_periastron = 0
    planet_mass = row['Predicted Mass (M_Earth)']
    if np.isnan(planet_mass) or planet_mass is None:
        planet_mass = planet_radius ** 1.5
    planet_semimajor_axis = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period)
    planet_impact_param = LcbuilderHelper.convert_from_to(planet_semimajor_axis, u.au, u.R_sun) / star_radius * np.cos(np.deg2rad(planet_inc))
    min_radius = planet_radius / 10
    max_radius = planet_radius / 5

    # planet_radius = 11.8142656578
    # planet_radius_err = 1.00
    # planet_period = 141.834025
    # planet_duration = 8.681 / 24
    # planet_inc = 89.903
    # planet_inc_err = 0.5
    # planet_ecc = 0.212
    # planet_ecc_err = 0.022
    # planet_arg_periastron = 0
    # planet_mass = 408.727
    # min_radius = 1
    # max_radius = 3

    flux_err = np.ma.filled(flux_err, 0)
    flux = wotan.flatten(times, flux, method="biweight", window_length=planet_duration * 5)
    t0s = [i for i in np.arange(planet_t0, np.max(times), planet_period)]
    emls = ExoMoonLeastSquares(object_dir, 4, star_mass, star_radius, ld_coeffs,
                               planet_radius, planet_radius_err, planet_mass, planet_period, planet_t0, planet_duration,
                               planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc, planet_ecc_err, planet_arg_periastron, planet_impact_param,
                               min_radius, max_radius, t0s, times, flux, flux_err, max_moon_density,
                               period_grid_size=10000, radius_grid_size=20)
    #emls.fit_single_transits(times, flux, flux_err)
    # emls.flux = emls.inject_moon(emls.time, emls.flux, emls.flux_err, emls.bary_t0s, planet_semimajor_axis,
    #                             planet_ecc, moon_radius, moon_period, moon_mass, plot=True, initial_alpha=moon_alpha,
    #                              inject_planet=False, inject_moon=False)
    #emls.fit_single_transits(times, flux, flux_err, plot=True)
    emls.planet_mass_grid = [planet_mass]
    emls.moon_inc_grid = [90]
    emls.moon_ecc_grid = [0]
    emls.moon_arg_periastron_grid = [0]
    emls.run(plot=True)
