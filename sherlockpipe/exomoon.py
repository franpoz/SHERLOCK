import itertools
import logging
import os
import sys
import time
from multiprocessing import Pool
from pytransit import QuadraticModel
import batman
import ellc
import numpy as np
import astropy.constants as ac
import astropy.units as u
import wotan
from lcbuilder.lcbuilder_class import LcBuilder
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


class ExoMoonLeastSquares:
    G = 6.674e-11  # m3 kg-1 s-2
    au_to_Rsun = 215.032
    Msolar_to_kg = 2.e30
    Mearth_to_kg = 5.972e24
    M_earth_to_M_sun = Mearth_to_kg / Msolar_to_kg
    R_earth_to_R_sun = 0.009175

    def __init__(self, cpus, star_mass, star_radius, ab, planet_radius, planet_period, planet_t0, planet_duration, planet_semimajor_axis, planet_inc, planet_ecc,
                 planet_arg_periastron, planet_impact_param, time, flux):
        self.cpus = cpus
        self.star_mass = star_mass
        self.star_radius = star_radius
        self.ab = ab
        self.planet_radius = planet_radius
        self.planet_period = planet_period
        self.planet_t0 = planet_t0
        self.planet_duration = planet_duration
        self.planet_semimajor_axis = planet_semimajor_axis
        self.planet_inc = planet_inc
        self.planet_ecc = planet_ecc
        self.planet_arg_periastron = planet_arg_periastron
        self.planet_impact_param = planet_impact_param
        self.time = time
        self.flux = flux

    @staticmethod
    def compute_semimajor_axis(major_mass, minor_period):
        period_seconds = minor_period * 24. * 3600.
        mass_kg = major_mass * ExoMoonLeastSquares.Msolar_to_kg
        a1 = (ExoMoonLeastSquares.G * mass_kg * period_seconds ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
        return a1 / 1.496e11

    def compute_hill_radius(self, major_mass, minor_mass, semimajor_axis, eccentricity=0):
        """
        @param major_mass: The main body mass
        @param minor_mass: The minor body mass
        @param semimajor_axis: The minor body semimajor axis in AU.
        @param eccentricity: the planet eccentricity
        @return: the hill radius of the minor body in the same units than the semimajor_axis
        """

        return self.au_to_Rsun * semimajor_axis * (1 - eccentricity) * (minor_mass / (3 * major_mass) ** (1 / 3))

    def au_to_period(self, mass, au):
        """
        Calculates the orbital period for the semi-major axis assuming a circular orbit.
        @param mass: the stellar mass
        @param au: the semi-major axis in astronomical units.
        @return: the period in days
        """
        mass_kg = mass * 2.e30
        a = au * 1.496e11
        return ((a ** 3) * 4 * (np.pi ** 2) / self.G / mass_kg) ** (1. / 2.) / 3600 / 24

    def compute_transit_duration(self, star_radius,
                                 transiting_body_semimajor_axis, transit_period, transiting_body_radius,
                                 impact_parameter=0):
        """

        @param star_radius: star radius
        @param transiting_body_semimajor_axis: orbit semimajor axis
        @param transit_period: in days
        @param transiting_body_radius: transiting body radius
        @param impact_parameter:
        @return:
        @rtype:
        """
        return transit_period / np.pi * np.arcsin(np.sqrt((star_radius + transiting_body_radius) ** 2 - (impact_parameter * star_radius) ** 2) / transiting_body_semimajor_axis)
        #return 2 * moon_semimajor_axis / (planet_semimajor_axis * 2 * np.pi) * planet_period

    def compute_moon_period_grid(self, min, max, mode="lin", samples=1000):
        if "log" == mode:
            return np.logspace(min, max, samples)
        else:
            return np.linspace(min, 10, samples)

    def tokenize_transits_moon_orbit_ranges(self, time, flux, star_mass, star_radius, planet_mass, planet_radius,
                                            planet_period, planet_t0, moon_semimajor_axis, planet_semimajor_axis, all_t0s,
                                            moon_period, moon_eccentrictiy=0,
                                            moon_arg_periastron=0, moon_inclination=90,
                                            planet_eccentricity=0, planet_arg_periastron=0, planet_inclination=90):
        moon_orbit_transit_duration = self.compute_transit_duration(star_radius, planet_semimajor_axis * self.au_to_Rsun,
                                                                    planet_period, moon_semimajor_axis * self.au_to_Rsun,
                                                                    self.planet_impact_param)
        # TODO we probably need to define left_transit_length and right_transit_length depending on moon orbit parameters
        moon_orbit_tokens = [[t0, t0 - self.planet_duration / 2, t0 - moon_orbit_transit_duration / 2, t0 + moon_orbit_transit_duration / 2] for t0 in all_t0s]
        return moon_orbit_tokens

    def subtract_planet_transit(self, ab, star_radius, star_mass, time, flux, planet_radius, planet_t0,
                                planet_period, planet_inc=90):
        P1 = planet_period * u.day
        a = np.cbrt((ac.G * star_mass * u.M_sun * P1 ** 2) / (4 * np.pi ** 2)).to(u.au)
        model = ellc.lc(
            t_obs=time,
            radius_1=(star_radius * u.R_sun).to(u.au) / a,  # star radius convert from AU to in units of a
            radius_2=(planet_radius * u.R_earth).to(u.au) / a,
            # convert from Rearth (equatorial) into AU and then into units of a
            sbratio=0,
            incl=planet_inc,
            light_3=0,
            t_zero=planet_t0,
            period=planet_period,
            a=None,
            q=1e-6,
            f_c=None, f_s=None,
            ldc_1=ab, ldc_2=None,
            gdc_1=None, gdc_2=None,
            didt=None,
            domdt=None,
            rotfac_1=1, rotfac_2=1,
            hf_1=1.5, hf_2=1.5,
            bfac_1=None, bfac_2=None,
            heat_1=None, heat_2=None,
            lambda_1=None, lambda_2=None,
            vsini_1=None, vsini_2=None,
            t_exp=None, n_int=None,
            grid_1='default', grid_2='default',
            ld_1='quad', ld_2=None,
            shape_1='sphere', shape_2='sphere',
            spots_1=None, spots_2=None,
            exact_grav=False, verbose=1)
        return flux - model + 1

    def compute_moon_transit_scenarios(self, time, flux, planet_t0, moon_initial_alpha, moon_period, moon_orbit_ranges,
                                       moon_orbit_transit_length, moon_transit_duration):
        #TODO need to take into account "prograde" or "retrograde" orbit
        orbit_scenarios = []
        for moon_orbit_range in moon_orbit_ranges:
            t0 = moon_orbit_range[0]
            t1 = moon_orbit_range[1]
            phase_delta = (t0 - planet_t0) % moon_period / moon_period * 2 * np.pi
            alpha = (moon_initial_alpha + phase_delta) % (2 * np.pi)
            time_alpha = np.cos(alpha) * moon_orbit_transit_length / 2
            moon_t0 = t1 + time_alpha
            time_args = np.argwhere((time > moon_t0 - moon_transit_duration / 2) & (time < moon_t0 + moon_transit_duration / 2))
            #TODO we'd need to fill measurement gaps (detected from the time array)
            time_moon_transit = time[time_args]
            flux_moon_transit = flux[time_args]
            time_moon_transit = time_moon_transit - moon_t0
            if len(time_moon_transit) > 0:
                orbit_scenarios.append([alpha, time_moon_transit, flux_moon_transit])
        return orbit_scenarios

    def search(self, search_input):
        logging.info("Searching for period=%.2fd and alpha=%.2frad", search_input.moon_period, search_input.moon_alpha)
        moon_semimajor_axis = self.compute_semimajor_axis(planet_mass * self.M_earth_to_M_sun, search_input.moon_period)
        moon_orbit_transit_length = self.compute_transit_duration(self.star_radius,
                                                                  self.planet_semimajor_axis * self.au_to_Rsun,
                                                                  self.planet_period,
                                                                  moon_semimajor_axis * self.au_to_Rsun,
                                                                  self.planet_impact_param)
        moon_transit_length = self.compute_transit_duration(self.star_radius, self.planet_semimajor_axis * self.au_to_Rsun,
                                                            self.planet_period, 1 * self.R_earth_to_R_sun,
                                                            self.planet_impact_param)
        moon_orbit_ranges = self.tokenize_transits_moon_orbit_ranges(self.time, self.flux,
                                                                     self.star_mass, self.star_radius, planet_mass,
                                                                     self.planet_radius, self.planet_period,
                                                                     self.planet_t0, moon_semimajor_axis,
                                                                     self.planet_semimajor_axis, search_input.all_t0s,
                                                                     search_input.moon_period, search_input.moon_ecc,
                                                                     search_input.moon_arg_periastron, search_input.moon_inc, self.planet_ecc,
                                                                     self.planet_arg_periastron, self.planet_inc)
        transit_scenarios = self.compute_moon_transit_scenarios(self.time, self.flux, self.planet_t0, search_input.moon_alpha,
                                                                search_input.moon_period, moon_orbit_ranges,
                                                                moon_orbit_transit_length, moon_transit_length)
        t0 = time.time()
        cadence = 2
        R_s = self.star_radius
        M_s = 1.3
        P_min = 0.5
        P_max = 22
        ld_coefficients = [0.2, 0.1]
        duration = self.compute_transit_duration(self.star_radius, self.planet_semimajor_axis * self.au_to_Rsun,
                                                 self.planet_period, 1 * self.R_earth_to_R_sun, self.planet_impact_param)

        duration_grid = [duration]
        curve_rms = np.std(self.flux)
        min_depth = curve_rms / 2
        initial_rp = (min_depth * (R_s ** 2)) ** (1 / 2)
        rp_rs = initial_rp / R_s
        tm = QuadraticModel()
        time_model = np.arange(0, 1, 0.0001)
        tm.set_data(time_model)
        # k is the radius ratio, ldc is the limb darkening coefficient vector, t0 the zero epoch, p the orbital period, a the
        # semi-major axis divided by the stellar radius, i the inclination in radians, e the eccentricity, and w the argument
        # of periastron. Eccentricity and argument of periastron are optional, and omitting them defaults to a circular orbit.
        a_au = self.compute_semimajor_axis(self.star_mass, 0.5)
        a_Rs = a_au / (self.star_radius * 0.00465047)
        model = tm.evaluate(k=rp_rs, ldc=ld_coefficients, t0=0.5, p=1.0, a=a_Rs, i=0.5 * np.pi)
        model = model[model < 1]
        # baseline_model = np.full(len(model), 1)
        # model = np.append(baseline_model, model)
        # model = np.append(model, baseline_model)
        i = 0
        all_residuals = []
        scenario_time = []
        scenario_flux = []
        for normalized_moon_transit_scenario in transit_scenarios:
            scenario_time = np.concatenate((scenario_time, normalized_moon_transit_scenario[1].flatten()))
            scenario_flux = np.concatenate((scenario_flux, normalized_moon_transit_scenario[2].flatten()))
        sorted_time_args = np.argsort(scenario_time)
        scenario_time = scenario_time[sorted_time_args]
        scenario_flux = scenario_flux[sorted_time_args]
        residual_calculation = self.calculate_residuals(scenario_time, scenario_flux, model, 0, duration)
        return residual_calculation, scenario_time, scenario_flux

    def downsample(self, array, npts: int):
        interpolated = interp1d(np.arange(len(array)), array, axis=0, fill_value='extrapolate')
        downsampled = interpolated(np.linspace(0, len(array), npts))
        return downsampled

    def calculate_residuals(self, time, flux, model_sample, flux_index, duration):
        last_time_index = np.argwhere(time[time < time[flux_index] + duration])[-1]
        time_subset = time[flux_index:int(flux_index + last_time_index)]
        model_sample = self.downsample(model_sample, len(time_subset))
        # TODO adjusting model to minimum flux value this might get improved by several scalations of min_flux
        model_sample_scaled = np.copy(model_sample)
        flux_mean = np.mean(flux)
        flux = flux - flux_mean + 1 if flux_mean > 1 else flux + flux_mean - 1
        flux_subset = flux[flux_index:int(flux_index + last_time_index)]
        max_flux = 1 + np.std(flux_subset) / 2
        flux_at_middle = np.mean(flux_subset)
        if flux_at_middle < max_flux:
            model_sample_scaled[model_sample_scaled < 1] = model_sample_scaled[model_sample_scaled < 1] * (
                        flux_at_middle / np.min(model_sample))
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
            return np.sum((flux_subset - model_sample_scaled) ** 2) ** 0.5
        return np.nan

    def inject_moon(self, time, flux, t0s, planet_mass, planet_semimajor_axis, planet_ecc, moon_radius, moon_period, initial_alpha=0):
        logging.info("Injecting moon with radius of  %.2fR_e, %.2fdays and %.2frad", moon_radius, moon_period, initial_alpha)
        moon_semimajor_axis = self.compute_semimajor_axis(planet_mass * self.M_earth_to_M_sun, moon_radius)
        moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                              planet_semimajor_axis * self.au_to_Rsun,
                                                              self.planet_period,
                                                              moon_radius * self.R_earth_to_R_sun,
                                                              self.planet_impact_param)
        moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                              planet_semimajor_axis * self.au_to_Rsun,
                                                              self.planet_period,
                                                              moon_semimajor_axis * self.au_to_Rsun,
                                                              self.planet_impact_param)
        first_t0 = t0s[0]
        for t0 in t0s:
            phase_delta = ((t0 - first_t0) % moon_period / moon_period) * 2 * np.pi
            moon_phase = (initial_alpha + phase_delta) % 2 * np.pi
            moon_tau = np.cos(moon_phase)
            moon_t0 = t0 + moon_tau * moon_orbit_transit_duration / 2
            time_transit = time[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)]
            if len(time_transit) == 0:
                continue
            ma = batman.TransitParams()
            ma.t0 = moon_t0  # time of inferior conjunction
            ma.per = self.planet_period  # orbital period, use Earth as a reference
            ma.rp = moon_radius * self.R_earth_to_R_sun / self.star_radius  # planet radius (in units of stellar radii)
            ma.a = planet_semimajor_axis * self.au_to_Rsun / self.star_radius  # semi-major axis (in units of stellar radii)
            ma.inc = 90  # orbital inclination (in degrees)
            ma.ecc = planet_ecc  # eccentricity
            ma.w = 0  # longitude of periastron (in degrees)
            ma.u = self.ab  # limb darkening coefficients
            ma.limb_dark = "quadratic"  # limb darkening model
            m = batman.TransitModel(ma, time_transit)  # initializes model
            model = m.light_curve(ma)  # calculates light curve
            fig_transit, axs = plt.subplots(2, 1, figsize=(12, 10))
            axs[0].plot(time_transit, flux[
                (moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)],
                        color='gray', alpha=1, rasterized=True, label="Flux")
            axs[0].set_title(
                "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(moon_phase))
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Flux')
            axs[0].plot(time_transit, model, color='red', alpha=1, rasterized=True, label="Model")
            flux[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)] = \
                flux[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)] + model - 1
            axs[1].plot(time_transit, flux[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)], color='gray', alpha=1, rasterized=True, label="Flux")
            axs[1].set_title("Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(moon_phase))
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Flux')
            axs[1].plot(time_transit, model, color='red', alpha=1, rasterized=True, label="Model")
            fig_transit.show()
        return flux

    def run(self):
        planet_mass_grid = self.planet_mass_grid
        moon_inc_grid = self.moon_inc_grid
        moon_ecc_grid = self.moon_ecc_grid
        moon_arg_periastron_grid = self.moon_arg_periastron_grid
        self.flux = self.subtract_planet_transit(self.ab, self.star_radius, self.star_mass, self.time, self.flux,
                                            self.planet_radius, self.planet_t0, self.planet_period, self.planet_inc)
        all_t0s = [i for i in np.arange(planet_t0, np.max(times), planet_period)]
        search_inputs = []
        for planet_mass in planet_mass_grid:
            min_period = 0.5 # TODO compute this value somehow
            max_period = self.au_to_period(planet_mass * self.M_earth_to_M_sun, self.compute_hill_radius(self.star_mass, planet_mass * self.M_earth_to_M_sun, self.planet_semimajor_axis))
            period_grid = self.compute_moon_period_grid(min_period, max_period)
            for moon_inc in moon_inc_grid:
                for moon_ecc in moon_ecc_grid:
                    for moon_arg_periastron in moon_arg_periastron_grid:
                        for moon_period in period_grid:
                            for moon_initial_alpha in np.linspace(0, np.pi * 2 - np.pi * 2 / 25, 25):
                                #TODO moon_orbit_ranges should use moon_radius ?
                                search_inputs.append(SearchInput(moon_period, moon_initial_alpha, moon_ecc, moon_inc, moon_arg_periastron, all_t0s))
        with Pool(processes=self.cpus) as pool:
            all_residuals = pool.map(self.search, search_inputs)
        best_residuals_per_scenarios = []
        for i in np.arange(0, len(search_inputs)):
            all_residual = all_residuals[i]
            residuals = all_residual[0]
            scenario_time = all_residual[1]
            scenario_flux = all_residual[2]
            moon_period = search_inputs[i].moon_period
            moon_initial_alpha = search_inputs[i].moon_alpha
            best_residuals_per_scenarios.append([moon_period, moon_initial_alpha, residuals, scenario_time, scenario_flux])
        best_residuals_per_scenarios = np.array(best_residuals_per_scenarios)
        best_residuals_per_scenarios = best_residuals_per_scenarios[np.argsort(np.array([best_residual_per_scenarios[2] for best_residual_per_scenarios in best_residuals_per_scenarios]).flatten())]
        for i in np.arange(0, 15):
            logging.info("Best residual for period %s, alpha %s: %s", best_residuals_per_scenarios[i][0],
                         best_residuals_per_scenarios[i][1], best_residuals_per_scenarios[i][2])
            fig_transit, axs = plt.subplots(1, 1, figsize=(12, 12))
            axs.plot(best_residuals_per_scenarios[i][3], best_residuals_per_scenarios[i][4],
                        color='gray', alpha=1, rasterized=True, label="Flux")
            axs.set_title(
                "Moon period " + str(best_residuals_per_scenarios[i][0]) + " with alpha="
                + str(best_residuals_per_scenarios[i][1]) + " and residual " + str(best_residuals_per_scenarios[i][2]))
            axs.set_xlabel('Time')
            axs.set_ylabel('Flux')
            fig_transit.show()
class SearchInput:
    def __init__(self, moon_period, moon_alpha, moon_ecc, moon_inc, moon_arg_periastron,
                all_t0s) -> None:
        self.moon_period = moon_period
        self.moon_alpha = moon_alpha
        self.moon_ecc = moon_ecc
        self.moon_inc = moon_inc
        self.moon_arg_periastron = moon_arg_periastron
        self.all_t0s = all_t0s


formatter = logging.Formatter('%(message)s')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
target_name = "TIC 350618622"
object_dir = target_name + "_EMLS"
lc_builder = LcBuilder()
object_info = lc_builder.build_object_info(target_name=target_name, author=None, sectors="all", file=None, cadence=120,
                              initial_mask=None, initial_transit_mask=None, star_info=None, aperture=None,
                              eleanor_corr_flux="pdcsap_flux", outliers_sigma=3, high_rms_enabled=False,
                              high_rms_threshold=1.5, high_rms_bin_hours=4, smooth_enabled=False,
                              auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                              auto_detrend_period=None, prepare_algorithm=None, reduce_simple_oscillations=False,
                              oscillation_snr_threshold=4, oscillation_amplitude_threshold=0.1, oscillation_ws_scale=60,
                              oscillation_min_period=0.002, oscillation_max_period=0.2)
if not os.path.exists(object_dir):
    os.mkdir(object_dir)
lc_build = lc_builder.build(object_info, object_dir)
star_mass = lc_build.star_info.mass
star_radius = lc_build.star_info.radius
ab = lc_build.star_info.ld_coefficients
times = lc_build.lc.time.value
flux = lc_build.lc.flux.value
planet_radius = 11.298672
planet_period = 52.97818
planet_t0 = 1376.0535
planet_duration = 4.452 / 24
planet_inc = 88.54
planet_ecc = 0.01
planet_arg_periastron = 0
t0s = [i for i in np.arange(planet_t0, np.max(times), planet_period)]
planet_mass = 133.4886
planet_impact_param = 0.42
planet_semimajor_axis = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period)
emls = ExoMoonLeastSquares(7, star_mass, star_radius, ab, planet_radius, planet_period, planet_t0, planet_duration, planet_semimajor_axis, planet_inc, planet_ecc, planet_arg_periastron, planet_impact_param, times, flux)
moon_radius = 3
moon_period = 2
emls.flux = emls.inject_moon(emls.time, emls.flux, t0s, planet_mass, planet_semimajor_axis, planet_ecc, moon_radius, moon_period)
emls.planet_mass_grid = [planet_mass]
emls.moon_inc_grid = [90]
emls.moon_ecc_grid = [0]
emls.moon_arg_periastron_grid = [0]
emls.run()
