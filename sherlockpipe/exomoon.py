import time

import ellc
import numpy as np
import astropy.constants as ac
import astropy.units as u
import wotan
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema


class ExoMoonLeastSquares:
    G = 6.674e-11  # m3 kg-1 s-2
    au_to_Rsun = 215.032
    Msolar_to_kg = 2.e30

    def compute_semimajor_axis(self, major_mass, minor_period):
        period_seconds = minor_period * 24. * 3600.
        mass_kg = major_mass * self.Msolar_to_kg
        a1 = (self.G * mass_kg * period_seconds ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
        return a1 / 1.496e11

    def compute_hill_radius(self, major_mass, minor_mass, semimajor_axis, eccentricity=0):
        return semimajor_axis * (1 - eccentricity) * (major_mass / (3 * minor_mass) ** (1 / 3))

    def compute_moon_min_max_orbit_periods(self, star_mass, planet_mass, planet_ecc, planet_semimajor_axis):
        planet_hill_r = self.compute_hill_radius(star_mass, planet_mass, planet_semimajor_axis, planet_ecc)
        return 0.5, planet_hill_r

    def compute_moon_orbit_transit_length(self, star_radius, planet_radius,
                                          planet_semimajor_axis, planet_period, moon_semimajor_axis,
                                          moon_eccentrictiy=0, moon_arg_periastron=0, moon_inclination=90,
                                          planet_eccentricity=0, planet_arg_periastron=0, planet_inclination=90):
        #planet_period / np.pi * np.asin(np.sqrt((star_radius + planet_radius) * 2 - (b * star_radius) * 2) / 2)
        return 2 * moon_semimajor_axis / (planet_semimajor_axis * 2 * np.pi) * planet_period


    def compute_moon_period_grid(self, min, max, mode="log", samples=20):
        if "log" == mode:
            return np.logspace(min, max, samples)
        else:
            return np.linspace(min, max, samples)

    def tokenize_transits_moon_orbit_ranges(self, time, flux, flux_err, star_mass, star_radius, planet_mass, planet_radius,
                                            planet_period, planet_t0, moon_semimajor_axis, planet_semimajor_axis,
                                            moon_period, moon_eccentrictiy=0,
                                            moon_arg_periastron=0, moon_inclination=90,
                                            planet_eccentricity=0, planet_arg_periastron=0, planet_inclination=90):
        moon_semimajor_axis = self.compute_semimajor_axis(planet_mass, moon_period)
        planet_semimajor_axis = self.compute_semimajor_axis(star_mass, planet_period)
        moon_orbit_transit_length = self.compute_moon_orbit_transit_length(star_radius, planet_radius,
                                                                           planet_semimajor_axis, planet_period,
                                                                           moon_semimajor_axis)
        # TODO we probably need to define left_transit_length and right_transit_length depending on moon orbit parameters
        all_t0s = np.linspace(planet_t0, time[len(time) - 1], planet_period)
        moon_orbit_tokens = [[t0, t0 - moon_orbit_transit_length / 2, t0 + moon_orbit_transit_length / 2] for t0 in all_t0s]
        return moon_orbit_tokens

    def subtract_planet_transit(self, ab, star_radius, star_mass, time, flux, planet_radius, planet_t0,
                                planet_period, planet_inc=90):
        P1 = planet_period * u.day
        a = np.cbrt((ac.G * star_mass * P1 ** 2) / (4 * np.pi ** 2)).to(u.au)
        model = ellc.lc(
            t_obs=time,
            radius_1=star_radius.to(u.au) / a,  # star radius convert from AU to in units of a
            radius_2=planet_radius.to(u.au) / a,
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

    def compute_moon_transit_scenarios(self, time, flux, planet_t0, moon_period, moon_orbit_ranges, moon_transit_length):
        #TODO need to take into account "prograde" or "retrograde" orbit
        orbit_scenarios = []
        tau = moon_transit_length / 2
        for moon_orbit_range in moon_orbit_ranges:
            t0 = moon_orbit_range[0]
            time_args = np.argwhere(time[(time > moon_orbit_range[1]) & (time < moon_orbit_range[2])])
            #TODO we'd need to fill measurement gaps (detected from the time array)
            time_orbit_range = time[time_args]
            flux_orbit_range = flux[time_args]
            tau1 = time_orbit_range - planet_t0
            alpha = np.arccos(tau1 / tau)
            alpha_comp = 2 * np.pi - alpha
            phase_delta = (moon_orbit_range[0] - planet_t0) / moon_period
            alpha_0 = alpha + phase_delta
            alpha_comp_0 = alpha_comp + phase_delta
            time_alpha_0 = t0 + np.cos(alpha_0) * tau
            time_alpha_comp_0 = t0 + np.cos(alpha_0) * tau
            orbit_scenarios = orbit_scenarios.append([time_orbit_range, tau1, flux_orbit_range, alpha_0, alpha_comp_0, time_alpha_0, time_alpha_comp_0])
        return orbit_scenarios

    def normalize_scenarios(self, moon_transit_scenarios):
        scenario = moon_transit_scenarios[0]
        flux = scenario[2]
        time_alpha_0 = scenario[5]
        time_alpha_comp_0 = scenario[6]
        normalized_scenarios = []
        normalized_scenarios = normalized_scenarios.append([[time_alpha_0, flux], [time_alpha_comp_0, flux]])
        for scenario in moon_transit_scenarios[1:]:
            flux = scenario[2]
            time_alpha_0 = scenario[5]
            time_alpha_comp_0 = scenario[6]
            normalized_scenarios = normalized_scenarios.append([[time_alpha_0, flux], [time_alpha_comp_0, flux]])
        return normalized_scenarios

    def search(self, normalized_moon_transit_scenarios):
        scenarios_grid = np.array(np.meshgrid(*np.array(normalized_moon_transit_scenarios))).T.reshape(-1, len(normalized_moon_transit_scenarios))
        stick_scenarios_time_grid = []
        stick_scenarios_flux_grid = []
        for scenario in scenarios_grid:
            scenario_time = []
            scenario_flux = []
            for datum in scenario:
                scenario_time.append(datum[0])
                scenario_flux.append(datum[1])
            sorted_scenario_times_args = np.argsort(scenario_time)
            stick_scenarios_time_grid.append(scenario_time[sorted_scenario_times_args])
            stick_scenarios_flux_grid.append(scenario_flux[sorted_scenario_times_args])
        t0 = time.time()
        cadence = 2
        R_s = 1.1
        M_s = 1.3
        P_min = 0.5
        P_max = 22
        ld_coefficients = [0.2, 0.1]
        duration = wotan.t14(self.star_radius, self.star_mass, self.planet_period, True)
        duration_grid = [duration]
        curve_rms = np.std(self.flux)
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
        a_au = self.compute_semimajor_axis(self.star_mass, 0.5)
        a_Rs = a_au / (self.star_radius * 0.00465047)
        model = tm.evaluate(k=rp_rs, ldc=ld_coefficients, t0=0.5, p=1.0, a=a_Rs, i=0.5 * np.pi)
        model = model[model < 1]
        baseline_model = np.full(len(model), 1)
        model = np.append(baseline_model, model)
        model = np.append(model, baseline_model)
        model_samples = [self.downsample(model, int(duration) * 3) for duration in duration_grid]
        i = 0
        for stick_scenarios_time in stick_scenarios_time_grid:
            stick_scenarios_flux = stick_scenarios_flux_grid[i]
            residual_calculation = np.full((len(model_samples), len(stick_scenarios_time)), np.nan)
            for model_index, model_sample in enumerate(model_samples):
                model_duration_days = duration_grid[model_index] * cadence / 60 / 24
                first_valid_time = stick_scenarios_time[stick_scenarios_time > stick_scenarios_time[0] + model_duration_days * 3][0]
                time_without_tail = stick_scenarios_time[stick_scenarios_time < stick_scenarios_time[len(stick_scenarios_time) - 1] - model_duration_days * 3]
                last_valid_time = time_without_tail[len(time_without_tail) - 1]
                first_valid_time = stick_scenarios_time[0]
                last_valid_time = stick_scenarios_time[len(stick_scenarios_time) - 1 - len(model_sample)]
                dt_flux = wotan.flatten(stick_scenarios_time, stick_scenarios_flux, model_duration_days * 4, method="biweight")
                dt_flux = stick_scenarios_flux
                for flux_index, flux_value in enumerate(
                        stick_scenarios_time[(stick_scenarios_time >= first_valid_time) & (stick_scenarios_time <= last_valid_time)]):
                    residual_calculation[model_index][flux_index] = self.calculate_residuals(stick_scenarios_time, dt_flux, model_sample,
                                                                                        flux_index)
                local_residual_minima = argrelextrema(residual_calculation[model_index], np.less)[0]
                minima_mask = np.full(len(residual_calculation[model_index]), False)
                minima_mask[local_residual_minima] = True
                max_allowed_residual = np.nanmax(residual_calculation[model_index])
                residual_calculation[model_index][
                    np.where(np.isnan(residual_calculation[model_index]))] = max_allowed_residual
                # residual_calculation[model_index][~minima_mask] = max_allowed_residual
                # time_plot = time[minima_mask]
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
            i = i + 1

    def downsample(self, array, npts: int):
        interpolated = interp1d(np.arange(len(array)), array, axis=0, fill_value='extrapolate')
        downsampled = interpolated(np.linspace(0, len(array), npts))
        return downsampled

    def calculate_residuals(self, time, flux, model_sample, flux_index):
        flux_subset = flux[flux_index:flux_index + len(model_sample)]
        time_subset = time[flux_index:flux_index + len(model_sample)]
        # TODO adjusting model to minimum flux value this might get improved by several scalations of min_flux
        model_sample_scaled = np.copy(model_sample)
        flux_subset_len = len(flux_subset)
        max_flux = 1 - np.std(flux_subset) / 2
        flux_at_middle = np.mean(flux_subset[flux_subset_len // 3:flux_subset_len * 2 // 3])
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
            return np.sum((flux_subset - model_sample_scaled) ** 2) ** 0.5 * depth
        return np.nan


    def run(self):
        min_period = 0.5 # TODO compute this value somehow
        max_period = self.compute_hill_radius(self.star_mass, self.planet_mass, self.planet_semimajor_axis)
        period_grid = self.compute_moon_period_grid(min_period, max_period)
        planet_mass_grid = self.planet_mass_grid
        moon_inc_grid = self.moon_inc_grid
        moon_ecc_grid = self.moon_ecc_grid
        moon_arg_periastron_grid = self.moon_arg_periastron_grid
        flux = self.subtract_planet_transit(self.ab, self.star_radius, self.star_mass, self.time, self.flux,
                                            self.planet_radius, self.planet_t0, self.planet_period, self.planet_inc)
        for planet_mass in planet_mass_grid:
            for moon_inc in moon_inc_grid:
                for moon_ecc in moon_ecc_grid:
                    for moon_arg_periastron in moon_arg_periastron_grid:
                        for moon_period in period_grid:
                            #TODO moon_orbit_ranges should use moon_radius ?
                            moon_semimajor_axis = self.moon_semimajor_axis = self.compute_semimajor_axis(planet_mass, moon_period)
                            planet_semimajor_axis = self.planet_semimajor_axis = self.compute_semimajor_axis(self.star_mass, self.planet_period)
                            moon_transit_length = self.compute_moon_orbit_transit_length(self.star_radius, self.planet_radius,
                                                                   planet_semimajor_axis, self.planet_period,
                                                                   moon_semimajor_axis, moon_ecc, moon_arg_periastron, moon_inc,
                                                                   self.planet_ecc, self.planet_arg_periastron, self.planet_inc)
                            moon_orbit_ranges = self.tokenize_transits_moon_orbit_ranges(self.time, flux, self.flux_err,
                                                                     self.star_mass, self.star_radius, planet_mass,
                                                                     self.planet_radius, self.planet_period,
                                                                     self.planet_t0, moon_semimajor_axis, planet_semimajor_axis,
                                                                     moon_period, moon_ecc,
                                                                     moon_arg_periastron, moon_inc, self.planet_ecc,
                                                                     self.planet_arg_periastron, self.planet_inc)
                            moon_transit_scenarios = self.compute_moon_transit_scenarios(self.time, flux, self.planet_t0, moon_period, moon_orbit_ranges, moon_transit_length)
                            normalized_moon_transit_scenarios = self.normalize_scenarios(moon_transit_scenarios)
                            self.search(normalized_moon_transit_scenarios)


