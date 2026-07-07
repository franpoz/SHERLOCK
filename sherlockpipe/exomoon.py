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
from scipy.signal import lombscargle
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
    """Performs exomoon least-squares search and fitting."""
    MAX_RESIDUAL = sys.maxsize

    def __init__(self, object_dir, cpus, star_mass, star_radius, ab, planet_radius, planet_radius_err,
                 planet_mass, planet_period,
                 planet_t0, planet_duration, planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc,
                 planet_ecc_err, planet_arg_periastron, planet_impact_param, min_radius, max_radius, t0s, time, flux,
                 flux_err, max_moon_density, period_grid_size=2000, radius_grid_size=10,
                 min_cadences_per_t0: bool = 20):
        """Initialize the exomoon least-squares search.

        Parameters
        ----------
        object_dir : str
            Directory for output files.
        cpus : int
            Number of CPU cores to use.
        star_mass : float
            Stellar mass in solar masses.
        star_radius : float
            Stellar radius in solar radii.
        ab : list
            Limb darkening coefficients [u1, u2].
        planet_radius : float
            Planet radius in Earth radii.
        planet_radius_err : float
            Planet radius uncertainty in Earth radii.
        planet_mass : float
            Planet mass in Earth masses.
        planet_period : float
            Planet orbital period in days.
        planet_t0 : float
            Planet transit epoch in BJD.
        planet_duration : float
            Planet transit duration in days.
        planet_semimajor_axis : float
            Planet semimajor axis in AU.
        planet_inc : float
            Planet orbital inclination in degrees.
        planet_inc_err : float
            Planet inclination uncertainty in degrees.
        planet_ecc : float
            Planet orbital eccentricity.
        planet_ecc_err : float
            Planet eccentricity uncertainty.
        planet_arg_periastron : float
            Planet argument of periastron in degrees.
        planet_impact_param : float
            Planet impact parameter.
        min_radius : float
            Minimum moon radius to search in Earth radii.
        max_radius : float
            Maximum moon radius to search in Earth radii.
        t0s : array_like
            Barycentric transit times.
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        max_moon_density : float
            Maximum moon density in g/cm^3.
        period_grid_size : int, optional
            Number of grid points for period search, default 2000.
        radius_grid_size : int, optional
            Number of grid points for radius search, default 10.
        min_cadences_per_t0 : bool, optional
            Minimum cadences required per transit, default 20.
        """
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
        self.planet_mass_grid = [planet_mass]
        self.moon_inc_grid = [90]
        self.moon_ecc_grid = [0]
        self.moon_arg_periastron_grid = [0]

    @staticmethod
    def compute_semimajor_axis(major_mass, minor_period):
        """Compute the semimajor axis from mass and orbital period.

        Parameters
        ----------
        major_mass : float
            Mass of the central body in solar masses.
        minor_period : float
            Orbital period in days.

        Returns
        -------
        float
            Semimajor axis in AU.
        """
        period_seconds = minor_period * 24. * 3600.
        mass_kg = major_mass * Msolar_to_kg
        a1 = (G * mass_kg * period_seconds ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
        return a1 / 1.496e11

    @staticmethod
    def compute_roche_period(moon_density):
        """Compute the Roche limit orbital period for a given moon density.

        Parameters
        ----------
        moon_density : float
            Moon density in g/cm^3.

        Returns
        -------
        float
            Roche limit period in days.
        """
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
        return semimajor_axis * (1 - eccentricity) * ((minor_mass / (3 * major_mass)) ** (1 / 3))

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
        argument = (
            np.sqrt(max(0, (star_radius + transiting_body_radius) ** 2
                        - (impact_parameter * star_radius) ** 2))
            / transiting_body_semimajor_axis
        )
        argument = np.clip(argument, -1.0, 1.0)
        return transit_period / np.pi * np.arcsin(argument)
        #return 2 * moon_semimajor_axis / (planet_semimajor_axis * 2 * np.pi) * planet_period

    @staticmethod
    def compute_moon_period_grid(min, max, mode="lin", samples=10000):
        """Generate a grid of moon orbital periods.

        Parameters
        ----------
        min : float
            Minimum period in days.
        max : float
            Maximum period in days.
        mode : str, optional
            Grid spacing mode, "lin" or "log", default "lin".
        samples : int, optional
            Number of grid samples, default 10000.

        Returns
        -------
        numpy.ndarray
            Array of period values.
        """
        if "log" == mode:
            return np.logspace(math.log(min, 10), math.log(max, 10), samples, base=10)
        else:
            return np.linspace(min, max, samples)

    @staticmethod
    def fit_single_t0(time, flux, flux_err, t0_guess, duration_window, transit_params):
        """Fit only t0 for a single transit. a, inc, rp must be pre-set on transit_params.

        Returns (best_theta, best_model) where best_theta is [t0].
        """
        bounds = [(t0_guess - duration_window / 4, t0_guess + duration_window / 4)]

        def model(t0_val, t):
            transit_params.t0 = t0_val
            m = TransitModel(transit_params, t)
            return m.light_curve(transit_params)

        def cost(t0_val):
            m = model(t0_val[0], time)
            return 0.5 * np.sum(((flux - m) / flux_err) ** 2)

        if len(time) > 0:
            res = minimize(cost, [t0_guess], method='Powell', bounds=bounds)
            best_theta = res.x
        else:
            best_theta = np.array([t0_guess])
        best_model = model(float(best_theta[0]), time)
        return best_theta, best_model

    @staticmethod
    def fit_single_transit(time, flux, flux_err, guess_params, guess_params_bounds,
                           transit_params: batman.TransitParams, expected_duration=None):
        """Fit a single transit to light curve data.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        guess_params : list
            Initial guess parameters [t0, a, inc, rp].
        guess_params_bounds : list
            Bounds for each parameter.
        transit_params : batman.TransitParams
            BATMAN transit parameters object.
        expected_duration : float or None, optional
            Expected transit duration in days. If provided, models with
            shorter duration are penalized to prevent the optimizer from
            shrinking the transit to avoid noisy wings.

        Returns
        -------
        tuple
            (best_theta, best_model) where best_theta is [t0, a, inc, rp]
            and best_model is the model light curve array.
        """
        # params = global transit params from literature, except T0
        def model(theta, t):
            """BATMAN model wrapper for a single transit.

            Parameters
            ----------
            theta : list
                Parameter array [t0, a, inc, rp].
            t : array_like
                Time array.

            Returns
            -------
            array_like
                Model light curve.
            """
            transit_params.t0 = theta[0]
            transit_params.a = theta[1]
            transit_params.inc = theta[2]
            transit_params.rp = theta[3]
            m = TransitModel(transit_params, t)
            return m.light_curve(transit_params)

        def neg_loglike(theta):
            """Negative log-likelihood for transit fit optimization.

            Parameters
            ----------
            theta : list
                Parameter array [t0, a, inc, rp].

            Returns
            -------
            float
                Negative log-likelihood value.
            """
            m = model(theta, time)
            res = (flux - m) / flux_err
            # penaliza geometrías sin tránsito (b>1) para evitar modelo plano
            # b = (a * cos i) / R*  con i en radianes:
            _, a, inc, _ = theta
            b = a * np.cos(np.deg2rad(inc))
            penalty = 0.0
            if b > 1.0:
                penalty += 1e6 * (b - 1.0) ** 2  # empuja lejos del "no-tránsito"
            if expected_duration is not None:
                rp = theta[3]
                dur_arg = np.sqrt(max(0, (1.0 + rp) ** 2 - b ** 2)) / a
                dur_arg = np.clip(dur_arg, -1.0, 1.0)
                model_duration = transit_params.per / np.pi * np.arcsin(dur_arg)
                if model_duration < expected_duration:
                    penalty += 1e4 * (expected_duration - model_duration) ** 2
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
        """Fit all individual planet transits to refine T0 values.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        plot : bool, optional
            If True, generate transit fit plots, default False.
        """
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        planet_duration_plot = self.planet_duration * 2
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
                                                                             guess_params, guess_params_bounds, params,
                                                                             expected_duration=self.planet_duration)
            best_t0 = best_theta[0]
            if plot:
                times_in_t0 = time[(time > best_t0 - planet_duration_plot) & (time < best_t0 + planet_duration_plot)]
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
                    plt.savefig(self.object_dir + "/T0_" + str(round(best_t0, 3)) + ".png")
                    plt.close(fig_transit)

    def subtract_planet_transit(self, time, flux, flux_err, mode='subtract', plot=False):
        """Subtract or mask the planet transit signal from the light curve.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        mode : str, optional
            'subtract' to subtract the planet model or 'mask' to remove
            in-transit points, default 'subtract'.
        plot : bool, optional
            If True, generate subtraction plots, default False.

        Returns
        -------
        tuple
            (result_time, result_model, result_flux_err) after planet removal.
        """
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
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
        params.a = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun,
                                                                                u.au)
        params.inc = self.planet_inc
        params.u = self.ab
        params.limb_dark = "quadratic"
        # Phase-fold all transits to get a reliable transit shape
        fold_time = []
        fold_flux = []
        fold_flux_err = []
        for t0 in self.bary_t0s:
            mask = np.abs(time - t0) < duration_window
            if len(time[mask]) < self.min_cadences_per_t0:
                continue
            fold_time.append(time[mask] - t0)
            fold_flux.append(flux[mask])
            if flux_err is not None:
                fold_flux_err.append(flux_err[mask])
        if len(fold_time) > 0:
            fold_time = np.concatenate(fold_time)
            fold_flux = np.concatenate(fold_flux)
            fold_flux_err = np.concatenate(fold_flux_err) if flux_err is not None else None
            fold_t0 = 0.0
            min_inc = np.rad2deg(np.arccos(1 / params.a))
            fold_guess = [fold_t0, params.a, self.planet_inc, params.rp]
            fold_bounds = [(fold_time.min(), fold_time.max()),
                           (fold_guess[1] * 0.9, fold_guess[1] * 1.1),
                           (90 - min_inc, 90),
                           (fold_guess[3] - self.planet_radius_err / star_radius_rearth,
                            fold_guess[3] + self.planet_radius_err / star_radius_rearth)]
            fold_theta, _ = ExoMoonLeastSquares.fit_single_transit(
                fold_time, fold_flux, fold_flux_err, fold_guess, fold_bounds, params)
            logging.info("Fold fit: a=%.2f→%.2f, inc=%.1f→%.1f, rp=%.4f→%.4f (depth %.1f→%.1f ppm)",
                         fold_guess[1], fold_theta[1],
                         fold_guess[2], fold_theta[2],
                         fold_guess[3], fold_theta[3],
                         fold_guess[3]**2 * 1e6, fold_theta[3]**2 * 1e6)
            params.a = fold_theta[1]
            params.inc = fold_theta[2]
            params.rp = fold_theta[3]
            if plot:
                # Build initial-guess model
                params_guess = batman.TransitParams()
                params_guess.t0 = fold_guess[0]; params_guess.per = params.per
                params_guess.rp = fold_guess[3]; params_guess.a = fold_guess[1]
                params_guess.inc = fold_guess[2]
                params_guess.ecc = params.ecc; params_guess.w = params.w
                params_guess.u = self.ab;
                params_guess.limb_dark = "quadratic"
                mg = batman.TransitModel(params_guess, fold_time)
                model_guess = mg.light_curve(params_guess)
                # Build best-fit model
                params_best = batman.TransitParams()
                params_best.t0 = fold_theta[0]; params_best.per = params.per
                params_best.rp = fold_theta[3]; params_best.a = fold_theta[1]
                params_best.inc = fold_theta[2]
                params_best.ecc = params.ecc; params_best.w = params.w
                params_best.u = self.ab; params_best.limb_dark = "quadratic"
                mb = batman.TransitModel(params_best, fold_time)
                model_best = mb.light_curve(params_best)
                fig_fold, (ax_fold, ax_res) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax_fold.scatter(fold_time, fold_flux, color='gray', alpha=0.5, s=5)
                ax_fold.plot(fold_time, model_guess, 'b--', alpha=0.7, label='Initial guess')
                ax_fold.plot(fold_time, model_best, 'r-', alpha=1, label='Fold fit')
                ax_fold.set_title(f"Phase-folded fit: rp={fold_guess[3]:.4f}->{fold_theta[3]:.4f}, depth={fold_guess[3]**2*1e6:.0f}->{fold_theta[3]**2*1e6:.0f} ppm")
                ax_fold.set_ylabel('Flux')
                ax_fold.legend()
                residuals_fold = fold_flux - model_best
                ax_res.scatter(fold_time, residuals_fold, color='gray', alpha=0.5, s=5)
                ax_res.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax_res.set_xlabel('Time from T0 (days)')
                ax_res.set_ylabel('Residuals')
                rms_res = np.std(residuals_fold)
                ax_res.set_title(f'Residuals RMS={rms_res*1e6:.0f} ppm')
                plt.tight_layout()
                plt.savefig(os.path.join(self.object_dir, "fold_fit.png"))
                plt.close(fig_fold)
        self.ttvs = []
        ttv_t0s = []
        rejected_t0s = []
        for t0 in self.bary_t0s:
            if mode == 'subtract':
                mask = np.abs(time - t0) < duration_window
                intransit_indexes = np.argwhere(np.abs(time - t0) < duration_window).flatten()
                params.t0 = t0
                params.u = self.ab
                best_theta, best_model = ExoMoonLeastSquares.fit_single_t0(
                    time[mask], flux[mask], flux_err[mask], t0, duration_window, params)
                best_t0 = best_theta[0]
                self.ttvs.append(t0 - best_t0)
                ttv_t0s.append(t0)
                result_model[intransit_indexes] = result_model[intransit_indexes] - (best_model - 1)
                # Quality control: post-subtraction RMS ratio
                transit_mask = np.abs(time[mask] - best_t0) < self.planet_duration / 2
                oot_mask = ~transit_mask
                if np.sum(transit_mask) > 5 and np.sum(oot_mask) > 5:
                    post_res = result_model[intransit_indexes]
                    it_rms = np.std(post_res[transit_mask])
                    oot_rms = np.std(post_res[oot_mask])
                    rms_ratio = it_rms / oot_rms if oot_rms > 0 else float('inf')
                    if rms_ratio > 2:
                        logging.info("  Rejecting transit T0=%.4f: post-sub RMS ratio=%.2f (it=%.6f, oot=%.6f)",
                                     best_t0, rms_ratio, it_rms, oot_rms)
                        rejected_t0s.append(t0)
                if plot:
                    suffix = "_rejected" if t0 in rejected_t0s else "_sub"
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
                    plt.savefig(self.object_dir + "/T0_" + str(round(best_t0, 3)) + suffix + ".png")
                    plt.close()
            else:
                intransit_mask = np.abs(self.time - t0) <= duration_window / 2
                result_time = result_time[~intransit_mask]
                result_model = result_model[~intransit_mask]
                if result_flux_err is not None:
                    result_flux_err = result_flux_err[~intransit_mask]
        if len(rejected_t0s) > 0:
            logging.info("Rejected %d/%d transits due to poor subtraction quality",
                         len(rejected_t0s), len(self.bary_t0s))
            self.bary_t0s = [t0 for t0 in self.bary_t0s if t0 not in rejected_t0s]
            self.ttvs = [ttv for t0, ttv in zip(ttv_t0s, self.ttvs) if t0 not in rejected_t0s]
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(self.ttvs)), self.ttvs, marker='o')
        plt.xticks(range(len(self.ttvs)), [f"{t0:.4f}" for t0 in self.bary_t0s], rotation=45)
        plt.xlabel("Barycentric T0 Values")
        plt.ylabel("TTV (days)")
        plt.title("Transit Timing Variations")
        plt.tight_layout()
        ttv_path = os.path.join(self.object_dir, "ttvs.png") if self.object_dir else None
        if ttv_path:
            plt.savefig(ttv_path)
        plt.close()
        if len(self.ttvs) > 5:
            ttv_rms = np.std(self.ttvs)
            logging.info("TTV diagnostics: RMS=%.4f d, median|O-C|=%.4f d, max|O-C|=%.4f d",
                         ttv_rms, np.median(np.abs(self.ttvs)), np.max(np.abs(self.ttvs)))
            ttv_times = np.array(self.bary_t0s)
            freqs = np.linspace(1 / (ttv_times[-1] - ttv_times[0]),
                                0.5 / np.median(np.diff(ttv_times)), 5000)
            power = lombscargle(ttv_times, self.ttvs, 2 * np.pi * freqs)
            periods = 1 / freqs
            best_idx = np.argmax(power)
            logging.info("TTV periodogram: best P=%.4f d, power=%.4f",
                         periods[best_idx], power[best_idx])
            fig_ttv_per, ax_ttv = plt.subplots(1, 1, figsize=(10, 6))
            ax_ttv.plot(periods, power, color='gray')
            ax_ttv.set_xlabel("Period (days)")
            ax_ttv.set_ylabel("Lomb-Scargle Power")
            ax_ttv.set_title("TTV Periodogram")
            ax_ttv.set_xscale('log')
            plt.tight_layout()
            plt.savefig(self.object_dir + "/ttv_periodogram.png")
            plt.close()
        return time, result_model, result_flux_err

    @staticmethod
    def compute_moon_ts(planet_t0, half_moon_orbit_transit_duration, half_moon_transit_duration):
        """Compute the maximum moon T0 offset (tau) for a given transit.

        Parameters
        ----------
        planet_t0 : float
            Planet transit epoch.
        half_moon_orbit_transit_duration : float
            Half the moon orbit transit duration in days.
        half_moon_transit_duration : float
            Half the moon transit duration in days.

        Returns
        -------
        float
            Maximum moon T0 tau offset in days.
        """
        min_t1 = planet_t0 - half_moon_orbit_transit_duration
        min_t0 = min_t1 + half_moon_transit_duration
        max_moon_t0_tau = np.abs(planet_t0 - min_t0)
        return max_moon_t0_tau

    @staticmethod
    def compute_moon_t0(planet_first_t0, planet_t0, moon_period, moon_initial_alpha, max_moon_t0_tau):
        """Compute the moon transit time and orbital phase.

        Parameters
        ----------
        planet_first_t0 : float
            First planet transit epoch.
        planet_t0 : float
            Current planet transit epoch.
        moon_period : float
            Moon orbital period in days.
        moon_initial_alpha : float
            Initial moon orbital phase in radians.
        max_moon_t0_tau : float
            Maximum moon T0 offset in days.

        Returns
        -------
        tuple
            (moon_t0, moon_phase) where moon_t0 is the transit time in days
            and moon_phase is the orbital phase in radians.
        """
        moon_phase = moon_initial_alpha + (((planet_t0 - planet_first_t0) % moon_period) / moon_period) * 2 * np.pi
        moon_tau = np.cos(moon_phase)
        moon_t0 = planet_t0 + moon_tau * max_moon_t0_tau
        return moon_t0, moon_phase


    @staticmethod
    #@njit(fastmath=True, parallel=False)
    def compute_moon_transit_scenarios(time, flux, flux_err, planet_t0, moon_initial_alpha, moon_period, bary_t0s,
                                       half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration,
                                       half_moon_transit_duration):
        """Compute moon transit scenarios for all barycentric transit times.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        planet_t0 : float
            First planet transit epoch.
        moon_initial_alpha : float
            Initial moon orbital phase in radians.
        moon_period : float
            Moon orbital period in days.
        bary_t0s : list
            List of barycentric transit times.
        half_moon_orbit_transit_duration : float
            Half the moon orbit transit duration in days.
        half_moon_max_orbit_transit_duration : float
            Half the maximum moon orbit transit duration in days.
        half_moon_transit_duration : float
            Half the moon transit duration in days.

        Returns
        -------
        list or None
            List of [moon_alpha, time, flux, flux_err] for each scenario,
            or None if no valid scenarios are found.
        """
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
        """Search for a moon signal for a given set of input parameters.

        Parameters
        ----------
        search_input : SearchInput
            Search input parameters.
        return_lc : bool, optional
            If True, return light curve data in addition to residuals,
            default False.

        Returns
        -------
        tuple
            (residual, baseline_residual, radius, model_snr) if return_lc is
            False, or (residual, baseline_residual, radius, model_snr,
            it_time, it_flux, oot_time, oot_flux, residual_model) if
            return_lc is True.
        """
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
        t0_flux_mean_value = np.mean(it_flux)
        if len(self.models) == 0:
            residual_calculation = ExoMoonLeastSquares.MAX_RESIDUAL
            residual_baseline = np.sum(((it_flux - 1) / it_flux_err) ** 2)
            residual_radius = self.min_radius
            residual_model = np.ones_like(it_flux)
        else:
            residual_calculation, residual_baseline, residual_radius, residual_model = (
                self.calculate_residuals(it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                                         t0_flux_mean_value, len(scenario_time), self.models, self.min_radius,
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
        """Compute SDE from chi-squared residuals (standard TLS formula).

        SDE = max(SR_detrended) / std(SR_detrended), where
        SR = min(chi²) / chi² and detrending is via running-median.

        No global normalisation; avoids instability from flat baselines.
        """
        SR = np.nan_to_num(np.min(chi2) / chi2, nan=1.0)
        kernel = oversampling_factor * kernel_size
        if kernel % 2 == 0:
            kernel += 1
        if len(SR) > 2 * kernel:
            baseline = ExoMoonLeastSquares.running_median(SR, kernel)
            power = SR - baseline
        else:
            power = SR - np.mean(SR)
        power = power - np.mean(power)
        SDE_raw = np.max(power / np.std(power)) if np.std(power) > 0 else 0
        power = power * SDE_raw / np.max(power) if np.max(power) > 0 else power
        SDE = float(np.max(power))
        return SR, np.zeros_like(SR), power, SDE_raw, SDE

    @staticmethod
    def refine_alpha(alpha_grid, residuals_matrix):
        """Refine the best alpha per period via parabolic interpolation.

        For each period (column of residuals_matrix), fits a quadratic
        through the three alpha values bracketing the grid minimum, then
        returns the alpha at the parabola's vertex. This gives sub-grid
        alpha resolution at zero extra search cost.

        Parameters
        ----------
        alpha_grid : array_like, shape (N_alpha,)
            Grid of alpha values in radians.
        residuals_matrix : array_like, shape (N_alpha, N_periods)
            Residuals matrix where rows index alpha and columns index period.

        Returns
        -------
        numpy.ndarray, shape (N_periods, 2)
            Each row is [refined_alpha, min_residual] for that period.
        """
        n_alpha = len(alpha_grid)
        refined = np.empty((residuals_matrix.shape[1], 2))
        for col in range(residuals_matrix.shape[1]):
            r_col = residuals_matrix[:, col]
            best_j = np.argmin(r_col)
            j_left = (best_j - 1) % n_alpha
            j_right = (best_j + 1) % n_alpha
            a_center = alpha_grid[best_j]
            a_local = np.array([
                alpha_grid[j_left] - a_center,
                0.0,
                alpha_grid[j_right] - a_center,
            ])
            r_local = np.array([r_col[j_left], r_col[best_j], r_col[j_right]])
            coeffs = np.polyfit(a_local, r_local, 2)
            # Parabola r = c2*a^2 + c1*a + c0, minimum at a = -c1 / (2*c2)
            delta = -coeffs[1] / (2.0 * coeffs[2])
            refined[col, 0] = (a_center + delta) % (2.0 * np.pi)
            refined[col, 1] = r_col[best_j]
        return refined

    @staticmethod
    #@njit(fastmath=True, parallel=False)
    def calculate_residuals(it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                            t0_flux_mean_value, datapoints, models, min_radius, max_radius,
                            star_radius, radius_grid_size, ootr_weight=0.5):
        """Calculate residuals between model and data for a transit scenario.

        Parameters
        ----------
        it_time : array_like
            In-transit time array in days.
        it_flux : array_like
            In-transit normalized flux array.
        it_flux_err : array_like
            In-transit flux uncertainty array.
        oot_time : array_like
            Out-of-transit time array in days.
        oot_flux : array_like
            Out-of-transit normalized flux array.
        oot_flux_err : array_like
            Out-of-transit flux uncertainty array.
        t0_flux_mean_value : float
            Mean flux at the transit center.
        datapoints : int
            Number of data points.
        models : list of (float, array_like)
            Pre-computed model transit shapes as [(radius_Rearth, flux), ...].
        min_radius : float
            Minimum moon radius in Earth radii.
        max_radius : float
            Maximum moon radius in Earth radii.
        star_radius : float
            Stellar radius in solar radii.
        radius_grid_size : int
            Number of grid points for radius search.
        ootr_weight : float, optional
            Weight for out-of-transit residuals, default 0.5.

        Returns
        -------
        tuple
            (best_residual, residuals_baseline, best_radius, best_model).
        """
        oot_radius_residuals = np.sum(((oot_flux - 1) / oot_flux_err) ** 2)
        n_it = len(it_flux)
        model_baseline = np.ones(n_it)
        residuals_baseline = np.sum(((it_flux - model_baseline) / it_flux_err) ** 2) + ootr_weight * oot_radius_residuals
        min_depth = ((min_radius * R_earth_to_R_sun) ** 2) / star_radius ** 2
        best_radius = min_radius
        best_residual = ExoMoonLeastSquares.MAX_RESIDUAL
        best_model = model_baseline
        if t0_flux_mean_value < 1 - min_depth:
            radius = np.sqrt(1 - t0_flux_mean_value) * star_radius / R_earth_to_R_sun
            closest_r, closest_model = min(models, key=lambda x: abs(x[0] - radius))
            if len(closest_model) < 2:
                return best_residual, residuals_baseline, best_radius, best_model
            interpolated = interp1d(np.arange(len(closest_model)), closest_model,
                                    axis=0, fill_value='extrapolate')
            model_sample = interpolated(np.linspace(0, len(closest_model), n_it))
            scale_factor = (t0_flux_mean_value - 1) / (np.min(model_sample) - 1)
            model_sample_scaled = 1 + (model_sample - 1) * scale_factor
            it_radius_residuals = np.sum(((it_flux - model_sample_scaled) / it_flux_err) ** 2)
            best_residual = it_radius_residuals + oot_radius_residuals
            best_radius = radius
            best_model = model_sample_scaled
        return best_residual, residuals_baseline, best_radius, best_model

    @staticmethod
    def running_median(data, kernel):
        """Compute sliding median of width kernel and same length as data.

        Parameters
        ----------
        data : array_like
            Input data array.
        kernel : int
            Window width.

        Returns
        -------
        numpy.ndarray
            Running median array with same length as input.
        """
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
        """Remove outliers from a light curve using sigma clipping.

        Parameters
        ----------
        flux : array_like
            Flux array to clip.
        sigma : float, optional
            Sigma threshold for clipping, default 5.0.
        sigma_lower : float or None, optional
            Lower sigma threshold, default None.
        sigma_upper : float or None, optional
            Upper sigma threshold, default None.
        \*\*kwargs
            Additional arguments passed to astropy.stats.sigma_clip.

        Returns
        -------
        numpy.ndarray
            Boolean outlier mask (True where outliers are found).
        """
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
        """Check if a transit epoch should be skipped due to insufficient cadences.

        Parameters
        ----------
        time : array_like
            Time array around the transit.

        Returns
        -------
        bool
            True if the transit should be skipped.
        """
        if len(time) < self.min_cadences_per_t0:
            return True
        return False

    def inject_moon(self, time, flux, flux_err, t0s, planet_semimajor_axis, planet_ecc, moon_radius,
                    moon_period, moon_mass, initial_alpha=0, plot=False, inject_planet=False, inject_moon=True):
        """Inject a moon transit signal into the light curve.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        t0s : list
            List of barycentric transit times.
        planet_semimajor_axis : float
            Planet semimajor axis in AU.
        planet_ecc : float
            Planet orbital eccentricity.
        moon_radius : float
            Moon radius in Earth radii.
        moon_period : float
            Moon orbital period in days.
        moon_mass : float
            Moon mass in Earth masses.
        initial_alpha : float, optional
            Initial moon orbital phase in radians, default 0.
        plot : bool, optional
            If True, generate injection plots, default False.
        inject_planet : bool, optional
            If True, inject planet transit as well, default False.
        inject_moon : bool, optional
            If True, inject moon transit signal, default True.

        Returns
        -------
        numpy.ndarray
            Flux array with injected moon signal.
        """
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
        params.tau_moon = (params.t0_bary / moon_period - (initial_alpha + np.pi) / (2 * np.pi)) % 1
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
        self.injected_flux_moon_pandora = injected_flux_moon
        self.injected_moon_period = moon_period
        self.injected_moon_alpha = initial_alpha
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
        moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, moon_period)
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
        """Remove flux points outside the moon transit windows.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        t0s : list
            List of barycentric transit times.
        max_planet_mass : float
            Maximum planet mass in Earth masses.

        Returns
        -------
        tuple
            (time, flux, flux_err) restricted to transit windows.
        """
        max_period = self.au_to_period(max_planet_mass * M_earth_to_M_sun, self.compute_hill_radius(self.star_mass, max_planet_mass * M_earth_to_M_sun, self.planet_semimajor_axis))
        moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, max_period)
        moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                              self.planet_semimajor_axis * AU_TO_RSUN,
                                                              self.planet_period,
                                                              moon_semimajor_axis * AU_TO_RSUN)
        mask_parts = []
        for t0 in t0s:
            mask_parts.append(np.argwhere(
                (time > t0 - moon_orbit_transit_duration) & (time < t0 + moon_orbit_transit_duration)).flatten())
        flux_mask = np.concatenate(mask_parts).astype(int)
        time = time[flux_mask]
        flux = flux[flux_mask]
        flux_err = flux_err[flux_mask]
        return time, flux, flux_err

    @staticmethod
    def depth_to_radius(depth, star_radius):
        """Convert a transit depth to a body radius.

        Parameters
        ----------
        depth : float
            Transit depth (fractional flux decrease).
        star_radius : float
            Stellar radius in solar radii.

        Returns
        -------
        float
            Body radius in Earth radii.
        """
        return np.sqrt(depth * ((star_radius / R_earth_to_R_sun) ** 2))

    def plot_signal(self, axs, scenario_time, scenario_flux, it_time, oot_time, model):
        """Plot the moon signal detection results.

        Parameters
        ----------
        axs : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        scenario_time : array_like
            Time array for the full scenario.
        scenario_flux : array_like
            Flux array for the full scenario.
        it_time : array_like
            In-transit time array.
        oot_time : array_like
            Out-of-transit time array.
        model : array_like
            Model light curve.
        """
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

    def run(self, plot=False, inject_params=None):
        """Run the full exomoon search over the parameter grid.

        Parameters
        ----------
        plot : bool, optional
            If True, generate diagnostic plots, default False.
        inject_params : dict or None, optional
            If a dict with keys 'moon_radius', 'moon_period', 'initial_alpha',
            'moon_mass', injects a moon signal into the flux before the grid
            search to test injection-recovery. Default None (no injection).
        """
        planet_mass_grid = self.planet_mass_grid
        moon_inc_grid = self.moon_inc_grid
        moon_ecc_grid = self.moon_ecc_grid
        moon_arg_periastron_grid = self.moon_arg_periastron_grid
        self._time_original = self.time.copy()  # save before trimming
        self.time, self.flux, self.flux_err = self.remove_non_transit_flux(self.time, self.flux, self.flux_err, self.bary_t0s, np.max(planet_mass_grid))
        rms = np.std(self.flux)
        self.time, self.flux, _ = self.subtract_planet_transit(self.time, self.flux, self.flux_err, plot=plot)
        if inject_params is not None:
            self.flux = self.inject_moon(self.time, self.flux, self.flux_err, self.bary_t0s, self.planet_semimajor_axis,
                                         self.planet_ecc, inject_params['moon_radius'], inject_params['moon_period'],
                                         inject_params['moon_mass'], initial_alpha=inject_params['initial_alpha'],
                                         inject_planet=False, inject_moon=True)
            logging.info("Injected moon: P=%.4f d, R=%.2f R_e, alpha=%.4f rad",
                         inject_params['moon_period'], inject_params['moon_radius'], inject_params['initial_alpha'])
        ma = batman.TransitParams()
        ma.t0 = 10  # time of inferior conjunction
        ma.per = self.planet_period  # orbital period, use Earth as a reference
        ma.a = self.planet_semimajor_axis * AU_TO_RSUN / self.star_radius  # semi-major axis (in units of stellar radii)
        ma.inc = self.planet_inc  # orbital inclination (in degrees)
        ma.ecc = self.planet_ecc  # eccentricity
        ma.w = 0  # longitude of periastron (in degrees)
        ma.u = self.ab  # limb darkening coefficients
        ma.limb_dark = "quadratic"  # limb darkening model
        time_model = np.linspace(ma.t0 - self.planet_duration * 0.7, ma.t0 + self.planet_duration * 0.7, 1000)
        self.models = []
        for r in np.linspace(self.min_radius, self.max_radius, self.radius_grid_size):
            ma.rp = r * R_earth_to_R_sun / self.star_radius
            m = batman.TransitModel(ma, time_model)
            model_r = m.light_curve(ma)
            trimmed = model_r[model_r < 1]
            if len(trimmed) > 1:
                self.models.append((r, trimmed))
        search_inputs = []
        if len(self.models) == 0:
            self.models = [(self.min_radius, np.array([0.9999, 0.9999]))]
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
        snr_matrix = np.full((alpha_grid_size, len(period_grid)), np.nan)
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
            corrected_residual = residuals / residuals_baseline if residuals_baseline > 1e-10 else 1.0
            if not (0 < corrected_residual < ExoMoonLeastSquares.MAX_RESIDUAL / 2):
                corrected_residual = 1.0
            if corrected_residual > max_residual and corrected_residual < ExoMoonLeastSquares.MAX_RESIDUAL / 2:
                max_residual = corrected_residual
            best_residuals_per_scenarios.append(
                [moon_period, moon_initial_alpha, corrected_residual, radius])
            residuals_matrix[i % alpha_grid_size][
                i // alpha_grid_size] = corrected_residual
            snr_matrix[i % alpha_grid_size][
                i // alpha_grid_size] = model_snr
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
        refined_alpha_per_period = ExoMoonLeastSquares.refine_alpha(alpha_grid, residuals_matrix)
        snr_refined_alpha_per_period = ExoMoonLeastSquares.refine_alpha(alpha_grid, -snr_matrix)
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
        snr_spectrum = np.transpose(best_snr_per_period)[1]
        axs[2].plot(period_grid, snr_spectrum)
        axs[2].set_title("SNRs")
        axs[2].set_xlabel('P (days)')
        axs[2].set_ylabel('SNR')
        kernel = 31
        SR, power_raw, power, SDE_raw, SDE = ExoMoonLeastSquares.spectra(best_residuals_spectra, kernel)
        axs[3].plot(period_grid, power, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
        axs[3].set_title("SDE (residuals)")
        axs[3].set_xlabel('P (days)')
        axs[3].set_ylabel('Power')
        snr_median = ExoMoonLeastSquares.running_median(snr_spectrum, kernel)
        snr_norm = snr_spectrum - snr_median
        snr_sde = np.max(snr_norm / np.std(snr_norm))
        snr_norm = snr_norm * snr_sde / np.max(snr_norm)
        axs[4].plot(period_grid, snr_norm, color='gray', alpha=1)
        axs[4].set_title("SDE (SNR)")
        axs[4].set_xlabel('P (days)')
        axs[4].set_ylabel('Power')
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
            if len(model) == 0:
                continue
            moon_depth = 1 - np.min(model)
            moon_radius = ExoMoonLeastSquares.depth_to_radius(moon_depth, self.star_radius)
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
            moon_initial_alpha = refined_alpha_per_period[top_sde_index][0]
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
            if len(model) == 0:
                continue
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
            if len(model) == 0:
                logging.info("Best SNR for period %s, alpha %s: no in-transit data, skipping",
                             moon_period, moon_initial_alpha)
                continue
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            moon_depth = 1 - np.min(model)
            moon_radius = ExoMoonLeastSquares.depth_to_radius(moon_depth, self.star_radius)
            if i == 0:
                self.best_snr_radius = moon_radius
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
        # --- Compute best-candidate T0s (reused by recovery + plotting) ---
        best_snr_idx = best_scenario_per_snr_args[-1]
        best_snr_period = period_grid[best_snr_idx]
        best_snr_alpha = snr_refined_alpha_per_period[best_snr_idx][0]
        moon_max_sma = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, period_grid[-1])
        moon_dur_orbit_max = self.compute_transit_duration(self.star_radius,
            self.planet_semimajor_axis * AU_TO_RSUN, self.planet_period,
            moon_max_sma * AU_TO_RSUN, self.planet_impact_param)
        half_orbit_dur_max = moon_dur_orbit_max / 2
        moon_sma_rec = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, best_snr_period)
        moon_dur_orbit_rec = self.compute_transit_duration(self.star_radius,
            self.planet_semimajor_axis * AU_TO_RSUN, self.planet_period,
            moon_sma_rec * AU_TO_RSUN, self.planet_impact_param)
        half_orbit_dur_rec = moon_dur_orbit_rec / 2
        moon_dur = self.compute_transit_duration(self.star_radius,
            self.planet_semimajor_axis * AU_TO_RSUN, self.planet_period,
            self.max_radius * R_earth_to_R_sun, self.planet_impact_param)
        half_moon_dur = moon_dur / 2
        rec_t0s, inj_t0s = [], []
        has_injected = hasattr(self, 'injected_flux_moon_pandora')
        for t0_i in self.bary_t0s:
            max_tau = ExoMoonLeastSquares.compute_moon_ts(t0_i, half_orbit_dur_rec, half_moon_dur)
            rec_t0, _ = ExoMoonLeastSquares.compute_moon_t0(
                self.bary_t0s[0], t0_i, best_snr_period, best_snr_alpha, max_tau)
            rec_t0s.append(rec_t0)
            if has_injected:
                # Use time that matches injected_flux_moon_pandora length
                t_for_mask = self._time_original if hasattr(self, '_time_original') and \
                    len(self._time_original) == len(self.injected_flux_moon_pandora) else self.time
                mask_w = np.abs(t_for_mask - t0_i) < half_orbit_dur_max
                inj_flux = self.injected_flux_moon_pandora[mask_w]
                if np.any(inj_flux < 1.0):
                    inj_t0s.append(t_for_mask[mask_w][np.argmin(inj_flux)])
        self.recovered_t0s = rec_t0s
        self.injected_t0s = inj_t0s if inj_t0s else None

        if inject_params is not None:
            period_err = abs(best_snr_period - inject_params['moon_period']) / inject_params['moon_period']
            alpha_err = abs(np.cos(best_snr_alpha) - np.cos(inject_params['initial_alpha']))
            alias_period = None
            alias_k = None
            min_alias_err = period_err
            for k in range(-5, 6):
                p_alias = 1 / (1 / inject_params['moon_period'] + k / self.planet_period)
                if p_alias > 0 and abs(p_alias - best_snr_period) < min_alias_err:
                    min_alias_err = abs(p_alias - best_snr_period) / inject_params['moon_period']
                    alias_period = p_alias
                    alias_k = k
            logging.info("INJECTION RECOVERY (SNR): injected P=%.4f d, alpha=%.4f rad -> recovered P=%.4f d, alpha=%.4f rad (delta_P=%.2f%%, delta_cos-alpha=%.4f)",
                         inject_params['moon_period'], inject_params['initial_alpha'],
                         best_snr_period, best_snr_alpha, period_err * 100, alpha_err)
            if alias_k is not None and alias_k != 0:
                logging.info("  Best alias match: k=%d, P_alias=%.4f d (delta=%.2f%%)",
                             alias_k, alias_period, min_alias_err * 100)
            self.recovered_period = float(best_snr_period)
            self.recovered_alpha = float(best_snr_alpha)
            # T0-based recovery
            if self.injected_t0s and len(rec_t0s) > 0:
                diffs = np.abs(np.array(rec_t0s[:len(self.injected_t0s)]) - np.array(self.injected_t0s))
                self.recovered = bool(np.median(diffs) < 0.1)
            else:
                self.recovered = bool(min_alias_err < 0.05)
        # Plot best moon model on each single transit
        try:
            has_injected_plot = hasattr(self, 'injected_flux_moon_pandora')
            # Get recovered radius from search at best (P, alpha)
            _, _, recovered_radius, _, _, _, _, _, _ = self.search(
                SearchInput(best_snr_period, best_snr_alpha,
                            0, 90, 0, self.min_radius, self.planet_impact_param,
                            moon_dur_orbit_max, moon_dur, moon_dur_orbit_rec,
                            half_moon_dur, half_orbit_dur_rec, half_orbit_dur_max),
                return_lc=True)
            logging.info("Recovered moon radius: %.4f R_earth", recovered_radius)
            for i, t0 in enumerate(self.bary_t0s):
                mask_orbit = np.abs(self.time - t0) < half_orbit_dur_max
                if len(self.time[mask_orbit]) < self.min_cadences_per_t0:
                    continue
                t_local = self.time[mask_orbit]
                f_local = self.flux[mask_orbit]
                # Use pre-computed T0s
                moon_t0_t = rec_t0s[i]
                moon_t0_t_inj = inj_t0s[i] if has_injected and i < len(inj_t0s) else None
                ma_moon = batman.TransitParams()
                ma_moon.t0 = moon_t0_t
                ma_moon.per = self.planet_period
                ma_moon.rp = max(recovered_radius, self.min_radius) * R_earth_to_R_sun / self.star_radius
                ma_moon.a = self.planet_semimajor_axis * AU_TO_RSUN / self.star_radius
                ma_moon.inc = self.planet_inc
                ma_moon.ecc = self.planet_ecc
                ma_moon.w = 0
                ma_moon.u = self.ab
                ma_moon.limb_dark = "quadratic"
                m_moon = batman.TransitModel(ma_moon, t_local)
                model_moon = m_moon.light_curve(ma_moon)
                fig_t, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.axvspan(t0 - self.planet_duration / 2, t0 + self.planet_duration / 2,
                           color='blue', alpha=0.1, label='Planet transit window')
                if has_injected_plot:
                    t_pandora = self._time_original if hasattr(self, '_time_original') and \
                        len(self._time_original) == len(self.injected_flux_moon_pandora) else self.time
                    mask_orbit_full = np.abs(t_pandora - t0) < half_orbit_dur_max
                    inj_local = self.injected_flux_moon_pandora[mask_orbit_full]
                    t_local_inj = t_pandora[mask_orbit_full]
                    ax.plot(t_local_inj, inj_local, color='green', alpha=0.6,
                            label='Pandora injected moon')
                    if moon_t0_t_inj is not None:
                        ax.axvspan(moon_t0_t_inj - half_moon_dur, moon_t0_t_inj + half_moon_dur,
                                   color='gray', alpha=0.2)
                        ax.axvline(moon_t0_t_inj, color='red', linestyle='--',
                                   label='Injected moon T0')
                ax.scatter(t_local, f_local, color='gray', alpha=0.7, s=10,
                           label='Subtracted flux')
                ax.plot(t_local, model_moon, color='darkorange', alpha=1,
                        label='Recovered batman model')
                ax.axvline(moon_t0_t, color='darkorange', linestyle=':',
                           label='Recovered moon T0')
                title = f"T0={t0:.4f}\nRecovered P={best_snr_period:.3f}d alpha={best_snr_alpha:.3f}rad"
                if has_injected_plot:
                    title += f"\nInjected P={self.injected_moon_period:.3f}d alpha={self.injected_moon_alpha:.3f}rad"
                ax.set_title(title)
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Flux')
                ax.legend(fontsize=7)
                plt.savefig(self.object_dir + "/T0_" + str(round(t0, 3)) + "_moon_model.png")
                plt.close(fig_t)
        except Exception as e:
            logging.warning("Could not plot moon model transits: %s", e)

        self.max_snr = float(np.max(np.transpose(best_snr_per_period)[1]))
        self.max_sde = float(SDE)

        # Assess candidate using IR calibration
        try:
            self.detection_prob = ExoMoonLeastSquares.assess_candidate(
                "resources/emls_ir", self.max_snr, len(self.bary_t0s))
            logging.info("Detection probability from IR calibration: %.3f", self.detection_prob)
        except Exception:
            self.detection_prob = 0.0

    @staticmethod
    def assess_candidate(work_dir, snr_obs, n_tr):
        """Return detection probability P ∈ [0, 1] from IR calibration.

        P = P_sig × P_rec where:
            P_sig = 1 − FAP(SNR_obs) ≈ min(0.99 × SNR_obs / SNR_99, 1.0)
            P_rec = completeness at the observed SNR bin and N_tr.
        """
        import json
        if not os.path.isabs(work_dir):
            work_dir = os.path.abspath(work_dir)
        cal_dir = os.path.join(work_dir, "calibration")
        try:
            with open(os.path.join(cal_dir, "sde_thresholds.json")) as f:
                thresholds = json.load(f)
            with open(os.path.join(cal_dir, "completeness.json")) as f:
                completeness = json.load(f)
        except Exception as e:
            logging.warning("assess_candidate: failed to load calibration: %s", e)
            return 0.0
        # P_sig from null SNR thresholds
        thr = thresholds.get(str(n_tr), {})
        snr_99 = float(thr.get("SNR_99", 999))
        snr_999 = float(thr.get("SNR_99.9", 999))
        if not thr:
            n_keys = sorted([int(k) for k in thresholds.keys()])
            if len(n_keys) >= 2 and n_keys[0] <= n_tr <= n_keys[-1]:
                lo = max(k for k in n_keys if k <= n_tr)
                hi = min(k for k in n_keys if k >= n_tr)
                if lo == hi:
                    thr = thresholds[str(lo)]
                else:
                    t_lo, t_hi = thresholds[str(lo)], thresholds[str(hi)]
                    w = (n_tr - lo) / (hi - lo) if hi > lo else 0
                    snr_99 = t_lo["SNR_99"] * (1 - w) + t_hi["SNR_99"] * w
                    snr_999 = t_lo.get("SNR_99.9", 999) * (1 - w) + t_hi.get("SNR_99.9", 999) * w
        if snr_obs >= snr_999:
            p_sig = 0.999
        elif snr_obs <= 0:
            p_sig = 0.0
        else:
            p_sig = min(0.99 * (snr_obs / snr_99), 1.0)
        # P_rec from SNR-bin completeness
        comp_n_tr = completeness.get(str(n_tr), completeness.get("all", {}))
        snr_bins = [0, 3, 5, 7, 10, 15, 20, 30, 50]
        p_rec = 0.0
        for i in range(len(snr_bins) - 1):
            label = f"{snr_bins[i]}-{snr_bins[i+1]}"
            if snr_obs >= snr_bins[i] and snr_obs < snr_bins[i+1]:
                p_rec = float(comp_n_tr.get(label, 0.0))
                break
        return p_sig * p_rec


class SearchInput:
    """Container for moon search input parameters."""
    def __init__(self, moon_period, moon_alpha, moon_ecc, moon_inc, moon_arg_periastron, min_radius, impact_param,
                 moon_max_orbit_transit_duration, moon_transit_duration, moon_orbit_transit_duration,
                 half_moon_transit_duration, half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration) -> None:
        """Initialize search input parameters for a single moon search scenario.

        Parameters
        ----------
        moon_period : float
            Moon orbital period in days.
        moon_alpha : float
            Moon initial orbital phase in radians.
        moon_ecc : float
            Moon orbital eccentricity.
        moon_inc : float
            Moon orbital inclination in degrees.
        moon_arg_periastron : float
            Moon argument of periastron in degrees.
        min_radius : float
            Minimum moon radius in Earth radii.
        impact_param : float
            Impact parameter.
        moon_max_orbit_transit_duration : float
            Maximum moon orbit transit duration in days.
        moon_transit_duration : float
            Moon transit duration in days.
        moon_orbit_transit_duration : float
            Moon orbit transit duration in days.
        half_moon_transit_duration : float
            Half the moon transit duration in days.
        half_moon_orbit_transit_duration : float
            Half the moon orbit transit duration in days.
        half_moon_max_orbit_transit_duration : float
            Half the maximum moon orbit transit duration in days.
        """
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

if __name__ == '__main__':

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
    confirmed_tois_df = tois_df.loc[((tois_df['Disposition'] == 'CP') | (tois_df['Disposition'] == 'KP'))]
    confirmed_tois_df = confirmed_tois_df.sort_values(by='Planet SNR', ascending=False)
    # Moon dynamical stability filter
    def hill_roche_stable(period, p_radius, p_mass, star_mass):
        """Check if Hill radius exceeds Roche limit: r_Hill > 2 * r_Roche.

        The Hill sphere (Hill 1878, Am. J. Math. 1, 5) defines the region
        where the planet's gravity dominates over the star:
            r_Hill = a_p × (M_p / 3M_star)^(1/3)

        The Roche limit (Roche 1849, Acad. Sci. Montpellier) for a rigid
        satellite is the closest orbit before tidal disruption:
            r_Roche = 2.44 × R_p × (ρ_p / ρ_m)^(1/3)

        A moon can only exist if r_Hill > r_Roche. We require a factor 2
        margin for long-term stability (Barnes & O'Brien 2002, ApJ 575, 1087)."""
        a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, period)
        r_hill = a * (p_mass * M_earth_to_M_sun / (3 * star_mass)) ** (1 / 3)
        moon_density = 5.5
        p_density = p_mass / p_radius ** 3
        r_roche = 2.44 * p_radius * max(p_density / moon_density, 0.1) ** (1 / 3)
        r_roche_au = r_roche * R_earth_to_R_sun / 215.032
        return r_hill > r_roche_au * 2

    def planet_planet_stable(a1, m1_earth, a2, m2_earth, star_mass, n_hill=5):
        """Check if two planets are separated by n_hill mutual Hill radii.

        Two planets are dynamically packed (and moons can be ejected) if
        their separation is less than K × R_mHill where K = 3-5
        (Gladman 1993, Icarus 106, 247; Chambers et al. 1996, Icarus 119, 261).

        The mutual Hill radius is:
            R_mHill = (a1 + a2)/2 × ((M1 + M2) / 3M_star)^(1/3)

        Masses in Earth masses, semi-major axes in AU, star_mass in M_sun."""
        delta_a = abs(a1 - a2)
        m_total = (m1_earth + m2_earth) * M_earth_to_M_sun
        r_mhill = (a1 + a2) / 2 * (m_total / (3 * star_mass)) ** (1 / 3)
        return delta_a >= n_hill * r_mhill

    def moon_tidal_stable(period, p_mass, p_radius, star_mass, t_age_gyr=5):
        """Check if a moon can survive tidal migration over the system age.

        Heller et al. (2014, Astrobiology 14, 798) derived an approximate
        threshold for moon survival against planet-moon tidal migration:
            P_planet > P_crit × (M_p/M_⊕)^(1/6) × (age/1 Gyr)^(1/4)

        with P_crit ≈ 10 days for rocky planets (Q_p ≈ 100, k2p ≈ 0.3).
        Below this period, tides cause the moon to spiral into the planet
        within the system's age (Barnes & O'Brien 2002, ApJ 575, 1087)."""
        P_crit = 10.0 * (p_mass ** (1.0 / 6.0)) * (t_age_gyr ** (1.0 / 4.0))
        return period > P_crit

    keep_mask = []
    n_hillroche, n_tidal, n_companion = 0, 0, 0
    for _, r in confirmed_tois_df.iterrows():
        star_mass = r['Stellar Mass (M_Sun)']
        p_mass = r['Predicted Mass (M_Earth)']
        p_radius = r['Planet Radius (R_Earth)']
        period = r['Period (days)']
        if np.isnan(star_mass) or star_mass <= 0:
            keep_mask.append(False)
            continue
        if np.isnan(p_mass) or p_mass <= 0:
            p_mass = p_radius ** 1.5
        a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, period)
        p_mass_sun = p_mass * M_earth_to_M_sun
        moon_stable = hill_roche_stable(period, p_radius, p_mass, star_mass)
        if not moon_stable:
            n_hillroche += 1
            keep_mask.append(False)
            continue
        if not moon_tidal_stable(period, p_mass, p_radius, star_mass):
            n_tidal += 1
            keep_mask.append(False)
            continue
        sisters = confirmed_tois_df[confirmed_tois_df['Object Id'] == r['Object Id']]
        companion_ok = True
        for _, sis in sisters.iterrows():
            if sis['OI'] == r['OI']:
                continue
            sis_mass = sis['Predicted Mass (M_Earth)']
            if np.isnan(sis_mass) or sis_mass <= 0:
                sis_mass = sis['Planet Radius (R_Earth)'] ** 1.5
            sis_a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, sis['Period (days)'])
            if not planet_planet_stable(a, p_mass, sis_a, sis_mass, star_mass):
                companion_ok = False
                break
        if not companion_ok:
            n_companion += 1
        keep_mask.append(companion_ok)
    confirmed_tois_df = confirmed_tois_df[keep_mask]
    logging.info("Dynamical filter: Hill-Roche=%d, Tidal=%d, Companion=%d removed, %d remain",
                 n_hillroche, n_tidal, n_companion, len(confirmed_tois_df))
    #confirmed_tois_df = confirmed_tois_df.loc[confirmed_tois_df['Object Id'] == 'TIC 309792357']
    confirmed_tois_df = confirmed_tois_df.iloc[0:5]
    for index, row in confirmed_tois_df.iterrows():
        target_name = row['Object Id']
        logging.info("Processing target: %s", target_name)
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
    
        flux_err = np.ma.filled(flux_err, np.ma.median(flux_err))
        flux = wotan.flatten(times, flux, method="biweight", window_length=planet_duration * 5)
        t0s = [i for i in np.arange(planet_t0, np.max(times), planet_period)]
        emls = ExoMoonLeastSquares(object_dir, 4, star_mass, star_radius, ld_coeffs,
                                   planet_radius, planet_radius_err, planet_mass, planet_period, planet_t0, planet_duration,
                                   planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc, planet_ecc_err, planet_arg_periastron, planet_impact_param,
                                   min_radius, max_radius, t0s, times, flux, flux_err, max_moon_density,
                                   period_grid_size=1000, radius_grid_size=20)
        #emls.fit_single_transits(times, flux, flux_err)
        # emls.flux = emls.inject_moon(emls.time, emls.flux, emls.flux_err, emls.bary_t0s, planet_semimajor_axis,
        #                             planet_ecc, moon_radius, moon_period, moon_mass, plot=True, initial_alpha=moon_alpha,
        #                              inject_planet=False, inject_moon=False)
        #emls.fit_single_transits(times, flux, flux_err, plot=True)
        emls.run(plot=True,
                 inject_params={
                    'moon_radius': planet_radius / 5,
                    'moon_period': planet_duration * 20,
                    'initial_alpha': 1.0,
                    'moon_mass': (planet_radius / 5) ** 1.5,
                }
        )
    