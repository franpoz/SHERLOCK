import dataclasses
import logging
import math
import re
import shutil
from pathlib import Path

import alexfitter
import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pyplot as plt
from lcbuilder.helper import LcbuilderHelper

from sherlockpipe.loading.tool_with_candidate import ToolWithCandidate


resources_dir = path.join(path.dirname(__file__))
resources_dir = str(Path(resources_dir).parent.absolute())


@dataclasses.dataclass
class DistributionParams:
    yerr: float = -7.0
    yerr_lower_err: float = -15.0
    yerr_upper_err: float = 0.0
    lnsigma: float = 0.0
    lnsigma_lower_err: float = -30.0
    lnsigma_upper_err: float = 30.0
    lnrho: float = 0.0
    lnrho_lower_err: float = -30.0
    lnrho_upper_err: float = 30.0


class Fitter2(ToolWithCandidate):
    """
    Used to run a bayesian fit on the Sherlock search results.
    """

    def __init__(self, object_dir, fit_dir, only_initial, is_candidate_from_search, candidates_df, mcmc=False,
                 detrend=None, estimate_noise=True, rho_err_multi=5.0, sigma_err_multi=5.0, yerr_err_multi=5.0):
        super().__init__(is_candidate_from_search, candidates_df)
        self.object_dir = os.getcwd() if object_dir is None else object_dir
        self.data_dir = fit_dir
        self.only_initial = only_initial
        self.mcmc = mcmc
        self.detrend = detrend
        self.estimate_noise = estimate_noise
        self.rho_err_multi = rho_err_multi
        self.sigma_err_multi = sigma_err_multi
        self.yerr_err_multi = yerr_err_multi

    def mask_non_fit_candidates(self, time, flux, flux_err, candidate_df, fit_candidate_df):
        """
        Masks all the candidates found in previous runs in the SHERLOCK search.

        :param time: the time array
        :param flux: the flux measurements array
        :param flux_err: the flux error measurements array
        :param candidate_df: the candidates used for noise estimation
        :param fit_candidate_df: the candidates used for fit
        :return: time, flux and flux_err with previous candidates in-transit data masked
        """
        mask_candidate_df = candidate_df.loc[~candidate_df['number'].isin(fit_candidate_df['number'])]
        for index, candidate_row in mask_candidate_df.iterrows():
            period = candidate_row["period"]
            duration = candidate_row["duration"]
            duration = duration / 60 / 24
            t0 = candidate_row["t0"]
            logging.info("Masking candidate number %.0f with P=%.3fd, T0=%.2f and D=%.2fd", index + 1, period, t0,
                         duration)
            time, flux, flux_err = LcbuilderHelper.mask_transits(time, flux, period, duration * 2, t0, flux_err)
        return time, flux, flux_err

    def fit(self, candidate_df, fit_candidate_df, star_df, cpus, allesfit_dir, tolerance, fit_orbit):
        """
        Main method to run the alexfitter fit.

        :param candidate_df: the candidates dataframe to be used for noise estimation.
        :param fit_candidate_df: the candidates dataframe to be used for the final fit.
        :param star_df: the star dataframe.
        :param cpus: the number of cpus to be used.
        :param allesfit_dir: the directory for the fit to be run.
        :param tolerance: the nested sampling tolerance threshold.
        :param fit_orbit: whether to fit eccentricity and arg. of periastron.
        """
        logging.info("Preparing fit files")
        sherlock_star_file = self.object_dir + "/params_star.csv"
        star_file = allesfit_dir + "/params_star.csv"
        params_file = allesfit_dir + "/params.csv"
        settings_file = allesfit_dir + "/settings.csv"
        run = int(candidate_df['number'].iloc[0])
        # We load the unprocessed raw PDCSAP flux curve
        lc = pd.read_csv(self.object_dir + '/lc.csv', header=0)
        time, flux, flux_err = lc["#time"].values, lc["flux"].values, lc["flux_err"].values
        #time, flux, flux_err = self.mask_previous_candidates(time, flux, flux_err, run)
        lc = pd.DataFrame(columns=['#time', 'flux', 'flux_err'])
        lc['#time'] = time
        lc['flux'] = flux
        lc['flux_err'] = flux_err
        curve_rms = np.nanstd(flux)
        lc.loc[(lc['flux_err'] == 0) | np.isnan(lc['flux_err']), 'flux_err'] = curve_rms
        lc.to_csv(allesfit_dir + "/lc.csv", index=False)
        if os.path.exists(sherlock_star_file) and os.path.isfile(sherlock_star_file):
            shutil.copyfile(sherlock_star_file, star_file)
        logging.info("Preparing fit properties")
        self.overwrite_settings(settings_file, cpus, candidate_df, tolerance, "multi")
        with open(params_file, 'w') as f:
            f.write(self.fill_candidates_params(candidate_df, star_df, fit_orbit, self.detrend))
            f.truncate()
        logging.info("Running initial guess")
        try:
            alexfitter.show_initial_guess(allesfit_dir)
        except Exception as e:
            logging.exception(str(e))
        # TODO fix custom_plot for all candidates
        # Fitter.custom_plot(candidate_row["name"], candidate_row["period"], fit_width, allesfit_dir, "initial_guess")
        shutil.copytree(allesfit_dir + '/results', allesfit_dir + '/results_initial_guess_before_noise')
        shutil.copy(allesfit_dir + '/params.csv', allesfit_dir + '/params_before_noise.csv')
        shutil.copy(allesfit_dir + '/settings.csv', allesfit_dir + '/settings_before_noise.csv')
        if not self.only_initial:
            logging.info("Preparing bayesian fit")
            if self.estimate_noise and self.detrend == 'gp':
                logging.info("Running noise estimation")
                alexfitter.estimate_noise_out_of_transit(allesfit_dir)
                noise_estimation = pd.read_csv(allesfit_dir + "/priors/summary_phot.csv")
                noise_estimation = noise_estimation.iloc[0]
                noise_distribution_params = DistributionParams(yerr=noise_estimation['ln_yerr_median'],
                                   yerr_lower_err=noise_estimation['ln_yerr_ll'],
                                   yerr_upper_err=noise_estimation['ln_yerr_ul'],
                                   lnsigma=noise_estimation['gp_ln_sigma_median'],
                                   lnsigma_lower_err=noise_estimation['gp_ln_sigma_ll'],
                                   lnsigma_upper_err=noise_estimation['gp_ln_sigma_ul'],
                                   lnrho=noise_estimation['gp_ln_rho_median'],
                                   lnrho_lower_err=noise_estimation['gp_ln_rho_ll'],
                                   lnrho_upper_err=noise_estimation['gp_ln_rho_ul'])
                self.overwrite_settings(settings_file, cpus, fit_candidate_df, tolerance, "multi")
                with open(params_file, 'w') as f:
                    f.write(self.fill_candidates_params(fit_candidate_df, star_df, fit_orbit, self.detrend,
                                                           distribution='normal',
                                                           distribution_params=noise_distribution_params))
                    f.truncate()
            time, flux, flux_err = self.mask_non_fit_candidates(time, flux, flux_err, candidate_df, fit_candidate_df)
            # time, flux, flux_err = self.mask_previous_candidates(time, flux, flux_err, run)
            lc = pd.DataFrame(columns=['#time', 'flux', 'flux_err'])
            lc['#time'] = time
            lc['flux'] = flux
            lc['flux_err'] = flux_err
            curve_rms = np.nanstd(flux)
            lc.loc[(lc['flux_err'] == 0) | np.isnan(lc['flux_err']), 'flux_err'] = curve_rms
            lc.to_csv(allesfit_dir + "/lc.csv", index=False)
            if not self.mcmc:
                logging.info("Running dynamic nested sampling")
                try:
                    alexfitter.show_initial_guess(allesfit_dir)
                    alexfitter.ns_fit(allesfit_dir)
                    alexfitter.ns_output(allesfit_dir)
                except Exception as e:
                    logging.exception(str(e))
            elif self.mcmc:
                logging.info("Running MCMC")
                try:
                    alexfitter.mcmc_fit(allesfit_dir)
                    alexfitter.mcmc_output(allesfit_dir)
                except Exception as e:
                    logging.exception(str(e))
            logging.info("Generating custom plots")
            # TODO fix custom_plot for all candidates
            # Fitter.custom_plot(candidate_row["name"], candidate_row["period"], fit_width, allesfit_dir, "posterior")


    def overwrite_settings(self, settings_file, cpus, candidate_df, tolerance, boundaries="single"):
        shutil.copyfile(resources_dir + "/resources/allesfitter/settings2.csv", settings_file)
        fit_width = Fitter2.select_fit_width(candidate_df)
        with open(settings_file, 'r+') as f:
            text = f.read()
            text = re.sub('\\${sherlock:cores}', str(cpus), text)
            text = re.sub('\\${sherlock:fit_width}', str(fit_width), text)
            text = re.sub('\\${sherlock:fit_ttvs}', "False", text)
            text = re.sub('\\${sherlock:names}', ' '.join(candidate_df["name"].astype('str')), text)
            text = re.sub('\\${sherlock:boundaries}', str(boundaries), text)
            text = re.sub('\\${sherlock:tolerance}', str(tolerance), text)
            if self.detrend == 'hybrid_spline':
                detrend_param = "baseline_flux_lc,hybrid_spline"
            elif self.detrend == 'gp':
                detrend_param = 'baseline_flux_lc,sample_GP_Matern32'
            else:
                detrend_param = ''
            text = re.sub('\\${sherlock:detrend}', detrend_param, text)
            f.seek(0)
            f.write(text)
            f.truncate()

    @staticmethod
    def select_fit_width(candidate_df):
        """
        Calculates the window to be used for the fit around the transits.

        :param candidate_df: the candidates dataframe
        :return: the maximum calculated fit_width
        """
        fit_width = 0
        for key, row in candidate_df.iterrows():
            if row["duration"] is not None:
                row_fit_width = float(row["duration"]) / 60 / 24 * 7
                fit_width = fit_width if fit_width > row_fit_width else row_fit_width
        return fit_width

    @staticmethod
    def custom_plot(name, period, fit_width, allesfit_dir, mode="posterior"):
        """
        Creates a custom fit plot from the allesfitter data

        :param name: the candidate name
        :param period: the final period
        :param fit_width: the fit_width window
        :param allesfit_dir: the directory where allesfitter data is stred
        :param mode: the allesfitter plot model
        """
        allesclass = alexfitter.allesclass(allesfit_dir)
        baseline_width = fit_width * 24
        baseline_to_period = fit_width / period
        fig, axes = plt.subplots(2, 3, figsize=(18, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
        fig.subplots_adjust(left=0.5, right=1.5, hspace=0)
        style = 'full'
        allesclass.plot('lc', name, style, ax=axes[0][0], mode=mode)
        axes[0][0].set_title('lc, ' + style)
        allesclass.plot('lc', name, style + '_residuals', ax=axes[1][0], mode=mode)
        axes[1][0].set_title('')
        style = 'phase'
        allesclass.plot('lc', name, style, ax=axes[0][1], mode=mode, zoomwindow=baseline_to_period)
        axes[0][1].set_title('lc, ' + style)
        axes[0][1].set_xlim([- baseline_to_period / 2, baseline_to_period / 2])
        allesclass.plot('lc', name, style + '_residuals', ax=axes[1][1], mode=mode, zoomwindow=baseline_to_period)
        axes[1][1].set_title('')
        axes[1][1].set_xlim([- baseline_to_period / 2, baseline_to_period / 2])
        style = 'phasezoom'
        allesclass.plot('lc', name, style, ax=axes[0][2], mode=mode, zoomwindow=baseline_width)
        axes[0][2].set_title('lc, ' + style)
        axes[0][2].set_xlim([- baseline_width / 2, baseline_width / 2])
        allesclass.plot('lc', name, style + '_residuals', ax=axes[1][2], mode=mode, zoomwindow=baseline_width)
        axes[1][2].set_title('')
        axes[1][2].set_xlim([- baseline_width / 2, baseline_width / 2])
        fig.savefig(allesfit_dir + '/results/ns_' + mode + '_' + name + '_custom.pdf', bbox_inches='tight')
        style = ['phasezoom_occ']
        style = ['phase_variations']

    def fill_candidates_params(self, candidate_df, star_df, fit_orbit, detrend_mode, distribution: str = 'uniform',
                               distribution_params: DistributionParams=None):
        """
        Fills the candidate planets initial parameters and their distribution to be fit.

        :param candidate_df: the candidates dataframe
        :param star_df: the star dataframe
        :param fit_orbit: whether to fit eccentricity and arg. of periastron.
        :param detrend_mode: type of detrend to be used
        :param distribution: uniform or normal
        :param distribution_params: the distribution parameters if known
        :return: the allesfitter candidates parameters
        """
        candidate_priors_text = ""
        for key, row in candidate_df.iterrows():
            candidate_priors_text = candidate_priors_text + \
                                    Fitter2.fill_candidate_params(row, star_df, fit_orbit)
        return candidate_priors_text + self.fill_instrument_params(star_df, detrend_mode, distribution,
                                                                      distribution_params)

    @staticmethod
    def fill_candidate_params(candidate_row, star_df, fit_orbit):
        """
        Fills the candidate planet initial parameters and their distribution to be fit.

        :param candidate_row: the candidate row from the dataframe
        :param star_df: the star dataframe
        :param fit_orbit: whether to fit eccentricity and arg. of periastron.
        :return: the allesfitter candidate parameters
        """
        candidate_params = """#name,value,fit,bounds,label,unit
#companion b astrophysical params,,,,,
${sherlock:name}_rr,${sherlock:rp_rs},1,uniform ${sherlock:rp_rs_min} ${sherlock:rp_rs_max},$R_b / R_\star$,
${sherlock:name}_rsuma,${sherlock:sum_rp_rs_a},1,uniform ${sherlock:sum_rp_rs_a_min} ${sherlock:sum_rp_rs_a_max},$(R_\star + R_b) / a_b$,
${sherlock:name}_cosi,0.0,1,uniform 0.0 0.06,$\cos{i_b}$,
${sherlock:name}_epoch,${sherlock:t0},1,uniform ${sherlock:t0_min} ${sherlock:t0_max},$T_{0;b}$,$\mathrm{BJD}$
${sherlock:name}_period,${sherlock:period},1,uniform ${sherlock:period_min} ${sherlock:period_max},$P_b$,$\mathrm{d}$
${sherlock:name}_f_c,0.0,${sherlock:fit_orbit},uniform -1.0 1.0,$\sqrt{e_b} \cos{\omega_b}$,
${sherlock:name}_f_s,0.0,${sherlock:fit_orbit},uniform -1.0 1.0,$\sqrt{e_b} \sin{\omega_b}$,
"""
        candidate_params = re.sub('\\${sherlock:t0}', str(candidate_row["t0"]), candidate_params)
        candidate_params = re.sub('\\${sherlock:t0_min}', str(candidate_row["t0"] - 0.02), candidate_params)
        candidate_params = re.sub('\\${sherlock:t0_max}', str(candidate_row["t0"] + 0.02), candidate_params)
        candidate_params = re.sub('\\${sherlock:period}', str(candidate_row["period"]), candidate_params)
        candidate_params = re.sub('\\${sherlock:period_min}', str(candidate_row["period"] - candidate_row["per_err"]),
                                  candidate_params)
        candidate_params = re.sub('\\${sherlock:period_max}', str(candidate_row["period"] + candidate_row["per_err"]),
                                  candidate_params)
        candidate_params = re.sub('\\${sherlock:fit_orbit}', str(int(fit_orbit)), candidate_params)
        rp_rs = candidate_row["rp_rs"] if candidate_row["rp_rs"] != "-" else 0.1
        depth = candidate_row["depth"] / 1000
        # TODO calculate depth error in SHERLOCK maybe given the std deviation from the depths or even using the
        #  residuals
        depth_err = depth * 0.2
        rp_rs_err = 0.5 / math.sqrt(depth) * depth_err
        candidate_params = re.sub('\\${sherlock:rp_rs}', str(rp_rs), candidate_params)
        rp_rs_min = rp_rs - 2 * rp_rs_err
        rp_rs_min = rp_rs_min if rp_rs_min > 0 else 0.0000001
        rp_rs_max = rp_rs + 2 * rp_rs_err
        rp_rs_max = rp_rs_max if rp_rs_max < 1 else 0.9999
        candidate_params = re.sub('\\${sherlock:rp_rs_min}', str(rp_rs_min), candidate_params)
        candidate_params = re.sub('\\${sherlock:rp_rs_max}', str(rp_rs_max), candidate_params)
        G = 6.674e-11
        mstar_kg = star_df.iloc[0]["M_star"] * 2e30
        mstar_kg_lerr = star_df.iloc[0]["M_star_lerr"] * 2e30
        mstar_kg_uerr = star_df.iloc[0]["M_star_uerr"] * 2e30
        rstar_au = star_df.iloc[0]['R_star'] * 0.00465047
        rstar_lerr_au = star_df.iloc[0]["R_star_lerr"] * 0.00465047
        rstar_uerr_au = star_df.iloc[0]["R_star_uerr"] * 0.00465047
        per = candidate_row["period"] * 24 * 3600
        per_err = candidate_row["per_err"] * 24 * 3600
        a = (G * mstar_kg * per ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
        a_au = a / 1.496e11
        sum_rp_rs_a = (np.sqrt(depth) + 1.) * rstar_au / a_au
        sum_rp_rs_a_lerr = np.sqrt(
            np.square(0.5 * depth_err / depth) + np.square(rstar_lerr_au / rstar_au) + np.square(
                2 * per_err / 3. / per) + np.square(mstar_kg_lerr / 3. / mstar_kg)) * sum_rp_rs_a
        sum_rp_rs_a_uerr = np.sqrt(
            np.square(0.5 * depth_err / depth) + np.square(rstar_uerr_au / rstar_au) + np.square(
                2 * per_err / 3. / per) + np.square(mstar_kg_uerr / 3. / mstar_kg)) * sum_rp_rs_a
        sum_rp_rs_a_min = sum_rp_rs_a - 2 * sum_rp_rs_a_lerr
        sum_rp_rs_a_max = sum_rp_rs_a + 2 * sum_rp_rs_a_uerr
        sum_rp_rs_a_min = sum_rp_rs_a_min if sum_rp_rs_a_min > 0 else 0.0000001
        candidate_params = re.sub('\\${sherlock:sum_rp_rs_a}', str(sum_rp_rs_a), candidate_params)
        candidate_params = re.sub('\\${sherlock:sum_rp_rs_a_min}', str(sum_rp_rs_a_min), candidate_params)
        candidate_params = re.sub('\\${sherlock:sum_rp_rs_a_max}', str(sum_rp_rs_a_max), candidate_params)
        candidate_params = re.sub('\\${sherlock:name}', str(candidate_row["name"]), candidate_params)
        return candidate_params
    def fill_instrument_params(self, star_df, detrend_mode, distribution='uniform', distribution_params: DistributionParams=None):
        """
        Fills the star and systematics information for each instrument (so far only one)

        :param star_df: the star dataframe
        :param detrend_mode: type of detrend to be used
        :param distribution: distribution: uniform or normal
        :param distribution_params : if set, chooses the baseline fit values
        :return: the allesfitter instrument params
        """
        instrument_params = """#limb darkening coefficients per instrument,,,,,
host_ldc_q1_lc,${sherlock:ld_a},1,uniform 0.0 1.0,$q_{1; \mathrm{lc}}$,
host_ldc_q2_lc,${sherlock:ld_b},1,uniform 0.0 1.0,$q_{2; \mathrm{lc}}$,
#spots per instrument and companion,,,,,
#errors per instrument,
${sherlock:err_params}
#baseline per instrument,
${sherlock:baseline_params}
"""
        if star_df.iloc[0]["ld_a"] is not None and not np.isnan(star_df.iloc[0]["ld_a"]):
            instrument_params = re.sub('\\${sherlock:ld_a}', str(star_df.iloc[0]["ld_a"]), instrument_params)
            instrument_params = re.sub('\\${sherlock:ld_b}', str(star_df.iloc[0]["ld_b"]), instrument_params)
        else:
            instrument_params = re.sub('\\${sherlock:ld_a}', "0.5", instrument_params)
            instrument_params = re.sub('\\${sherlock:ld_b}', "0.5", instrument_params)
        if detrend_mode == 'gp':
            if distribution_params is None:
                distribution_params = DistributionParams()
            if 'uniform' == distribution:
                errs_params = "ln_err_flux_lc," + str(distribution_params.yerr) + ",1," + distribution + " " + \
                              str(distribution_params.yerr_lower_err) + " " + \
                              str(distribution_params.yerr_upper_err) + "," \
                              "$\\\\log{\\\\sigma_\\\\mathrm{lc}}$,\\$\\\\log{ \\\\mathrm{rel. flux.} }$"
                baseline_params = "baseline_gp_matern32_lnsigma_flux_lc," + str(
                    distribution_params.lnsigma) + ",1," + distribution + \
                                  " " + str(distribution_params.lnsigma_lower_err) + " " + \
                                  str(distribution_params.lnsigma_upper_err) + "," \
                                  "$\\\\mathrm{gp: \\\\ln{\\\\sigma} (lc)}$,\n" + \
                                  "baseline_gp_matern32_lnrho_flux_lc," + str(distribution_params.lnrho) + ",1," + \
                                  distribution + " " + str(distribution_params.lnrho_lower_err) + " " + \
                                  str(distribution_params.lnrho_upper_err) + "," \
                                  "$\\\\mathrm{gp: \\\\ln{\\\\rho} (lc)}$,"
            else:
                sigma_lnyerr = (distribution_params.yerr_lower_err \
                    if distribution_params.yerr_lower_err > distribution_params.yerr_upper_err \
                    else distribution_params.yerr_upper_err) * self.yerr_err_multi
                sigma_lnsigma = (distribution_params.lnsigma_lower_err \
                    if distribution_params.lnsigma_lower_err > distribution_params.lnsigma_upper_err \
                    else distribution_params.lnsigma_upper_err) * self.sigma_err_multi
                sigma_lnrho = (distribution_params.lnrho_lower_err \
                    if distribution_params.lnrho_lower_err > distribution_params.lnrho_upper_err \
                    else distribution_params.lnrho_upper_err) * self.rho_err_multi
                errs_params = "ln_err_flux_lc," + str(distribution_params.yerr) + ",1," + distribution + " " + \
                              str(distribution_params.yerr) + " " + \
                              str(sigma_lnyerr) + "," \
                              "$\\\\log{\\\\sigma_\\\\mathrm{lc}}$,\\$\\\\log{ \\\\mathrm{rel. flux.} }$"
                baseline_params = "baseline_gp_matern32_lnsigma_flux_lc," + str(distribution_params.lnsigma) + ",1," + distribution + \
                                  " " + str(distribution_params.lnsigma) + " " + str(sigma_lnsigma) + "," \
                                  "$\\\\mathrm{gp: \\\\ln{\\\\sigma} (lc)}$,\n" + \
                                  "baseline_gp_matern32_lnrho_flux_lc," + str(distribution_params.lnrho) + ",1," + distribution + \
                                  " " + str(distribution_params.lnrho) + " " + str(sigma_lnrho) + "," \
                                  "$\\\\mathrm{gp: \\\\ln{\\\\rho} (lc)}$,"
        else:
            baseline_params = ""
        instrument_params = re.sub('\\${sherlock:err_params}', errs_params, instrument_params)
        instrument_params = re.sub('\\${sherlock:baseline_params}', baseline_params, instrument_params)
        return instrument_params
