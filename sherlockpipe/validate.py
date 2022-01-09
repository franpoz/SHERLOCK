# from __future__ import print_function, absolute_import, division
import copy
import logging
import multiprocessing
import shutil
from multiprocessing import Pool
import traceback
import numpy as np
import matplotlib.pyplot as plt
import yaml
from astroquery.mast import TesscutClass
from lcbuilder.lcbuilder_class import LcBuilder
from lightkurve import TessLightCurve, KeplerLightCurve
from argparse import ArgumentParser
import sys
import pandas as pd
import os
import ast
import triceratops.triceratops as tr
from sherlockpipe.vet import Vetter
from lcbuilder.eleanor import maxsector


class Validator:
    """
    This class intends to provide a statistical validation tool for SHERLOCK Candidates.
    """
    def __init__(self, object_dir, validation_dir):
        self.object_dir = os.getcwd() if object_dir is None else object_dir
        self.data_dir = validation_dir

    def validate(self, candidate, star, cpus, contrast_curve_file, bins=100, scenarios=5, sigma_mode="flux_err"):
        """
        @param candidate: a candidate dataframe containing TICID, period, duration, t0, transits, depth, rp_rs, number,
        curve and sectors data.
        @param star: the star dataframe.
        @param cpus: the number of cpus to be used.
        @param contrast_curve_file: the auxiliary contrast curve file to give more information to the validation engine.
        @param bins: the number of bins to resize the light curve
        @param scenarios: the number of scenarios to compute the validation and get the average
        @param sigma_mode: whether to compute the sigma for the validation from the 'flux_err' or the 'binning'.
        """
        object_id = candidate["id"]
        period = candidate.loc[candidate['id'] == object_id]['period'].iloc[0]
        duration = candidate.loc[candidate['id'] == object_id]['duration'].iloc[0]
        t0 = candidate.loc[candidate['id'] == object_id]['t0'].iloc[0]
        transit_depth = candidate.loc[candidate['id'] == object_id]['depth'].iloc[0]
        run = int(candidate.loc[candidate['id'] == object_id]['number'].iloc[0])
        curve = int(candidate.loc[candidate['id'] == object_id]['curve'].iloc[0])
        rp_rstar = candidate.loc[candidate['id'] == object_id]['rp_rs'].iloc[0]
        a_rstar = candidate.loc[candidate['id'] == object_id]['a'].iloc[0] / star["R_star"]
        logging.info("------------------")
        logging.info("Candidate info")
        logging.info("------------------")
        logging.info("Period (d): %.2f", period)
        logging.info("Epoch (d): %.2f", t0)
        logging.info("Duration (min): %.2f", duration)
        logging.info("Depth (ppt): %.2f", transit_depth)
        logging.info("Run: %.0f", run)
        logging.info("Detrend curve: %.0f", curve)
        logging.info("Contrast curve file %s", contrast_curve_file)
        lc_file = "/" + str(run) + "/lc_" + str(curve) + ".csv"
        lc_file = self.data_dir + lc_file
        try:
            sectors_in = ast.literal_eval(str(((candidate.loc[candidate['id'] == object_id]['sectors']).values)[0]))
            if (type(sectors_in) == int) or (type(sectors_in) == float):
                sectors = [sectors_in]
            else:
                sectors = list(sectors_in)
        except:
            sectors = [0]
        self.data_dir = validation_dir
        object_id = object_id.iloc[0]
        try:
            Validator.execute_triceratops(cpus, validation_dir, object_id, sectors, lc_file, transit_depth,
                                          period, t0, duration, rp_rstar, a_rstar, bins, scenarios, sigma_mode, contrast_curve_file)
        except Exception as e:
            traceback.print_exc()
        # try:
        #     self.execute_vespa(cpus, validation_dir, object_id, sectors, lc_file, transit_depth, period, t0, duration, rprs)
        # except Exception as e:
        #     traceback.print_exc()

    @staticmethod
    def execute_triceratops(cpus, indir, object_id, sectors, lc_file, transit_depth, period, t0,
                            transit_duration, rp_rstar, a_rstar, bins, scenarios, sigma_mode, contrast_curve_file):
        """ Calculates probabilities of the signal being caused by any of the following astrophysical sources:
        TP No unresolved companion. Transiting planet with Porb around target star. (i, Rp)
        EB No unresolved companion. Eclipsing binary with Porb around target star. (i, qshort)
        EBx2P No unresolved companion. Eclipsing binary with 2 × Porb around target star. (i, qshort)
        PTP Unresolved bound companion. Transiting planet with Porb around primary star. (i, Rp, qlong)
        PEB Unresolved bound companion. Eclipsing binary with Porb around primary star. (i, qshort, qlong)
        PEBx2P Unresolved bound companion. Eclipsing binary with 2 × Porb around primary star. (i, qshort, qlong)
        STP Unresolved bound companion. Transiting planet with Porb around secondary star. (i, Rp, qlong)
        SEB Unresolved bound companion. Eclipsing binary with Porb around secondary star. (i, qshort, qlong)
        SEBx2P Unresolved bound companion. Eclipsing binary with 2 × Porb around secondary star. (i, qshort, qlong)
        DTP Unresolved background star. Transiting planet with Porb around target star. (i, Rp, simulated star)
        DEB Unresolved background star. Eclipsing binary with Porb around target star. (i, qshort, simulated star)
        DEBx2P Unresolved background star. Eclipsing binary with 2 × Porb around target star. (i, qshort, simulated star)
        BTP Unresolved background star. Transiting planet with Porb around background star. (i, Rp, simulated star)
        BEB Unresolved background star. Eclipsing binary with Porb around background star. (i, qshort, simulated star)
        BEBx2P Unresolved background star. Eclipsing binary with 2 × Porb around background star. (i, qshort, simulated star)
        NTP No unresolved companion. Transiting planet with Porb around nearby star. (i, Rp)
        NEB No unresolved companion. Eclipsing binary with Porb around nearby star. (i, qshort)
        NEBx2P No unresolved companion. Eclipsing binary with 2 × Porb around nearby star. (i, qshort)
        FPP = 1 - (TP + PTP + DTP)
        NFPP = NTP + NEB + NEBx2P
        Giacalone & Dressing (2020) define validated planets as TOIs with NFPP < 10−3 and FPP < 0.015 (or FPP ≤ 0.01,
        when rounding to the nearest percent)
        @param cpus: number of cpus to be used
        @param indir: root directory to store the results
        @param id_int: the object id for which the analysis will be run
        @param sectors: the sectors of the tic
        @param lc_file: the light curve source file
        @param transit_depth: the depth of the transit signal (ppts)
        @param period: the period of the transit signal /days)
        @param t0: the t0 of the transit signal (days)
        @param transit_duration: the duration of the transit signal (minutes)
        @param rp_rstar: radius of planet divided by radius of star
        @param a_rstar: semimajor axis divided by radius of star
        @param bins: the number of bins to average the folded curve
        @param scenarios: the number of scenarios to validate
        @param sigma_mode: the way to calculate the sigma for the validation ['flux_err' | 'binning']
        @param contrast_curve_file: the auxiliary contrast curve file to give more information to the validation engine.
        """
        save_dir = indir + "/triceratops"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        duration = transit_duration / 60 / 24
        logging.info("----------------------")
        logging.info("Validation procedures")
        logging.info("----------------------")
        logging.info("Pre-processing sectors")
        mission, mission_prefix, id_int = LcBuilder().parse_object_info(object_id)
        if mission == "TESS":
            sectors = np.array(sectors)
            sectors_cut = TesscutClass().get_sectors("TIC " + str(id_int))
            sectors_cut = np.array([sector_row["sector"] for sector_row in sectors_cut])
            if len(sectors) != len(sectors_cut):
                logging.warning("WARN: Some sectors were not found in TESSCUT")
                logging.warning("WARN: Sherlock sectors were: " + str(sectors))
                logging.warning("WARN: TESSCUT sectors were: " + str(sectors_cut))
            sectors = np.intersect1d(sectors, sectors_cut)
            if len(sectors) == 0:
                logging.warning("There are no available sectors to be validated, skipping TRICERATOPS.")
                return save_dir, None, None
        logging.info("Will execute validation for sectors: " + str(sectors))
        logging.info("Acquiring triceratops target")
        target = tr.target(ID=id_int, mission=mission, sectors=sectors)
        # TODO allow user input apertures
        logging.info("Reading apertures from directory")
        apertures = yaml.load(open(object_dir + "/apertures.yaml"), yaml.SafeLoader)
        apertures = apertures["sectors"]
        valid_apertures = {}
        for sector, aperture in apertures.items():
            if sector in sectors:
                valid_apertures[sector] = aperture
                target.plot_field(save=True, fname=save_dir + "/field_S" + str(sector), sector=sector, ap_pixels=aperture)
        apertures = np.array([aperture for sector, aperture in apertures.items()])
        valid_apertures = np.array([aperture for sector, aperture in valid_apertures.items()])
        depth = transit_depth / 1000
        if contrast_curve_file is not None:
            logging.info("Reading contrast curve %s", contrast_curve_file)
            plt.clf()
            cc = pd.read_csv(contrast_curve_file, header=None)
            sep, dmag = cc[0].values, cc[1].values
            plt.plot(sep, dmag, 'k-')
            plt.ylim(9, 0)
            plt.ylabel("$\\Delta K_s$", fontsize=20)
            plt.xlabel("separation ('')", fontsize=20)
            plt.savefig(save_dir + "/contrast_curve.png")
            plt.clf()
        logging.info("Calculating validation closest stars depths")
        target.calc_depths(depth, valid_apertures)
        target.stars.to_csv(save_dir + "/stars.csv", index=False)
        lc = pd.read_csv(lc_file, header=0)
        time, flux, flux_err = lc["#time"].values, lc["flux"].values, lc["flux_err"].values
        lc_len = len(time)
        zeros_lc = np.zeros(lc_len)
        logging.info("Preparing validation light curve for target")
        if mission == "TESS":
            lc = TessLightCurve(time=time, flux=flux, flux_err=flux_err, quality=zeros_lc)
        else:
            lc = KeplerLightCurve(time=time, flux=flux, flux_err=flux_err, quality=zeros_lc)
        lc.extra_columns = []
        fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        axs, bin_centers, bin_means, bin_errs = Vetter.compute_phased_values_and_fill_plot(object_id, axs, lc, period,
                                                                                           t0 + period / 2, depth,
                                                                                           duration, rp_rstar, a_rstar,
                                                                                           bins=bins)
        plt.savefig(save_dir + "/folded_curve.png")
        plt.clf()
        bin_centers = (bin_centers - 0.5) * period
        logging.info("Sigma mode is %s", sigma_mode)
        sigma = np.nanmean(bin_errs) if sigma_mode == 'binning' else np.nanmean(flux_err)
        logging.info("Computed folded curve sigma = %s", sigma)
        logging.info("Preparing validation processes inputs")
        input_n_times = [ValidatorInput(save_dir, copy.deepcopy(target), bin_centers, bin_means, sigma, period, depth,
                                        valid_apertures, value, contrast_curve_file)
                         for value in range(0, scenarios)]
        thread_validator = TriceratopsThreadValidator()
        logging.info("Start validation processes")
        with Pool(processes=cpus) as pool:
            validation_results = pool.map(thread_validator.validate, input_n_times)
        logging.info("Finished validation processes")
        fpp_sum = 0
        fpp2_sum = 0
        fpp3_sum = 0
        nfpp_sum = 0
        probs_total_df = None
        scenarios_num = len(validation_results[0][4])
        star_num = np.zeros((5, scenarios_num))
        u1 = np.zeros((5, scenarios_num))
        u2 = np.zeros((5, scenarios_num))
        fluxratio_EB = np.zeros((5, scenarios_num))
        fluxratio_comp = np.zeros((5, scenarios_num))
        target = input_n_times[0].target
        target.star_num = np.zeros(scenarios_num)
        target.u1 = np.zeros(scenarios_num)
        target.u2 = np.zeros(scenarios_num)
        target.fluxratio_EB = np.zeros(scenarios_num)
        target.fluxratio_comp = np.zeros(scenarios_num)
        logging.info("Computing final probabilities from the %s scenarios", scenarios)
        i = 0
        with open(save_dir + "/validation.csv", 'w') as the_file:
            the_file.write("scenario,FPP,NFPP,FPP2,FPP3+\n")
            for fpp, nfpp, fpp2, fpp3, probs_df, star_num_arr, u1_arr, u2_arr, fluxratio_EB_arr, fluxratio_comp_arr \
                    in validation_results:
                if probs_total_df is None:
                    probs_total_df = probs_df
                else:
                    probs_total_df = pd.concat((probs_total_df, probs_df))
                fpp_sum = fpp_sum + fpp
                fpp2_sum = fpp2_sum + fpp2
                fpp3_sum = fpp3_sum + fpp3
                nfpp_sum = nfpp_sum + nfpp
                star_num[i] = star_num_arr
                u1[i] = u1_arr
                u2[i] = u2_arr
                fluxratio_EB[i] = fluxratio_EB_arr
                fluxratio_comp[i] = fluxratio_comp_arr
                the_file.write(str(i) + "," + str(fpp) + "," + str(nfpp) + "," + str(fpp2) + "," + str(fpp3) + "\n")
                i = i + 1
            for i in range(0, scenarios_num):
                target.star_num[i] = np.mean(star_num[:, i])
                target.u1[i] = np.mean(u1[:, i])
                target.u2[i] = np.mean(u2[:, i])
                target.fluxratio_EB[i] = np.mean(fluxratio_EB[:, i])
                target.fluxratio_comp[i] = np.mean(fluxratio_comp[:, i])
            fpp_sum = fpp_sum / scenarios
            nfpp_sum = nfpp_sum / scenarios
            fpp2_sum = fpp2_sum / scenarios
            fpp3_sum = fpp3_sum / scenarios
            logging.info("---------------------------------")
            logging.info("Final probabilities computed")
            logging.info("---------------------------------")
            logging.info("FPP=%s", fpp_sum)
            logging.info("NFPP=%s", nfpp_sum)
            logging.info("FPP2(Lissauer et al, 2012)=%s", fpp2_sum)
            logging.info("FPP3+(Lissauer et al, 2012)=%s", fpp3_sum)
            the_file.write("MEAN" + "," + str(fpp_sum) + "," + str(nfpp_sum) + "," + str(fpp2_sum) + "," +
                           str(fpp3_sum))
        probs_total_df = probs_total_df.groupby("scenario", as_index=False).mean()
        probs_total_df["scenario"] = pd.Categorical(probs_total_df["scenario"], ["TP", "EB", "EBx2P", "PTP", "PEB",
                                                                                 "PEBx2P", "STP", "SEB", "SEBx2P",
                                                                                 "DTP", "DEB", "DEBx2P", "BTP", "BEB",
                                                                                 "BEBx2P", "NTP", "NEB", "NEBx2P"])
        probs_total_df = probs_total_df.sort_values("scenario")
        probs_total_df.to_csv(save_dir + "/validation_scenarios.csv", index=False)
        target.probs = probs_total_df
        # target.plot_fits(save=True, fname=save_dir + "/scenario_fits", time=lc.time.value, flux_0=lc.flux.value,
        #                  flux_err_0=sigma)
        return save_dir


    # def execute_vespa(self, cpus, indir, object_id, sectors, lc_file, transit_depth, period, epoch, duration, rprs):
    #     vespa_dir = indir + "/vespa/"
    #     if os.path.exists(vespa_dir):
    #         shutil.rmtree(vespa_dir, ignore_errors=True)
    #     os.mkdir(vespa_dir)
    #     lc = pd.read_csv(lc_file, header=0)
    #     time, flux, flux_err = lc["#time"].values, lc["flux"].values, lc["flux_err"].values
    #     lc_len = len(time)
    #     zeros_lc = np.zeros(lc_len)
    #     logging.info("Preparing validation light curve for target")
    #     lc = TessLightCurve(time=time, flux=flux, flux_err=flux_err, quality=zeros_lc)
    #     lc.extra_columns = []
    #     time_float = lc.time.value
    #     cadence_array = np.diff(time_float)
    #     cadence_array = cadence_array[~np.isnan(cadence_array)]
    #     cadence_array = cadence_array[cadence_array > 0]
    #     cadence_days = np.nanmedian(cadence_array)
    #     lc = lc.fold(period=period, epoch_time=epoch, normalize_phase=True)
    #     folded_plot_range = duration / 60 / 24 / 2 / period * 5
    #     inner_folded_range_args = np.where(
    #         (0 - folded_plot_range < lc.time.value) & (lc.time.value < 0 + folded_plot_range))
    #     lc = lc[inner_folded_range_args]
    #     lc.time = lc.time * period
    #     bin_means, bin_edges, binnumber = stats.binned_statistic(lc.time.value, lc.flux.value, statistic='mean',
    #                                                              bins=500)
    #     bin_means_err, bin_edges_err, binnumber_err = stats.binned_statistic(lc.time.value, lc.flux_err.value, statistic='mean',
    #                                                              bins=500)
    #     bin_width = (bin_edges[1] - bin_edges[0])
    #     bin_centers = bin_edges[1:] - bin_width / 2
    #     lc.plot()
    #     plt.title("Target " + str(object_id))
    #     plt.savefig(vespa_dir + "/folded_curve.png")
    #     plt.plot(bin_centers, bin_means)
    #     plt.title("Target " + str(object_id))
    #     plt.xlabel("Time")
    #     plt.ylabel("Flux")
    #     plt.savefig(vespa_dir + "/folded_curve_binned.png")
    #     lc_df = pandas.DataFrame(columns=['days_from_midtransit', 'flux', 'flux_err'])
    #     lc_df['days_from_midtransit'] = bin_centers
    #     lc_df['flux'] = bin_means
    #     lc_df['flux_err'] = bin_means_err
    #     lc_df.to_csv(vespa_dir + "/lc.csv", index=False)
    #     star_df = pd.read_csv(self.object_dir + "/params_star.csv")
    #     with open(vespa_dir + "fpp.ini", 'w+') as f:
    #         f.write("name = " + str(object_id) + "\n")
    #         f.write("ra = " + str(star_df.at[0, "ra"]) + "\n")
    #         f.write("dec = " + str(star_df.at[0, "dec"]) + "\n")
    #         f.write("period = " + str(period) + "\n")
    #         f.write("rprs = " + str(rprs) + "\n")
    #         # TODO rewrite lc file to match vespa expected
    #         f.write("photfile = " + vespa_dir + "/lc.csv" + "\n")
    #         f.write("band = J" + "\n")
    #         #TODO cadence
    #         f.write("cadence = " + str(cadence_days) + "\n")
    #         f.write("[constraints]" + "\n")
    #         f.write("maxrad = 12" + "\n")
    #         f.write("secthresh = 1e-4" + "\n")
    #     try:
    #         self.isochrones_starfit(vespa_dir, star_df)
    #         f = FPPCalculation.from_ini(vespa_dir, ini_file="fpp.ini",
    #                                     recalc=True,
    #                                     refit_trap=False,
    #                                     n=20000)
    #         f.trsig.MCMC(refit=True)
    #         f.trsig.save(os.path.join(vespa_dir, 'trsig.pkl'))
    #         trap_corner_file = os.path.join(vespa_dir, 'trap_corner.png')
    #         if not os.path.exists(trap_corner_file) or args.refit_trsig:
    #             f.trsig.corner(outfile=trap_corner_file)
    #         # Including artificial models
    #         boxmodel = BoxyModel(args.artificial_prior, f['pl'].stars.slope.max())
    #         longmodel = LongModel(args.artificial_prior, f['pl'].stars.duration.quantile(0.99))
    #         f.add_population(boxmodel)
    #         f.add_population(longmodel)
    #         f.FPPplots(recalc_lhood=args.recalc_lhood)
    #         logger.info('Re-fitting trapezoid MCMC model...')
    #         f.bootstrap_FPP(1)
    #         for mult, Model in zip(['single', 'binary', 'triple'],
    #                                [StarModel, BinaryStarModel, TripleStarModel]):
    #             starmodel_file = os.path.join(vespa_dir, '{}_starmodel_{}.h5'.format(args.ichrone, mult))
    #             corner_file1 = os.path.join(vespa_dir,
    #                                         '{}_corner_{}_physical.png'.format(args.ichrone, mult))
    #             corner_file2 = os.path.join(vespa_dir,
    #                                         '{}_corner_{}_observed.png'.format(args.ichrone, mult))
    #             if not os.path.exists(corner_file1) or not os.path.exists(corner_file2):
    #                 logger.info('Making StarModel corner plots...')
    #                 starmodel = Model.load_hdf(starmodel_file)
    #                 corner_base = os.path.join(vespa_dir,
    #                                            '{}_corner_{}'.format(args.ichrone, mult))
    #                 starmodel.corner_plots(corner_base)
    #             logger.info('Bootstrap results ({}) written to {}.'.format(1,
    #                                                                        os.path.join(os.path.abspath(vespa_dir),
    #                                                                                     'results_bootstrap.txt')))
    #         logger.info('VESPA FPP calculation successful. ' +
    #                     'Results/plots written to {}.'.format(os.path.abspath(vespa_dir)))
    #         print('VESPA FPP for {}: {}'.format(f.name, f.FPP()))
    #         fpp_df = pandas.DataFrame(columns=['fpp'])
    #         fpp_df.append({'name': f.name, 'fpp': f.FPP()})
    #         fpp_df.to_csv(vespa_dir + "fpp.csv", index=False)
    #     except KeyboardInterrupt:
    #         raise
    #     except:
    #         logger.error('FPP calculation failed for {}.'.format(vespa_dir), exc_info=True)
    #
    # def isochrones_starfit(self, vespa_dir, star_df):
    #     with open(vespa_dir + "star.ini", 'w+') as f:
    #         feh = star_df.at[0, "feh"]
    #         feh_err = star_df.at[0, "feh_err"]
    #         logg = star_df.at[0, "logg"]
    #         logg_err = star_df.at[0, "logg_err"]
    #         j = star_df.at[0, "j"]
    #         j_err = star_df.at[0, "j_err"]
    #         h = star_df.at[0, "h"]
    #         h_err = star_df.at[0, "h_err"]
    #         k = star_df.at[0, "k"]
    #         k_err = star_df.at[0, "k_err"]
    #         kp = star_df.at[0, "kp"]
    #         f.write("Teff = " + str(star_df.at[0, "Teff_star"]) + ", " + str(star_df.at[0, "Teff_star"] * 0.1) + "\n")
    #         if feh is not None and not np.isnan(feh) and feh_err is not None and not np.isnan(feh_err):
    #             f.write("feh = " + str(star_df.at[0, "feh"]) + ", " + str(star_df.at[0, "feh_err"]) + "\n")
    #         elif feh is not None and not np.isnan(feh):
    #             f.write("feh = " + str(star_df.at[0, "feh"]) + "\n")
    #         if logg is not None and not np.isnan(logg) and logg_err is not None and not np.isnan(logg_err):
    #             f.write("logg = " + str(star_df.at[0, "logg"]) + ", " + str(star_df.at[0, "logg_err"]) + "\n")
    #         elif logg is not None and not np.isnan(logg):
    #             f.write("logg = " + str(star_df.at[0, "logg"]) + "\n")
    #         f.write("[sherlock]\n")
    #         if j is not None and not np.isnan(j) and j_err is not None and not np.isnan(j_err):
    #             f.write("J = " + str(star_df.at[0, "j"]) + ", " + str(star_df.at[0, "j_err"]) + "\n")
    #         elif j is not None and not np.isnan(j):
    #             f.write("J = " + str(star_df.at[0, "j"]) + "\n")
    #         if h is not None  and not np.isnan(h) and h_err is not None and not np.isnan(h_err):
    #             f.write("H = " + str(star_df.at[0, "h"]) + ", " + str(star_df.at[0, "h_err"]) + "\n")
    #         elif h is not None and not np.isnan(h):
    #             f.write("H = " + str(star_df.at[0, "h"]) + "\n")
    #         if k is not None and not np.isnan(k) and logg_err is not None and not np.isnan(logg_err):
    #             f.write("K = " + str(star_df.at[0, "k"]) + ", " + str(star_df.at[0, "k_err"]) + "\n")
    #         elif k is not None:
    #             f.write("K = " + str(star_df.at[0, "kp"]) + "\n")
    #         if kp is not None and not np.isnan(kp):
    #             f.write("Kepler = " + str(star_df.at[0, "kp"]) + "\n")
    #     starfit(
    #         vespa_dir,
    #         multiplicities=["single", "binary", "triple"],
    #         models="mist",
    #         use_emcee=True,
    #         plot_only=False,
    #         overwrite=False,
    #         verbose=False,
    #         logger=None,
    #         starmodel_type=None,
    #         skip_initial_state_check=True,
    #         ini_file="star.ini",
    #         no_plots=False,
    #         bands=None
    #     )


class TriceratopsThreadValidator:
    """
    Used to run a single scenario validation with TRICERATOPS
    """
    def __init__(self) -> None:
        super().__init__()

    def validate(self, input):
        """
        Computes the input scenario FPP and NFPP. In addition, FPP2 and FPP3+, from the probability boost proposed in
        Lissauer et al. (2012) eq. 8 and 9 for systems where one or more planets have already been confirmed, are also
        provided just in case they are useful so they don't need to be manually calculated.
        @param input: ValidatorInput
        @return: the FPP values, the probabilities dataframe and additional target values.
        """
        input.target.calc_depths(tdepth=input.depth, all_ap_pixels=input.apertures)
        input.target.calc_probs(time=input.time, flux_0=input.flux, flux_err_0=input.sigma, P_orb=input.period,
                                contrast_curve_file=input.contrast_curve, parallel=True)
        fpp2 = 1 - 25 * (1 - input.target.FPP) / (25 * (1 - input.target.FPP) + input.target.FPP)
        fpp3 = 1 - 50 * (1 - input.target.FPP) / (50 * (1 - input.target.FPP) + input.target.FPP)
        input.target.probs.to_csv(input.save_dir + "/validation_" + str(input.run) + "_scenarios.csv", index=False)
        input.target.plot_fits(save=True, fname=input.save_dir + "/scenario_" + str(input.run) + "_fits",
                         time=input.time, flux_0=input.flux, flux_err_0=input.sigma)
        return input.target.FPP, input.target.NFPP, fpp2, fpp3, input.target.probs, input.target.star_num, \
               input.target.u1, input.target.u2, input.target.fluxratio_EB, input.target.fluxratio_comp


class ValidatorInput:
    """
    Wrapper class for input arguments of TriceratopsThreadValidator.
    """
    def __init__(self, save_dir, target, time, flux, sigma, period, depth, apertures, run, contrast_curve):
        self.save_dir = save_dir
        self.target = target
        self.time = time
        self.flux = flux
        self.sigma = sigma
        self.period = period
        self.depth = depth
        self.apertures = apertures
        self.run = run
        self.contrast_curve = contrast_curve


if __name__ == '__main__':
    ap = ArgumentParser(description='Validation of Sherlock objects of interest')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--candidate', type=int, default=None, help="The candidate signal to be used.", required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    ap.add_argument('--cpus', type=int, default=None, help="The number of CPU cores to be used.", required=False)
    ap.add_argument('--bins', type=int, default=100, help="The number of bins to be used for the folded curve "
                                                          "validation.", required=False)
    ap.add_argument('--sigma_mode', type=str, default='flux_err', help="The way to calculate the sigma value for the "
                                                                       "validation. [flux_err|binning]", required=False)
    ap.add_argument('--scenarios', type=int, default=5, help="The number of scenarios to be used for the validation",
                    required=False)
    ap.add_argument('--contrast_curve', type=str, default=None, help="The contrast curve in csv format.",
                    required=False)
    args = ap.parse_args()
    index = 0
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    validation_dir = object_dir + "/validation_" + str(index)
    while os.path.exists(validation_dir) or os.path.isdir(validation_dir):
        validation_dir = object_dir + "/validation_" + str(index)
        index = index + 1
    os.mkdir(validation_dir)
    validator = Validator(object_dir, validation_dir)
    file_dir = validation_dir + "/validation.log"
    if os.path.exists(file_dir):
        os.remove(file_dir)
    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger()
    while len(logger.handlers) > 0:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.FileHandler(file_dir)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.info("Starting validation")
    star_df = pd.read_csv(validator.object_dir + "/params_star.csv")
    if args.candidate is None:
        logging.info("Reading validation input from properties file: %s", args.properties)
        user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
        candidate = pd.DataFrame(columns=['id', 'period', 'depth', 't0', 'sectors', 'ffi', 'number', 'lc'])
        candidate = candidate.append(user_properties, ignore_index=True)
        candidate['id'] = star_df.iloc[0]["obj_id"]
    else:
        candidate_selection = int(args.candidate)
        candidates = pd.read_csv(validator.object_dir + "/candidates.csv")
        if candidate_selection < 1 or candidate_selection > len(candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        candidates = candidates.rename(columns={'Object Id': 'id'})
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate['number'] = [candidate_selection]
        validator.data_dir = validator.object_dir
        logging.info("Selected signal number " + str(candidate_selection))
    if args.cpus is None:
        cpus = multiprocessing.cpu_count() - 1
    else:
        cpus = args.cpus
    validator.validate(candidate, star_df.iloc[0], cpus, args.contrast_curve, args.bins, args.scenarios, args.sigma_mode)
