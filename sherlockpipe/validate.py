# from __future__ import print_function, absolute_import, division
import copy
import logging
import multiprocessing
import shutil
from multiprocessing import Pool
import traceback
import lightkurve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
from astroquery.mast import TesscutClass
from lcbuilder.lcbuilder_class import LcBuilder
from lightkurve import TessLightCurve
from argparse import ArgumentParser
from scipy import stats
from sherlockpipe import constants as const
from sherlockpipe import eleanor
import six
import sys
import sherlockpipe.LATTE
sys.modules['astropy.extern.six'] = six
sys.modules['LATTE'] = sherlockpipe.LATTE

matplotlib.use('Agg')
import pandas as pd
import os
import ast
from sherlockpipe.LATTE import LATTEutils, LATTEbrew
from os import path
import triceratops.triceratops as tr
from matplotlib import cm, ticker
from triceratops.likelihoods import (simulate_TP_transit, simulate_EB_transit)
from triceratops.funcs import renorm_flux
from astropy import constants
from sherlockpipe.eleanor import maxsector

'''WATSON: Verboseless Vetting and Adjustments of Transits for Sherlock Objects of iNterest
This class intends to provide a inspection and transit fitting tool for SHERLOCK Candidates.
'''
# get the system path
syspath = str(os.path.abspath(LATTEutils.__file__))[0:-14]
# ---------

# --- IMPORTANT TO SET THIS ----
out = 'pipeline'  # or TALK or 'pipeline'
ttran = 0.1
resources_dir = path.join(path.dirname(__file__))

class Validator:
    def __init__(self, object_dir):
        self.object_dir = os.getcwd() if object_dir is None else object_dir
        self.data_dir = self.object_dir
        self.validation_runs = 5

    def validate(self, candidate, cpus):
        object_id = candidate["TICID"]
        period = candidate.loc[candidate['TICID'] == object_id]['period'].iloc[0]
        duration = candidate.loc[candidate['TICID'] == object_id]['duration'].iloc[0]
        t0 = candidate.loc[candidate['TICID'] == object_id]['t0'].iloc[0]
        transit_list = ast.literal_eval(((candidate.loc[candidate['TICID'] == object_id]['transits']).values)[0])
        transit_depth = candidate.loc[candidate['TICID'] == object_id]['depth'].iloc[0]
        rprs = candidate.loc[candidate['TICID'] == object_id]['rp_rs'].iloc[0]
        candidate_row = candidate.iloc[0]
        if candidate_row["number"] is None or np.isnan(candidate_row["number"]):
            lc_file = "/" + candidate_row["lc"]
        else:
            lc_file = "/" + str(candidate_row["number"]) + "/lc_" + str(candidate_row["curve"]) + ".csv"
        lc_file = self.data_dir + lc_file
        try:
            sectors_in = ast.literal_eval(str(((candidate.loc[candidate['TICID'] == object_id]['sectors']).values)[0]))
            if (type(sectors_in) == int) or (type(sectors_in) == float):
                sectors = [sectors_in]
            else:
                sectors = list(sectors_in)
        except:
            sectors = [0]
        index = 0
        validation_dir = self.data_dir + "/validation_" + str(index)
        while os.path.exists(validation_dir) or os.path.isdir(validation_dir):
            validation_dir = self.data_dir + "/validation_" + str(index)
            index = index + 1
        os.mkdir(validation_dir)
        self.data_dir = validation_dir
        object_id = object_id[0]
        mission, mission_prefix, id_int = LcBuilder().parse_object_info(object_id)
        try:
            self.execute_triceratops(cpus, validation_dir, mission, str(id_int), sectors, lc_file, transit_depth,
                                     period, t0, duration)
        except Exception as e:
            traceback.print_exc()
        # try:
        #     self.execute_vespa(cpus, validation_dir, object_id, sectors, lc_file, transit_depth, period, t0, duration, rprs)
        # except Exception as e:
        #     traceback.print_exc()

    def execute_triceratops(self, cpus, indir, mission, id_int, sectors, lc_file, transit_depth, period, t0,
                            transit_duration):
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
        @param id_int: the tess object id for which the analysis will be run
        @param sectors: the sectors of the tic
        @param lc_file: the light curve source file
        @param transit_depth: the depth of the transit signal (ppts)
        @param period: the period of the transit signal /days)
        @param t0: the t0 of the transit signal (days)
        @param transit_duration: the duration of the transit signal (minutes)
        """
        save_dir = indir + "/triceratops"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        duration = transit_duration / 60 / 24
        if mission != "TESS":
            star_df = pd.read_csv(self.object_dir + "/params_star.csv")
            star = eleanor.multi_sectors(coords=(star_df.at[0, "ra"], star_df.at[0, "dec"]), sectors='all',
                                         tesscut_size=31, post_dir=const.USER_HOME_ELEANOR_CACHE)
            if len(star) == 0 or star[0].tic is None:
                raise ValueError("Can't validate target as there is no data from TESS FFIs")
            tic = star[0].tic
            sectors = np.array([target.sector for target in star])
        else:
            sectors = np.array(sectors)
            tic = id_int
            sectors_cut = TesscutClass().get_sectors("TIC " + str(tic))
            sectors_cut = np.array([sector_row["sector"] for sector_row in sectors_cut])
            if len(sectors) != len(sectors_cut):
                logging.warning("WARN: Some sectors were not found in TESSCUT")
                logging.warning("WARN: Sherlock sectors were: " + str(sectors))
                logging.warning("WARN: TESSCUT sectors were: " + str(sectors_cut))
            sectors = np.intersect1d(sectors, sectors_cut)
            eleanor_sectors = sectors[sectors <= maxsector.maxsector]
            if len(sectors) != len(eleanor_sectors):
                logging.warning("WARN: Some sectors were not found in ELEANOR")
                logging.warning("WARN: Sectors were: " + str(sectors))
                logging.warning("WARN: ELEANOR (where maxsector is %s) sectors were: " + str(eleanor_sectors),
                                maxsector.maxsector)
            sectors = eleanor_sectors
            if len(sectors) == 0:
                logging.warning("There are no available sectors to be validated, skipping TRICERATOPS.")
                return save_dir, None, None
        target = tr.target(ID=tic, sectors=sectors)
        logging.info("Will execute validation for sectors: " + str(sectors))
        # TODO allow user input apertures
        tpfs = lightkurve.search_targetpixelfile("TIC " + str(tic), mission="TESS", cadence="short", sector=sectors.tolist())\
            .download_all()
        star = eleanor.multi_sectors(tic=tic, sectors=sectors, tesscut_size=11, post_dir=const.USER_HOME_ELEANOR_CACHE)
        apertures = []
        sector_num = 0
        logging.info("Calculating validation masks")
        for s in star:
            if tpfs is None:
                target_data = eleanor.TargetData(s, height=11, width=11)
                pipeline_mask = target_data.aperture.astype(bool)
                column = s.position_on_chip[0]
                row = s.position_on_chip[1]
            else:
                tpf_idx = [data.sector if data.sector == s.sector else -1 for data in tpfs.data]
                tpf = tpfs[np.where(tpf_idx > np.zeros(len(tpf_idx)))[0][0]]
                pipeline_mask = tpfs[np.where(tpf_idx > np.zeros(len(tpf_idx)))[0][0]].pipeline_mask
                column = tpf.column
                row = tpf.row
            pipeline_mask = np.transpose(pipeline_mask)
            pipeline_mask_triceratops = np.zeros((len(pipeline_mask[0]), len(pipeline_mask[:][0]), 2))
            for i in range(0, len(pipeline_mask[0])):
                for j in range(0, len(pipeline_mask[:][0])):
                    pipeline_mask_triceratops[i, j] = [column + i, row + j]
            pipeline_mask_triceratops[~pipeline_mask] = None
            aperture = []
            for i in range(0, len(pipeline_mask_triceratops[0])):
                for j in range(0, len(pipeline_mask_triceratops[:][0])):
                    if not np.isnan(pipeline_mask_triceratops[i, j]).any():
                        aperture.append(pipeline_mask_triceratops[i, j])
            apertures.append(aperture)
            logging.info("Saving validation mask plot for sector %s", s.sector)
            target.plot_field(save=True, fname=save_dir + "/field_S" + str(s.sector), sector=s.sector,
                            ap_pixels=aperture)
            sector_num = sector_num + 1
        apertures = np.array(apertures)
        depth = transit_depth / 1000
        logging.info("Calculating validation closest stars depths")
        target.calc_depths(depth, apertures)
        target.stars.to_csv(save_dir + "/stars.csv", index=False)
        lc = pd.read_csv(lc_file, header=0)
        time, flux, flux_err = lc["#time"].values, lc["flux"].values, lc["flux_err"].values
        lc_len = len(time)
        zeros_lc = np.zeros(lc_len)
        logging.info("Preparing validation light curve for target")
        lc = TessLightCurve(time=time, flux=flux, flux_err=flux_err, quality=zeros_lc)
        lc.extra_columns = []
        lc = lc.fold(period=period, epoch_time=t0, normalize_phase=True)
        folded_plot_range = duration / 2 / period * 5
        inner_folded_range_args = np.where((0 - folded_plot_range < lc.time.value) & (lc.time.value < 0 + folded_plot_range))
        lc = lc[inner_folded_range_args]
        lc.time = lc.time * period
        bin_means, bin_edges, binnumber = stats.binned_statistic(lc.time.value, lc.flux.value, statistic='mean', bins=500)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        lc.plot()
        plt.title("TIC " + str(id_int))
        plt.savefig(save_dir + "/folded_curve.png")
        plt.plot(bin_centers, bin_means)
        plt.title("TIC " + str(id_int))
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.savefig(save_dir + "/folded_curve_binned.png")
        sigma = np.nanmean(lc.flux_err)
        logging.info("Preparing validation processes inputs")
        input_n_times = [ValidatorInput(save_dir, copy.deepcopy(target), bin_centers, bin_means, sigma, period, depth,
                                        apertures, value)
                         for value in range(0, self.validation_runs)]
        validator = TriceratopsThreadValidator()
        logging.info("Start validation processes")
        with Pool(processes=cpus) as pool:
            validation_results = pool.map(validator.validate, input_n_times)
        logging.info("Finished validation processes")
        fpp_sum = 0
        nfpp_sum = 0
        probs_total_df = None
        scenarios_num = len(validation_results[0][2])
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
        logging.info("Computing final probabilities from the %s scenarios", self.validation_runs)
        i = 0
        for fpp, nfpp, probs_df, star_num_arr, u1_arr, u2_arr, fluxratio_EB_arr, fluxratio_comp_arr in validation_results:
            if probs_total_df is None:
                probs_total_df = probs_df
            else:
                probs_total_df = pd.concat((probs_total_df, probs_df))
            fpp_sum = fpp_sum + fpp
            nfpp_sum = nfpp_sum + nfpp
            star_num[i] = star_num_arr
            u1[i] = u1_arr
            u2[i] = u2_arr
            fluxratio_EB[i] = fluxratio_EB_arr
            fluxratio_comp[i] = fluxratio_comp_arr
            i = i + 1
        for i in range(0, scenarios_num):
            target.star_num[i] = np.mean(star_num[:, i])
            target.u1[i] = np.mean(u1[:, i])
            target.u2[i] = np.mean(u2[:, i])
            target.fluxratio_EB[i] = np.mean(fluxratio_EB[:, i])
            target.fluxratio_comp[i] = np.mean(fluxratio_comp[:, i])
        with open(save_dir + "/validation.csv", 'w') as the_file:
            the_file.write("FPP,NFPP\n")
            the_file.write(str(fpp_sum / self.validation_runs) + "," + str(nfpp_sum / self.validation_runs))
        probs_total_df = probs_total_df.groupby("scenario", as_index=False).mean()
        probs_total_df["scenario"] = pd.Categorical(probs_total_df["scenario"], ["TP", "EB", "EBx2P", "PTP", "PEB", "PEBx2P",
                                                                         "STP", "SEB", "SEBx2P", "DTP", "DEB", "DEBx2P",
                                                                         "BTP", "BEB", "BEBx2P", "NTP", "NEB", "NEBx2P"])
        probs_total_df = probs_total_df.sort_values("scenario")
        probs_total_df.to_csv(save_dir + "/validation_scenarios.csv", index=False)
        target.probs = probs_total_df
        # target.plot_fits(save=True, fname=save_dir + "/scenario_fits", time=lc.time.value, flux_0=lc.flux.value,
        #                  flux_err_0=sigma)
        return save_dir, star[0].coords[0], star[0].coords[1]

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

    @staticmethod
    def plot_fits(target, save_file, time: np.ndarray, flux_0: np.ndarray, sigma_0: float):
        """
        Plots light curve for best fit instance of each scenario.
        Args:
            time (numpy array): Time of each data point
                                [days from transit midpoint].
            flux_0 (numpy array): Normalized flux of each data point.
            sigma_0 (numpy array): Uncertainty of flux.
        """
        scenario_idx = target.probs[target.probs["ID"] != 0].index.values
        df = target.probs[target.probs["ID"] != 0]
        star_num = target.star_num[target.probs["ID"] != 0]
        u1s = target.u1[target.probs["ID"] != 0]
        u2s = target.u2[target.probs["ID"] != 0]
        fluxratios_EB = target.fluxratio_EB[target.probs["ID"] != 0]
        fluxratios_comp = target.fluxratio_comp[target.probs["ID"] != 0]

        model_time = np.linspace(min(time), max(time), 100)

        f, ax = plt.subplots(
            len(df)//3, 3, figsize=(12, len(df)//3*4), sharex=True
            )
        G = constants.G.cgs.value
        M_sun = constants.M_sun.cgs.value
        for i in range(len(df)//3):
            for j in range(3):
                if i == 0:
                    k = j
                else:
                    k = 3*i+j
                # subtract flux from other stars in the aperture
                idx = np.argwhere(
                    target.stars["ID"].values == str(df["ID"].values[k])
                    )[0, 0]
                flux, sigma = renorm_flux(
                    flux_0, sigma_0, target.stars["fluxratio"].values[idx]
                    )
                # all TPs
                if j == 0:
                    if star_num[k] == 1:
                        comp = False
                    else:
                        comp = True
                    a = (
                        (G*df["M_s"].values[k]*M_sun)/(4*np.pi**2)
                        * (df['P_orb'].values[k]*86400)**2
                        )**(1/3)
                    u1, u2 = u1s[k], u2s[k]
                    best_model = simulate_TP_transit(
                        model_time,
                        df['R_p'].values[k], df['P_orb'].values[k],
                        df['inc'].values[k], a, df["R_s"].values[k],
                        u1, u2, fluxratios_comp[k], comp
                        )
                # all small EBs
                elif j == 1:
                    if star_num[k] == 1:
                        comp = False
                    else:
                        comp = True
                    mass = df["M_s"].values[k] + df["M_EB"].values[k]
                    a = (
                        (G*mass*M_sun)/(4*np.pi**2)
                        * (df['P_orb'].values[k]*86400)**2
                        )**(1/3)
                    u1, u2 = u1s[k], u2s[k]
                    best_model = simulate_EB_transit(
                        model_time,
                        df["R_EB"].values[k], fluxratios_EB[k],
                        df['P_orb'].values[k], df['inc'].values[k],
                        a, df["R_s"].values[k], u1, u2,
                        fluxratios_comp[k], comp
                        )[0]
                # all twin EBs
                elif j == 2:
                    if star_num[k] == 1:
                        comp = False
                    else:
                        comp = True
                    mass = df["M_s"].values[k] + df["M_EB"].values[k]
                    a = (
                        (G*mass*M_sun)/(4*np.pi**2)
                        * (df['P_orb'].values[k]*86400)**2
                        )**(1/3)
                    u1, u2 = u1s[k], u2s[k]
                    best_model = simulate_EB_transit(
                        model_time,
                        df["R_EB"].values[k], fluxratios_EB[k],
                        df['P_orb'].values[k], df['inc'].values[k],
                        a, df["R_s"].values[k], u1, u2,
                        fluxratios_comp[k], comp
                        )[0]

                y_formatter = ticker.ScalarFormatter(useOffset=False)
                ax[i, j].yaxis.set_major_formatter(y_formatter)
                ax[i, j].errorbar(
                    time, flux, sigma, fmt=".",
                    color="blue", alpha=0.1, zorder=0
                    )
                ax[i, j].plot(
                    model_time, best_model, "k-", lw=5, zorder=2
                    )
                ax[i, j].set_ylabel("normalized flux", fontsize=12)
                ax[i, j].annotate(
                    str(df["ID"].values[k]), xy=(0.05, 0.92),
                    xycoords="axes fraction", fontsize=12
                    )
                ax[i, j].annotate(
                    str(df["scenario"].values[k]), xy=(0.05, 0.05),
                    xycoords="axes fraction", fontsize=12
                    )
        ax[len(df)//3-1, 0].set_xlabel(
            "days from transit center", fontsize=12
            )
        ax[len(df)//3-1, 1].set_xlabel(
            "days from transit center", fontsize=12
            )
        ax[len(df)//3-1, 2].set_xlabel(
            "days from transit center", fontsize=12
            )
        plt.tight_layout()
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()
        return

    def show_candidates(self):
        self.candidates = pd.read_csv(self.object_dir + "/candidates.csv")
        self.candidates.index = np.arange(1, len(self.candidates) + 1)
        logging.info("Suggested candidates are:")
        logging.info(self.candidates.to_markdown(index=True))
        pass

    def demand_candidate_selection(self):
        user_input = input("Select candidate number to be examined and fit: ")
        if user_input.startswith("q"):
            raise SystemExit("User quitted")
        self.candidate_selection = int(user_input)
        if self.candidate_selection < 1 or self.candidate_selection > len(self.candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        self.data_dir = self.object_dir + "/" + str(self.candidate_selection)

class TriceratopsThreadValidator:
    def __init__(self) -> None:
        super().__init__()

    def validate(self, input):
        input.target.calc_depths(tdepth=input.depth, all_ap_pixels=input.apertures)
        input.target.calc_probs(time=input.time, flux_0=input.flux, flux_err_0=input.sigma, P_orb=input.period)
        with open(input.save_dir + "/validation_" + str(input.run) + ".csv", 'w') as the_file:
            the_file.write("FPP,NFPP\n")
            the_file.write(str(input.target.FPP) + "," + str(input.target.NFPP))
        input.target.probs.to_csv(input.save_dir + "/validation_" + str(input.run) + "_scenarios.csv", index=False)
        input.target.plot_fits(save=True, fname=input.save_dir + "/scenario_" + str(input.run) + "_fits",
                         time=input.time, flux_0=input.flux, flux_err_0=input.sigma)
        return input.target.FPP, input.target.NFPP, input.target.probs, input.target.star_num, input.target.u1, \
               input.target.u2, input.target.fluxratio_EB, input.target.fluxratio_comp

class ValidatorInput:
    def __init__(self, save_dir, target, time, flux, sigma, period, depth, apertures, run):
        self.save_dir = save_dir
        self.target = target
        self.time = time
        self.flux = flux
        self.sigma = sigma
        self.period = period
        self.depth = depth
        self.apertures = apertures
        self.run = run

if __name__ == '__main__':
    ap = ArgumentParser(description='Vetting of Sherlock objects of interest')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--candidate', type=int, default=None, help="The candidate signal to be used.", required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    ap.add_argument('--cpus', type=int, default=None, help="The number of CPU cores to be used.", required=False)
    args = ap.parse_args()
    validator = Validator(args.object_dir)
    file_dir = validator.object_dir + "/validation.log"
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
    if args.candidate is None:
        user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
        candidate = pd.DataFrame(columns=['id', 'transits', 'sectors', 'FFI'])
        candidate = candidate.append(user_properties, ignore_index=True)
        cpus = user_properties["settings"]["cpus"]
    else:
        candidate_selection = int(args.candidate)
        candidates = pd.read_csv(validator.object_dir + "/candidates.csv")
        if candidate_selection < 1 or candidate_selection > len(candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        candidates = candidates.rename(columns={'Object Id': 'TICID'})
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate['number'] = [candidate_selection]
        validator.data_dir = validator.object_dir
        logging.info("Selected signal number " + str(candidate_selection))
        if args.cpus is None:
            cpus = multiprocessing.cpu_count() - 1
        else:
            cpus = args.cpus
    validator.validate(candidate, cpus)
