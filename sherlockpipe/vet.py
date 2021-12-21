# from __future__ import print_function, absolute_import, division
import logging
import multiprocessing
import shutil
import types
from distutils.dir_util import copy_tree
from pathlib import Path
import traceback

import batman
import foldedleastsquares
import lightkurve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from lcbuilder.constants import CUTOUT_SIZE
from lcbuilder.photometry.aperture_extractor import ApertureExtractor
from lightkurve import TessLightCurve
from matplotlib.colorbar import Colorbar
from matplotlib import patches
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table
from astropy.io import ascii
import astropy.visualization as stretching
from argparse import ArgumentParser

from scipy import stats

from sherlockpipe import tpfplotter
import six
import sys
import sherlockpipe.LATTE
sys.modules['astropy.extern.six'] = six
sys.modules['LATTE'] = sherlockpipe.LATTE

matplotlib.use('Agg')
import pandas as pd
import os
from os.path import exists
import ast
import csv
from sherlockpipe.LATTE import LATTEutils, LATTEbrew
from os import path
from math import ceil


# get the system path
syspath = str(os.path.abspath(LATTEutils.__file__))[0:-14]
# ---------

# --- IMPORTANT TO SET THIS ----
out = 'pipeline'  # or TALK or 'pipeline'
ttran = 0.1
resources_dir = path.join(path.dirname(__file__))


class Vetter:
    """
    Provides transiting candidate vetting information like centroids and spaceship motion, momentum dumps, neighbours
    curves inspection and more to give a deeper insight on the quality of the candidate signal.
    """
    def __init__(self, object_dir, validate):
        self.args = types.SimpleNamespace()
        self.args.noshow = True
        self.args.north = False
        self.args.o = True
        self.args.mpi = False
        self.args.auto = True
        self.args.save = True
        self.args.nickname = ""  # TODO do we set the sherlock id?
        self.args.FFI = False  # TODO get this from input
        self.args.targetlist = "best_signal_latte_input.csv"
        self.args.new_path = ""  # TODO check what to do with this
        self.object_dir = os.getcwd() if object_dir is None else object_dir
        self.latte_dir = str(Path.home()) + "/.sherlockpipe/latte/"
        if not os.path.exists(self.latte_dir):
            os.mkdir(self.latte_dir)
        self.data_dir = self.object_dir
        self.validation_runs = 5
        self.validate = validate

    def update(self):
        """
        Updates the LATTE metadata to be up to date with the latest TESS information.
        """
        indir = self.latte_dir
        if os.path.exists(indir) and os.path.isdir(indir):
            shutil.rmtree(indir, ignore_errors=True)
        os.makedirs(indir)
        with open("{}/_config.txt".format(indir), 'w') as f:
            f.write(str(indir))
        logging.info("Download the text files required ... ")
        logging.info("Only the manifest text files (~325 M) will be downloaded and no TESS data.")
        logging.info("This step may take a while but luckily it only has to run once... \n")
        if not os.path.exists("{}".format(indir)):
            os.makedirs(indir)
        if not os.path.exists("{}/data".format(indir)):
            os.makedirs("{}/data".format(indir))
        outF = open(indir + "/data/temp.txt", "w")
        outF.write("#all LC file links")
        outF.close()
        outF = open(indir + "/data/temp_tp.txt", "w+")
        outF.write("#all LC file links")
        outF.close()
        LATTEutils.data_files(indir)
        LATTEutils.tp_files(indir)
        LATTEutils.TOI_TCE_files(indir)
        LATTEutils.momentum_dumps_info(indir)

    def __prepare(self, candidate_df):
        """
        Downloads and fills files to be used by LATTE analysis.
        @return: the latte used directory, an open df to be used by LATTE analysis and the tic selected
        """
        # check whether a path already exists
        indir = self.latte_dir
        # SAVE the new output path
        if not os.path.exists("{}/_config.txt".format(indir)):
            self.update()
        candidate_df['id'] = candidate_df['id'].str.replace("TIC ", "")
        TIC_wanted = list(set(candidate_df['id']))
        nlc = len(TIC_wanted)
        logging.info("nlc length: {}".format(nlc))
        logging.info('{}/manifest.csv'.format(str(indir)))
        if exists('{}/manifest.csv'.format(str(indir))):
            logging.info("Existing manifest file found, will skip previously processed LCs and append to end of manifest file")
        else:
            logging.info("Creating new manifest file")
            metadata_header = ['id', 'Marked Transits', 'Sectors', 'RA', 'DEC', 'Solar Rad', 'TMag', 'Teff',
                               'thissector', 'TOI', 'TCE', 'TCE link', 'EB', 'Systematics', 'Background Flux',
                               'Centroids Positions', 'Momentum Dumps', 'Aperture Size', 'In/out Flux', 'Keep',
                               'Comment', 'starttime']
            with open('{}/manifest.csv'.format(str(indir)), 'w') as f:  # save in the photometry folder
                writer = csv.writer(f, delimiter=',')
                writer.writerow(metadata_header)
        return indir, candidate_df, TIC_wanted, candidate_df.iloc[0]["ffi"]

    def __process(self, indir, tic, sectors_in, t0, period, duration, depth, rp_rstar, a_rstar, ffi, run, curve):
        """
        Performs the LATTE analysis to generate PNGs and also the TPFPlotter analysis to get the field of view
        information.
        @param indir: the vetting source and resources directory
        @param tic: the tic to be processed
        @param sectors_in: the sectors to be used for the given tic
        @param t0: the candidate signal first epoch
        @param period: the candidate signal period
        @param duration: the candidate signal duration
        @param depth: the candidate signal depth in ppts
        @param ffi: Whether the candidate came from FFI data
        @param run: The run where the selected light curve appeared
        @param curve: The selected light curve in the given run
        @return: the given tic
        """
        logging.info("Running TESS Point")
        sectors_all, ra, dec = LATTEutils.tess_point(indir, tic)
        try:
            sectors = list(set(sectors_in) & set(sectors_all))
            if len(sectors) == 0:
                logging.info("The target was not observed in the sector(s) you stated ({}). \
                        Therefore take all sectors that it was observed in: {}".format(sectors, sectors_all))
                sectors = sectors_all
        except:
            sectors = sectors_all
        logging.info("Downloading LATTE data")
        sectors = np.sort(sectors)
        if not ffi:
            alltime, allflux, allflux_err, all_md, alltimebinned, allfluxbinned, allx1, allx2, ally1, ally2, alltime12, allfbkg, start_sec, end_sec, in_sec, tessmag, teff, srad = LATTEutils.download_data(
                indir, sectors, tic)
        else:
            alltime_list, allflux, allflux_small, allflux_flat, all_md, allfbkg, allfbkg_t, start_sec, end_sec, in_sec, X1_list, X4_list, apmask_list, arrshape_list, tpf_filt_list, t_list, bkg_list, tpf_list = LATTEutils.download_data_FFI(indir, sectors, syspath, sectors_all, tic, True)
            srad = "-"
            tessmag = "-"
            teff = "-"
            alltime = alltime_list
        simple = False
        BLS = False
        model = False
        save = True
        DV = True
        lc_file = "/" + str(run) + "/lc_" + str(curve) + ".csv"
        lc_file = self.object_dir + lc_file
        lc = pd.read_csv(lc_file, header=0)
        time, flux, flux_err = lc["#time"].values, lc["flux"].values, lc["flux_err"].values
        lc_len = len(time)
        zeros_lc = np.zeros(lc_len)
        logging.info("Preparing folded light curves for target")
        lc = TessLightCurve(time=time, flux=flux, flux_err=flux_err, quality=zeros_lc)
        lc.extra_columns = []
        self.plot_folded_curve(self.data_dir, "TIC " + tic, lc, period, t0, duration, depth / 1000, rp_rstar, a_rstar)
        transit_list = []
        last_time = alltime[len(alltime) - 1]
        num_of_transits = int(ceil(((last_time - t0) / period)))
        transit_lists = t0 + period * range(0, num_of_transits)
        time_as_array = np.array(alltime)
        transits_in_data = [time_as_array[(transit > time_as_array - 0.5) & (transit < time_as_array + 0.5)] for transit in transit_lists]
        transit_lists = transit_lists[[len(transits_in_data_set) > 0 for transits_in_data_set in transits_in_data]]
        transit_lists = [transit_lists[x:x + 3] for x in range(0, len(transit_lists), 3)]
        for index, transit_list in enumerate(transit_lists):
            transit_results_dir = self.data_dir + "/" + str(index)
            if not os.path.exists(transit_results_dir):
                os.mkdir(transit_results_dir)
            Vetter.plot_single_transits(transit_results_dir, "TIC " + str(tic), lc, transit_list, duration, depth / 1000, period, rp_rstar, a_rstar)
            logging.info("Brewing LATTE data for transits at T0s: %s", str(transit_list))
            try:
                if not ffi:
                    LATTEbrew.brew_LATTE(tic, indir, syspath, transit_list, simple, BLS, model, save, DV, sectors,
                                         sectors_all,
                                         alltime, allflux, allflux_err, all_md, alltimebinned, allfluxbinned, allx1, allx2,
                                         ally1, ally2, alltime12, allfbkg, start_sec, end_sec, in_sec, tessmag, teff, srad, ra,
                                         dec, self.args)
                else:
                    LATTEbrew.brew_LATTE_FFI(tic, indir, syspath, transit_list, simple, BLS, model, save, DV, sectors,
                                             sectors_all, alltime, allflux_flat, allflux_small, allflux, all_md, allfbkg,
                                             allfbkg_t, start_sec, end_sec, in_sec, X1_list, X4_list, apmask_list,
                                             arrshape_list, tpf_filt_list, t_list, bkg_list, tpf_list, ra, dec, self.args)
                # LATTE_DV.LATTE_DV(tic, indir, syspath, transit_list, sectors_all, simple, BLS, model, save, DV, sectors,
                #                      sectors_all,
                #                      alltime, allflux, allflux_err, all_md, alltimebinned, allfluxbinned, allx1, allx2,
                #                      ally1, ally2, alltime12, allfbkg, start_sec, end_sec, in_sec, tessmag, teff, srad, ra,
                #                      dec, self.args)
                tp_downloaded = True
                copy_tree(vetter.latte_dir + "/" + tic, transit_results_dir)
                shutil.rmtree(vetter.latte_dir + "/" + tic, ignore_errors=True)
            except Exception as e:
                traceback.print_exc()
                # see if it made any plots - often it just fails on the TPs as they are very large
                if exists("{}/{}/{}_fullLC_md.png".format(indir, tic, tic)):
                    logging.warning("couldn't download TP but continue anyway")
                    tp_downloaded = False
                    shutil.move(vetter.latte_dir + "/" + tic, transit_results_dir)
                else:
                    continue
        # check again whether the TPs downloaded - depending on where the code failed it might still have worked.
        if exists("{}/{}/{}_aperture_size.png".format(indir, tic, tic)):
            tp_downloaded = True
        else:
            tp_downloaded = False
            logging.warn("code ran but no TP -- continue anyway")
        # -------------
        # check whether it's a TCE or a TOI

        # TCE -----
        lc_dv = np.genfromtxt('{}/data/tesscurl_sector_all_dv.sh'.format(indir), dtype=str)
        TCE_links = []
        for i in lc_dv:
            if str(tic) in str(i[6]):
                TCE_links.append(i[6])
        if len(TCE_links) == 0:
            TCE = " - "
            TCE = False
        else:
            TCE_links = np.sort(TCE_links)
            TCE_link = TCE_links[0]  # this link should allow you to acess the MAST DV report
            TCE = True
        # TOI -----TOI_planets = pd.read_csv('{}/data/TOI_list.txt'.format(indir), comment="#")
        # TOIpl = TOI_planets.loc[TOI_planets['TIC'] == float(tic)]
        #TOI = False
        # TODO check why TOI is useful
        # else:
        #     TOI = True
        #     TOI_name = (float(TOIpl["Full TOI ID"]))
        # -------------
        # return the tic so that it can be stored in the manifest to keep track of which files have already been produced
        # and to be able to skip the ones that have already been processed if the code has to be restarted.
        mnd = {}
        mnd['id'] = tic
        mnd['MarkedTransits'] = transit_list
        mnd['Sectors'] = sectors_all
        mnd['RA'] = ra
        mnd['DEC'] = dec
        mnd['SolarRad'] = srad
        mnd['TMag'] = tessmag
        mnd['Teff'] = teff
        mnd['thissector'] = sectors
        # make empty fields for the test to be checked
        mnd['TOI'] = " "
        if TCE == True:
            mnd['TCE'] = "Yes"
            mnd['TCE_link'] = TCE_link
        else:
            mnd['TCE'] = " "
            mnd['TCE_link'] = " "
        mnd['EB'] = " "
        mnd['Systematics'] = " "
        mnd['TransitShape'] = " "
        mnd['BackgroundFlux'] = " "
        mnd['CentroidsPositions'] = " "
        mnd['MomentumDumps'] = " "
        mnd['ApertureSize'] = " "
        mnd['InoutFlux'] = " "
        mnd['Keep'] = " "
        mnd['Comment'] = " "
        mnd['starttime'] = np.nanmin(alltime) if not isinstance(alltime, str) else "-"
        return mnd

    @staticmethod
    def plot_single_transits(file_dir, id, lc, transit_times, duration, depth, period, rp_rstar, a_rstar):
        """
        Plots the phase-folded curve of the candidate for period, 2 * period and period / 2.
        @param file_dir: the directory to store the plot
        @param id: the target id
        @param lc: the input light curve with the data
        @param transit_times: the single transits T0s
        @param duration: the transit duration
        @param depth: the transit depth
        """
        duration = duration / 60 / 24
        figsize = (8, 8)  # x,y
        rows = len(transit_times)
        cols = 1
        fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        for index in np.arange(0, len(transit_times)):
            current_axs = axs[index] if isinstance(axs, np.ndarray) or isinstance(axs, list) else axs
            Vetter.plot_single_transit(id, current_axs, lc, transit_times[index], depth, duration, period, rp_rstar, a_rstar)
        plt.savefig(file_dir + "/single_transits.png", dpi=200)
        fig.clf()
        plt.close(fig)

    @staticmethod
    def plot_single_transit(id, axs, lc, transit_time, depth, duration, period, rp_rstar, a_rstar):
        sort_args = np.argsort(lc.time.value)
        time = lc.time.value[sort_args]
        flux = lc.flux.value[sort_args]
        flux_err = lc.flux_err.value[sort_args]
        folded_plot_range = duration * 2
        folded_phase_zoom_mask = np.where((time > transit_time - folded_plot_range) &
                                          (time < transit_time + folded_plot_range))
        folded_phase = time[folded_phase_zoom_mask]
        folded_y = flux[folded_phase_zoom_mask]
        if len(folded_phase) > 0:
            in_transit = (folded_phase > transit_time - duration / 2) & (folded_phase < transit_time + duration / 2)
            in_transit_center = (np.abs(folded_phase - transit_time)).argmin()
            model_time, model_flux = Vetter.get_transit_model(duration, transit_time,
                                                  (transit_time - folded_plot_range, transit_time + folded_plot_range),
                                                  depth, period, rp_rstar, a_rstar)
            axs.plot(model_time, model_flux, color="red")
            axs.scatter(folded_phase[~in_transit], folded_y[~in_transit], color="gray")
            axs.scatter(folded_phase[in_transit], folded_y[in_transit], color="darkorange")
            axs.set_xlim([transit_time - folded_plot_range, transit_time + folded_plot_range])
            axs.set_title(str(id) + " Single Transit at T={:.2f}d".format(transit_time))
            #axs.set_ylim([1 - 3 * depth, 1 + 3 * depth])
            axs.set_xlim([transit_time - folded_plot_range, transit_time + folded_plot_range])
            axs.set_xlabel("Time (d)")
            axs.set_ylabel("Flux norm.")
            logging.info("Processed single transit plot for T0=%.2f", transit_time)
        else:
            logging.info("Not plotting single transit for T0=%.2f as the data is empty", transit_time)

    @staticmethod
    def plot_folded_curve(file_dir, id, lc, period, epoch, duration, depth, rp_rstar, a_rstar):
        """
        Plots the phase-folded curve of the candidate for period, 2 * period and period / 2.
        @param file_dir: the directory to store the plot
        @param id: the target id
        @param period: the transit period
        @param epoch: the transit epoch
        @param duration: the transit duration
        @param depth: the transit depth
        """
        duration = duration / 60 / 24
        figsize = (16, 16)  # x,y
        rows = 3
        cols = 2
        fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        logging.info("Preparing folded light curves for target")
        #TODO bins = None for FFI
        bins = 100
        Vetter.compute_phased_values_and_fill_plot(id, axs[0][0], lc, period, epoch + period / 2, depth, duration,
                                                   rp_rstar, a_rstar, bins=bins)
        Vetter.compute_phased_values_and_fill_plot(id, axs[0][1], lc, period, epoch, depth, duration, rp_rstar, a_rstar, bins=bins)
        period = 2 * period
        Vetter.compute_phased_values_and_fill_plot(id, axs[1][0], lc, period, epoch + period / 2, depth, duration,
                                                   rp_rstar, a_rstar, bins=bins)
        Vetter.compute_phased_values_and_fill_plot(id, axs[1][1], lc, period, epoch, depth, duration, rp_rstar, a_rstar, bins=bins)
        period = period / 4
        Vetter.compute_phased_values_and_fill_plot(id, axs[2][0], lc, period, epoch + period / 2, depth, duration,
                                                   rp_rstar, a_rstar, bins=bins)
        Vetter.compute_phased_values_and_fill_plot(id, axs[2][1], lc, period, epoch, depth, duration, rp_rstar, a_rstar, bins=bins)
        plt.savefig(file_dir + "/odd_even_folded_curves.png", dpi=200)
        fig.clf()
        plt.close(fig)

    @staticmethod
    def compute_phased_values_and_fill_plot(id, axs, lc, period, epoch, depth, duration, rp_rstar, a_rstar, range=5, bins=None):
        """
        Phase-folds the input light curve and plots it centered in the given epoch
        @param id: the candidate name
        @param axs: the plot axis to be drawn
        @param lc: the lightkurve object containing the data
        @param period: the period for the phase-folding
        @param epoch: the epoch to center the fold
        @param depth: the transit depth
        @param duration: the transit duration
        @param range: the range to be used from the midtransit time in half-duration units.
        @param bins: the number of bins
        @return: the drawn axis and the computed bins
        """
        time = foldedleastsquares.core.fold(lc.time.value, period, epoch)
        axs.scatter(time, lc.flux.value, 2, color="blue", alpha=0.1)
        sort_args = np.argsort(time)
        time = time[sort_args]
        flux = lc.flux.value[sort_args]
        flux_err = lc.flux_err.value[sort_args]
        half_duration_phase = duration / 2 / period
        folded_plot_range = half_duration_phase * range
        folded_plot_range = folded_plot_range if folded_plot_range < 0.5 else 0.5
        folded_phase_zoom_mask = np.where((time > 0.5 - folded_plot_range) &
                                          (time < 0.5 + folded_plot_range))
        folded_phase = time[folded_phase_zoom_mask]
        folded_y = flux[folded_phase_zoom_mask]
        folded_y_err = flux_err[folded_phase_zoom_mask]
        axs.set_xlim([0.5 - folded_plot_range, 0.5 + folded_plot_range])
        # TODO if FFI no binning
        if bins is not None:
            bin_means, bin_edges, binnumber = stats.binned_statistic(folded_phase, folded_y,
                                                                     statistic='mean', bins=bins)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            bin_stds, _, _ = stats.binned_statistic(folded_phase, folded_y, statistic='std', bins=bins)
            axs.errorbar(bin_centers, bin_means, yerr=bin_stds / 2, xerr=bin_width / 2, marker='o', markersize=4,
                         color='darkorange', alpha=1, linestyle='none')
        else:
            bin_centers = folded_phase
            bin_means = folded_y
            bin_stds = folded_y_err * 2
        model_time, model_flux = Vetter.get_transit_model(half_duration_phase * 2, 0.5,
                                                          (0.5 - half_duration_phase * range, 0.5 + half_duration_phase * range),
                                                          depth, period, rp_rstar, a_rstar, 2 * len(time))
        axs.plot(model_time, model_flux, color="red")
        axs.set_title(str(id) + " Folded Curve with P={:.2f}d and T0={:.2f}".format(period, epoch))
        axs.set_xlabel("Time (d)")
        axs.set_ylabel("Flux norm.")
        #axs.set_ylim([1 - 3 * depth, 1 + 3 * depth])
        logging.info("Processed phase-folded plot for P=%.2f and T0=%.2f", period, epoch)
        return axs, bin_centers, bin_means, bin_stds / 2

    @staticmethod
    #TODO build model from selected transit_template
    def get_transit_model(duration, t0, start_end, depth, period, rp_to_rstar, a_to_rstar, model_len=10000):
        t = np.linspace(-6, 6, model_len)
        ma = batman.TransitParams()
        ma.t0 = 0  # time of inferior conjunction
        ma.per = 365  # orbital period, use Earth as a reference
        ma.rp = rp_to_rstar  # planet radius (in units of stellar radii)
        ma.a = a_to_rstar  # semi-major axis (in units of stellar radii)
        ma.inc = 90  # orbital inclination (in degrees)
        ma.ecc = 0  # eccentricity
        ma.w = 0  # longitude of periastron (in degrees)
        ma.u = [0.4804, 0.1867]  # limb darkening coefficients
        ma.limb_dark = "quadratic"  # limb darkening model
        m = batman.TransitModel(ma, t)  # initializes model
        model = m.light_curve(ma)  # calculates light curve
        model_intransit = np.argwhere(model < 1)[:, 0]
        model_time = np.linspace(start_end[0], start_end[1], len(model))
        in_transit_indexes = np.where((model_time > t0 - duration / 2) & (model_time < t0 + duration / 2))[0]
        model_time_in_transit = model_time[in_transit_indexes]
        scaled_intransit = np.interp(
            np.linspace(model_time_in_transit[0], model_time_in_transit[-1], len(in_transit_indexes)),
            model_time[model_intransit], model[model_intransit])
        model = np.full((model_len), 1.0)
        model[in_transit_indexes] = scaled_intransit
        model[model < 1] = 1 - ((1 - model[model < 1]) * depth / (1 - np.min(model)))
        return model_time, model

    def vetting(self, candidate, star, cpus):
        """
        Performs the LATTE vetting procedures
        @param candidate: the candidate dataframe containing id, period, t0, transits and sectors data.
        @param cpus: the number of cpus to be used. This parameter is of no use yet.
        """
        indir, df, TIC_wanted, ffi = self.__prepare(candidate)
        for tic in TIC_wanted:
            # check the existing manifest to see if I've processed this file!
            manifest_table = pd.read_csv('{}/manifest.csv'.format(str(indir)))
            # get the transit time list
            period = df.loc[df['id'] == tic]['period'].iloc[0]
            t0 = df.loc[df['id'] == tic]['t0'].iloc[0]
            rp_rstar = df.loc[df['id'] == tic]['rp_rs'].iloc[0]
            a_rstar = df.loc[df['id'] == tic]['a'].iloc[0] / star["R_star"]
            duration = df.loc[df['id'] == tic]['duration'].iloc[0]
            depth = df.loc[df['id'] == tic]['depth'].iloc[0]
            ffi = df.loc[df['id'] == tic]['ffi'].iloc[0]
            run = int(candidate.loc[candidate['id'] == tic]['number'].iloc[0])
            curve = int(candidate.loc[candidate['id'] == tic]['curve'].iloc[0])
            logging.info("------------------")
            logging.info("Candidate info")
            logging.info("------------------")
            logging.info("Period (d): %.2f", period)
            logging.info("Epoch (d): %.2f", t0)
            logging.info("Duration (min): %.2f", duration)
            logging.info("Depth (ppt): %.2f", depth)
            logging.info("FFI: %s", ffi)
            logging.info("Run: %.0f", run)
            logging.info("Detrend curve: %.0f", curve)
            try:
                sectors_in = ast.literal_eval(str(((df.loc[df['id'] == tic]['sectors']).values)[0]))
                if (type(sectors_in) == int) or (type(sectors_in) == float):
                    sectors = [sectors_in]
                else:
                    sectors = list(sectors_in)
            except:
                sectors = [0]
            index = 0
            vetting_dir = self.data_dir + "/vetting_" + str(index)
            while os.path.exists(vetting_dir) or os.path.isdir(vetting_dir):
                vetting_dir = self.data_dir + "/vetting_" + str(index)
                index = index + 1
            os.mkdir(vetting_dir)
            self.data_dir = vetting_dir
            ra = None
            dec = None
            try:
                res = self.__process(indir, tic, sectors, t0, period, duration, depth, rp_rstar, a_rstar, ffi, run, curve)
                ra = res['RA']
                dec = res['DEC']
                if res['id'] == -99:
                    logging.error('something went wrong with the LATTE results')
            except Exception as e:
                traceback.print_exc()
                try:
                    sectors_all, ra, dec = LATTEutils.tess_point(indir, tic)
                except Exception as e1:
                    traceback.print_exc()

    @staticmethod
    def plot_tpf(tpf, sector, aperture, dir):
        logging.info("Plotting FOV curves for sector %.0f", sector)
        if not os.path.exists(dir):
            os.mkdir(dir)
        tpf.plot_pixels(aperture_mask=aperture)
        plt.savefig(dir + "/Flux_pixels[" + str(sector) + "].png")
        plt.close()

    @staticmethod
    def compute_pixels_curves(tpf):
        masks = np.zeros(
            (tpf.shape[1] * tpf.shape[2], tpf.shape[1], tpf.shape[2]),
            dtype="bool",
        )
        for i in range(tpf.shape[1] * tpf.shape[2]):
            masks[i][np.unravel_index(i, (tpf.shape[1], tpf.shape[2]))] = True
        pixel_list = []
        for j in range(tpf.shape[1] * tpf.shape[2]):
            lc = tpf.to_lightcurve(aperture_mask=masks[j])
            lc = lc.remove_outliers(sigma_upper=3, sigma_lower=float('inf'))
            if len(lc.remove_nans().flux) == 0:
                pixel_list.append(None)
            else:
                pixel_list.append(lc)

    @staticmethod
    def vetting_field_of_view(indir, mission, tic, cadence, ra, dec, sectors, source, apertures):
        """
        Runs TPFPlotter to get field of view data.
        @param indir: the data source directory
        @param mission: the mission of the target
        @param tic: the target id
        @param cadence: the exposure time between measurements in seconds
        @param ra: the right ascension of the target
        @param dec: the declination of the target
        @param sectors: the sectors where the target was observed
        @param source: the source where the aperture was generated [tpf, tesscut, eleanor]
        @param apertures: a dict mapping sectors to boolean apertures
        @return: the directory where resulting data is stored
        """
        try:
            maglim = 6
            sectors_search = None if sectors is not None and len(sectors) == 0 else sectors
            logging.info("Preparing target pixel files for field of view plots")
            ra_str = str(ra)
            dec_str = "+" + str(dec) if dec >= 0 else str(dec)
            coords_str = ra_str + " " + dec_str
            if mission != "TESS":
                return
            target_title = "TIC " + str(tic)
            #TODO use retrieval method depending on source parameter
            if cadence > 120:
                tpf_source = lightkurve.search_tesscut(target_title, sector=sectors_search)
                if tpf_source is None or len(tpf_source) == 0:
                    tpf_source = lightkurve.search_tesscut(coords_str, sector=sectors_search)
                    target_title = "RA={:.4f},DEC={:.4f}".format(ra, dec)
            else:
                tpf_source = lightkurve.search_targetpixelfile(target_title, sector=sectors_search, author="SPOC",
                                                               cadence=cadence)
            save_dir = indir
            for i in range(0, len(tpf_source)):
                tpf = tpf_source[i].download(cutout_size=(CUTOUT_SIZE, CUTOUT_SIZE))
                row = tpf.row
                column = tpf.column
                plt.close()
                fig = plt.figure(figsize=(6.93, 5.5))
                gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 0.05, 0.01])
                gs.update(left=0.05, right=0.95, bottom=0.12, top=0.95, wspace=0.01, hspace=0.03)
                ax1 = plt.subplot(gs[0, 0])
                # TPF plot
                mean_tpf = np.mean(tpf.flux.value, axis=0)
                nx, ny = np.shape(mean_tpf)
                norm = ImageNormalize(stretch=stretching.LogStretch())
                division = np.int(np.log10(np.nanmax(tpf.flux.value)))
                splot = plt.imshow(np.nanmean(tpf.flux, axis=0) / 10 ** division, norm=norm, cmap="viridis",\
                                   extent=[column, column + ny, row, row + nx], origin='lower', zorder=0)
                aperture = apertures[tpf.sector]
                aperture = aperture if isinstance(aperture, np.ndarray) else np.array(aperture)
                aperture_boolean = ApertureExtractor.from_pixels_to_boolean_mask(aperture, column, row, CUTOUT_SIZE,
                                                                                 CUTOUT_SIZE)
                Vetter.plot_tpf(tpf, tpf.sector, aperture_boolean, indir)
                maskcolor = 'salmon'
                logging.info("    --> Using SHERLOCK aperture for sector %s...", tpf.sector)
                if aperture is not None:
                    for pixels in aperture:
                        ax1.add_patch(patches.Rectangle((pixels[0], pixels[1]),
                                                        1, 1, color=maskcolor, fill=True, alpha=0.4))
                        ax1.add_patch(patches.Rectangle((pixels[0], pixels[1]),
                                                        1, 1, color=maskcolor, fill=False, alpha=1, lw=2))
                # Gaia sources
                gaia_id, mag = tpfplotter.get_gaia_data(ra, dec)
                r, res = tpfplotter.add_gaia_figure_elements(tpf, magnitude_limit=mag + np.float(maglim), targ_mag=mag)
                x, y, gaiamags = r
                x, y, gaiamags = np.array(x) + 0.5, np.array(y) + 0.5, np.array(gaiamags)
                size = 128.0 / 2 ** ((gaiamags - mag))
                plt.scatter(x, y, s=size, c='red', alpha=0.6, edgecolor=None, zorder=10)
                # Gaia source for the target
                this = np.where(np.array(res['Source']) == int(gaia_id))[0]
                plt.scatter(x[this], y[this], marker='x', c='white', s=32, zorder=11)
                # Legend
                add = 0
                if np.int(maglim) % 2 != 0:
                    add = 1
                maxmag = np.int(maglim) + add
                legend_mags = np.linspace(-2, maxmag, np.int((maxmag + 2) / 2 + 1))
                fake_sizes = mag + legend_mags  # np.array([mag-2,mag,mag+2,mag+5, mag+8])
                for f in fake_sizes:
                    size = 128.0 / 2 ** ((f - mag))
                    plt.scatter(0, 0, s=size, c='red', alpha=0.6, edgecolor=None, zorder=10,
                                label=r'$\Delta m=$ ' + str(np.int(f - mag)))
                ax1.legend(fancybox=True, framealpha=0.7)
                # Source labels
                dist = np.sqrt((x - x[this]) ** 2 + (y - y[this]) ** 2)
                dsort = np.argsort(dist)
                for d, elem in enumerate(dsort):
                    if dist[elem] < 6:
                        plt.text(x[elem] + 0.1, y[elem] + 0.1, str(d + 1), color='white', zorder=100)
                # Orientation arrows
                tpfplotter.plot_orientation(tpf)
                # Labels and titles
                plt.xlim(column, column + ny)
                plt.ylim(row, row + nx)
                plt.xlabel('Pixel Column Number', fontsize=16)
                plt.ylabel('Pixel Row Number', fontsize=16)
                plt.title('Coordinates ' + target_title + ' - Sector ' + str(tpf.sector),
                          fontsize=16)  # + ' - Camera '+str(tpf.camera))  #
                # Colorbar
                cbax = plt.subplot(gs[0, 1])  # Place it where it should be.
                pos1 = cbax.get_position()  # get the original position
                pos2 = [pos1.x0 - 0.05, pos1.y0, pos1.width, pos1.height]
                cbax.set_position(pos2)  # set a new position
                cb = Colorbar(ax=cbax, cmap="viridis", mappable=splot, orientation='vertical', ticklocation='right')
                plt.xticks(fontsize=14)
                exponent = r'$\times 10^' + str(division) + '$'
                cb.set_label(r'Flux ' + exponent + r' (e$^-$)', labelpad=10, fontsize=16)
                save_dir = indir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(save_dir + '/TPF_Gaia_' + target_title + '_S' + str(tpf.sector) + '.pdf')
                # Save Gaia sources info
                dist = np.sqrt((x - x[this]) ** 2 + (y - y[this]) ** 2)
                GaiaID = np.array(res['Source'])
                srt = np.argsort(dist)
                x, y, gaiamags, dist, GaiaID = x[srt], y[srt], gaiamags[srt], dist[srt], GaiaID[srt]
                IDs = np.arange(len(x)) + 1
                inside = np.zeros(len(x))
                for pixels in aperture:
                    xtpf, ytpf = pixels[0], pixels[1]
                    _inside = np.where((x > xtpf) & (x < xtpf + 1) &
                                       (y > ytpf) & (y < ytpf + 1))[0]
                    inside[_inside] = 1
                data = Table([IDs, GaiaID, x, y, dist, dist * 21., gaiamags, inside.astype('int')],
                             names=['# ID', 'GaiaID', 'x', 'y', 'Dist_pix', 'Dist_arcsec', 'Gmag', 'InAper'])
                ascii.write(data, save_dir + '/Gaia_' + target_title + '_S' + str(tpf.sector) + '.dat', overwrite=True)
            return save_dir
        except SystemExit:
            logging.exception("Field Of View generation tried to exit.")
        except Exception as e:
            logging.exception("Exception found when generating Field Of View plots")


if __name__ == '__main__':
    ap = ArgumentParser(description='Vetting of Sherlock objects of interest')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--candidate', type=int, default=None, help="The candidate signal to be used.", required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    ap.add_argument('--cpus', type=int, default=None, help="The number of CPU cores to be used.", required=False)
    ap.add_argument('--no_validate', dest='validate', action='store_false',
                    help="Whether to avoid running statistical validation")
    args = ap.parse_args()
    vetter = Vetter(args.object_dir, args.validate)
    file_dir = vetter.object_dir + "/vetting.log"
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
    logging.info("Starting vetting")
    star_df = pd.read_csv(vetter.object_dir + "/params_star.csv")
    if args.candidate is None:
        user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
        candidate = pd.DataFrame(columns=['id', 'period', 'depth', 't0', 'sectors', 'ffi', 'number', 'lc'])
        candidate = candidate.append(user_properties, ignore_index=True)
        candidate['id'] = star_df.iloc[0]["obj_id"]
    else:
        candidate_selection = int(args.candidate)
        candidates = pd.read_csv(vetter.object_dir + "/candidates.csv")
        if candidate_selection < 1 or candidate_selection > len(candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        candidates = candidates.rename(columns={'Object Id': 'id'})
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate['number'] = [candidate_selection]
        vetter.data_dir = vetter.object_dir
        logging.info("Selected signal number " + str(candidate_selection))
    if args.cpus is None:
        cpus = multiprocessing.cpu_count() - 1
    else:
        cpus = args.cpus
    vetter.vetting(candidate, star_df.iloc[0], cpus)
