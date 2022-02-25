# from __future__ import print_function, absolute_import, division
import logging
import multiprocessing
from pathlib import Path
import traceback

import PIL
import batman
import foldedleastsquares
import lcbuilder
import lightkurve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wotan
import yaml
from lcbuilder.constants import CUTOUT_SIZE
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.photometry.aperture_extractor import ApertureExtractor
from lightkurve import TessLightCurve, TessTargetPixelFile, KeplerTargetPixelFile
from matplotlib.colorbar import Colorbar
from matplotlib import patches
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table
from astropy.io import ascii
import astropy.visualization as stretching
from argparse import ArgumentParser
from scipy import stats, ndimage
from sherlockpipe import tpfplotter, constants
import six
import sys
sys.modules['astropy.extern.six'] = six
matplotlib.use('Agg')
import pandas as pd
import os
from math import ceil


class Vetter:
    """
    Provides transiting candidate vetting information like centroids and spaceship motion, momentum dumps, neighbours
    curves inspection and more to give a deeper insight on the quality of the candidate signal.
    """
    def __init__(self, object_dir):
        self.object_dir = os.getcwd() if object_dir is None else object_dir
        self.latte_dir = str(Path.home()) + "/.sherlockpipe/latte/"
        if not os.path.exists(self.latte_dir):
            os.mkdir(self.latte_dir)
        self.data_dir = self.object_dir

    def __process(self, indir, id, sectors_in, t0, period, duration, depth, rp_rstar, a_rstar, ffi, run, curve, cpus):
        """
        Performs the analysis to generate PNGs and Data Validation Report.
        @param indir: the vetting source and resources directory
        @param id: the tic to be processed
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
        logging.info("Running Transit Plots")
        lc_file = "/lc.csv"
        lc_file = self.object_dir + lc_file
        lc_data_file = self.object_dir + "/lc_data.csv"
        lc = pd.read_csv(lc_file, header=0)
        lc_data = pd.read_csv(lc_data_file, header=0)
        lc_data = Vetter.normalize_lc_data(lc_data)
        logging.info("Reading apertures from directory")
        apertures = yaml.load(open(self.object_dir + "/apertures.yaml"), yaml.SafeLoader)
        apertures = apertures["sectors"]
        time, flux, flux_err = lc["#time"].values, lc["flux"].values, lc["flux_err"].values
        lc_len = len(time)
        zeros_lc = np.zeros(lc_len)
        logging.info("Preparing folded light curves for target")
        lc = TessLightCurve(time=time, flux=flux, flux_err=flux_err, quality=zeros_lc)
        lc.extra_columns = []
        tpfs_dir = self.object_dir + "/tpfs"
        tpfs = []
        mission, mission_prefix, mission_int_id = LcBuilder().parse_object_info(id)
        for tpf_file in os.listdir(tpfs_dir):
            tpf = TessTargetPixelFile(tpfs_dir + "/" + tpf_file) if mission == lcbuilder.constants.MISSION_TESS else \
                KeplerTargetPixelFile(tpfs_dir + "/" + tpf_file)
            tpfs.append(tpf)
        self.plot_folded_curve(self.data_dir, id, lc, period, t0, duration, depth / 1000, rp_rstar, a_rstar)
        last_time = time[len(time) - 1]
        num_of_transits = int(ceil(((last_time - t0) / period)))
        transit_lists = t0 + period * range(0, num_of_transits)
        time_as_array = np.array(time)
        transits_in_data = [time_as_array[(transit > time_as_array - 0.5) & (transit < time_as_array + 0.5)] for transit in transit_lists]
        transit_lists = transit_lists[[len(transits_in_data_set) > 0 for transits_in_data_set in transits_in_data]]
        plot_transits_inputs = []
        for index, transit_times in enumerate(transit_lists):
            plot_transits_inputs.append((self.data_dir, str(id), lc, lc_data, tpfs, apertures, transit_times,
                                       depth / 1000, duration, period, rp_rstar, a_rstar))
        with multiprocessing.Pool(processes=cpus) as pool:
            pool.map(Vetter.plot_single_transit, plot_transits_inputs)

    @staticmethod
    def normalize_lc_data(lc_data):
        logging.info("Normalizing lc_data")
        time = lc_data["time"].to_numpy()
        dif = time[1:] - time[:-1]
        jumps = np.where(np.abs(dif) > 1)[0]
        jumps = np.append(jumps, len(lc_data))
        previous_jump_index = 0
        for jumpIndex in jumps:
            token = lc_data["centroids_x"][previous_jump_index:jumpIndex]
            lc_data["centroids_x"][previous_jump_index:jumpIndex] = token - np.nanmedian(token)
            token = lc_data["centroids_y"][previous_jump_index:jumpIndex]
            lc_data["centroids_y"][previous_jump_index:jumpIndex] = token - np.nanmedian(token)
            token = lc_data["motion_x"][previous_jump_index:jumpIndex]
            lc_data["motion_x"][previous_jump_index:jumpIndex] = token - np.nanmedian(token)
            token = lc_data["motion_y"][previous_jump_index:jumpIndex]
            lc_data["motion_y"][previous_jump_index:jumpIndex] = token - np.nanmedian(token)
            previous_jump_index = jumpIndex
        return lc_data

    @staticmethod
    def plot_single_transit(file_dir, id, lc, lc_data, tpfs, apertures, transit_time, depth, duration, period, rp_rstar, a_rstar):
        """
        Plots the phase-folded curve of the candidate for period, 2 * period and period / 2.
        @param file_dir: the directory to store the plot
        @param id: the target id
        @param lc: the input light curve with the data
        @param lc_data: the input light curve with the data and motion, centroids and bck flux data
        @param transit_time: the single transit T0
        @param depth: the transit depth
        @param duration: the transit duration
        @param period: the transit duration
        @param rp_rstar: radius of the planet / radius of the star
        @param a_rstar: semimajor-axis / radius of the star
        """
        duration = duration / 60 / 24
        fig = plt.figure(figsize=(24, 6), constrained_layout=True)
        fig.suptitle('Vetting of ' + str(id) + ' single transit at T=' + str(round(transit_time, 2)) + 'd', size=26)
        gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.1)  # 2 rows, 3 columns
        ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
        ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
        ax3 = fig.add_subplot(gs[0, 2])  # First row, third column
        ax4 = fig.add_subplot(gs[1, 0])  # First row, third column
        ax5 = fig.add_subplot(gs[1, 1])  # First row, third column
        ax6 = fig.add_subplot(gs[1, 2])  # First row, third column
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        sort_args = np.argsort(lc.time.value)
        time = lc.time.value[sort_args]
        flux = lc.flux.value[sort_args]
        flux_err = lc.flux_err.value[sort_args]
        plot_range = duration * 2
        zoom_mask = np.where((time > transit_time - plot_range) & (time < transit_time + plot_range))
        plot_time = time[zoom_mask]
        plot_flux = flux[zoom_mask]
        zoom_lc_data = lc_data[(lc_data["time"] > transit_time - plot_range) & (lc_data["time"] < transit_time + plot_range)]
        aperture_mask = None
        chosen_aperture_lc = None
        smaller_aperture_lc = None
        eroded_aperture_mask = None
        tpf_short_framed = None
        if len(plot_time) > 0:
            for tpf in tpfs:
                tpf_zoom_mask = tpf[(tpf.time.value > transit_time - plot_range) & (tpf.time.value < transit_time + plot_range)]
                if len(tpf_zoom_mask) > 0:
                    if hasattr(tpf, 'sector') and tpf.sector is not None:
                        sector = tpf.sector
                    elif hasattr(tpf, 'campaign') and tpf.campaign:
                        sector = tpf.campaign
                    else:
                        sector = tpf.quarter
                    tpf_short_framed = tpf[(tpf.time.value > transit_time - plot_range) &
                                           (tpf.time.value < transit_time + plot_range)]
                    if len(tpf_short_framed) == 0:
                        break
                    aperture_mask = ApertureExtractor.from_pixels_to_boolean_mask(apertures[sector], tpf.column,
                                                                                  tpf.row, tpf.shape[1], tpf.shape[2])
                    eroded_aperture_mask = ndimage.binary_erosion(aperture_mask)
                    chosen_aperture_lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
                    smaller_aperture_lc = tpf.to_lightcurve(aperture_mask=eroded_aperture_mask)
                    break
            single_transit_file = file_dir + "/single_transit_" + str(transit_time) + ".png"
            tpf_single_transit_file = file_dir + "/tpf_single_transit_" + str(transit_time) + ".png"
            t1 = transit_time - duration / 2
            t4 = transit_time + duration / 2
            model_time, model_flux = Vetter.get_transit_model(duration, transit_time,
                                                              (transit_time - plot_range,
                                                               transit_time + plot_range),
                                                              depth, period, rp_rstar, a_rstar)
            momentum_dumps_lc_data = None
            if not zoom_lc_data["quality"].isnull().all():
                momentum_dumps_lc_data = zoom_lc_data[np.bitwise_and(zoom_lc_data["quality"].to_numpy(),
                                                                     constants.MOMENTUM_DUMP_QUALITY_FLAG) >= 1]

            axs[0].plot(model_time, model_flux, color="red")
            axs[0].scatter(plot_time, plot_flux, color="darkorange", label="Photometry used aperture")
            axs[0].set_xlim([transit_time - plot_range, transit_time + plot_range])
            axs[0].set_title("Single Transit")
            if momentum_dumps_lc_data is not None and len(momentum_dumps_lc_data) > 1:
                first_momentum_dump = True
                for index, momentum_dump_row in momentum_dumps_lc_data.iterrows():
                    if first_momentum_dump:
                        axs[0].axvline(x=momentum_dump_row["time"], color='purple', label='Momentum dump')
                        first_momentum_dump = False
                    else:
                        axs[0].axvline(x=momentum_dump_row["time"], color='purple')
            axs[0].legend(loc='upper left')
            axs[0].set_xlim([transit_time - plot_range, transit_time + plot_range])
            axs[0].set_xlabel("Time (d)")
            axs[0].set_ylabel("Flux norm.")
            axs[1].scatter(zoom_lc_data["time"], zoom_lc_data["motion_x"], color="red",
                           label="X-axis motion")
            axs[1].scatter(zoom_lc_data["time"], zoom_lc_data["centroids_x"],
                           color="black", label="X-axis centroids")
            axs[1].axvline(x=transit_time - duration / 2, color='r', label='T1')
            axs[1].axvline(x=transit_time + duration / 2, color='r', label='T4')
            axs[1].legend(loc='upper left')
            axs[1].set_xlim([transit_time - plot_range, transit_time + plot_range])
            axs[1].set_title("X-axis drift")
            axs[1].set_xlabel("Time (d)")
            axs[1].set_ylabel("Normalized Y-axis data")
            axs[2].scatter(zoom_lc_data["time"], zoom_lc_data["motion_y"], color="red",
                           label="Y-axis motion")
            axs[2].scatter(zoom_lc_data["time"], zoom_lc_data["centroids_y"],
                           color="black", label="Y-axis centroids")
            axs[2].axvline(x=t1, color='r', label='T1')
            axs[2].axvline(x=t4, color='r', label='T4')
            axs[2].legend(loc='upper left')
            axs[2].set_xlim([transit_time - plot_range, transit_time + plot_range])
            axs[2].set_title("Y-axis drift")
            axs[2].set_xlabel("Time (d)")
            axs[2].set_ylabel("Normalized Y-axis data")
            if eroded_aperture_mask is not None and True in eroded_aperture_mask and smaller_aperture_lc is not None:
                axs[3].plot(model_time, model_flux, color="red")
                axs[3].set_xlim([transit_time - plot_range, transit_time + plot_range])
                axs[3].set_title("SAP comparison")
                axs[3].set_xlim([transit_time - plot_range, transit_time + plot_range])
                axs[3].set_xlabel("Time (d)")
                axs[3].set_ylabel("Flux norm.")
                chosen_aperture_lc.flux = wotan.flatten(chosen_aperture_lc.time.value,
                                                    chosen_aperture_lc.flux.value, window_length=0.75,
                                                    return_trend=False, method="biweight", break_tolerance=0.5)
                smaller_aperture_lc.flux = wotan.flatten(smaller_aperture_lc.time.value,
                                                    smaller_aperture_lc.flux.value, window_length=0.75,
                                                    return_trend=False, method="biweight", break_tolerance=0.5)
                chosen_aperture_lc = chosen_aperture_lc[
                    (chosen_aperture_lc.time.value - plot_range < transit_time) &
                    (chosen_aperture_lc.time.value + plot_range > transit_time)]
                smaller_aperture_lc = smaller_aperture_lc[
                    (smaller_aperture_lc.time.value - plot_range < transit_time) &
                    (smaller_aperture_lc.time.value + plot_range > transit_time)]
                axs[3].scatter(chosen_aperture_lc.time.value, chosen_aperture_lc.flux.value, color="darkorange",
                               label="Photometry used aperture")
                axs[3].scatter(smaller_aperture_lc.time.value, smaller_aperture_lc.flux.value,
                               color="c", label="Photometry smaller aperture")
            axs[3].legend(loc='upper left')
            axs[4] = tpf_short_framed.plot(axs[4], aperture_mask=aperture_mask)
            axs[4].set_title("TPF apertures comparison")
            if eroded_aperture_mask is not None and True in eroded_aperture_mask and smaller_aperture_lc is not None:
                parsed_aperture = tpf_short_framed._parse_aperture_mask(eroded_aperture_mask)
                for i in range(tpf_short_framed.shape[1]):
                    for j in range(tpf_short_framed.shape[2]):
                        if parsed_aperture[i, j]:
                            rect = patches.Rectangle(
                                xy=(j + tpf_short_framed.column - 0.5, i + tpf_short_framed.row - 0.5),
                                width=1,
                                height=1,
                                color='black',
                                fill=False,
                                hatch="\\\\",
                            )
                            axs[4].add_patch(rect)
            axs[5].scatter(zoom_lc_data["time"], zoom_lc_data["background_flux"],
                           color="blue", label="Background Flux")
            axs[5].axvline(x=t1, color='r', label='T1')
            axs[5].axvline(x=t4, color='r', label='T4')
            axs[5].legend(loc='upper left')
            axs[5].set_xlim([transit_time - plot_range, transit_time + plot_range])
            axs[5].set_title("Background flux")
            axs[5].set_xlabel("Time (d)")
            axs[5].set_ylabel("Background flux (e/s)")
            plt.savefig(single_transit_file, dpi=100, bbox_inches='tight')
            plt.clf()
            plt.close()
            tpf_short_framed.plot_pixels(aperture_mask=aperture_mask)
            plt.savefig(tpf_single_transit_file, dpi=100)
            plt.clf()
            plt.close()
            images_list = [single_transit_file, tpf_single_transit_file]
            imgs = [PIL.Image.open(i) for i in images_list]
            imgs[0] = imgs[0].resize((imgs[1].size[0],
                                      int(imgs[1].size[0] / imgs[0].size[0] * imgs[0].size[1])),
                                      PIL.Image.ANTIALIAS)
            img_merge = np.vstack((np.asarray(i) for i in imgs))
            img_merge = PIL.Image.fromarray(img_merge)
            img_merge.save(file_dir + "/single_transit_" + str(transit_time) + ".png", quality=95, optimize=True)
            os.remove(tpf_single_transit_file)
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


    def vetting(self, candidate_df, star, cpus):
        """
        Performs the LATTE vetting procedures
        @param candidate: the candidate dataframe containing id, period, t0, transits and sectors data.
        @param cpus: the number of cpus to be used. This parameter is of no use yet.
        """
        df = candidate_df.iloc[0]
        # TODO get the transit time list
        id = df['id']
        period = df['period']
        t0 = df['t0']
        rp_rstar = df['rp_rs']
        a_rstar = df['a'] / star["R_star"] * constants.AU_TO_RSUN
        duration = df['duration']
        depth = df['depth']
        ffi = df['ffi']
        run = int(df['number'])
        curve = int(df['curve'])
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
            if (type(df["sectors"]) == int) or (type(df["sectors"]) == float):
                sectors = [df["sectors"]]
            else:
                sectors = [int(sector) for sector in df["sectors"].split(',')]
        except:
            sectors = [0]
        index = 0
        vetting_dir = self.data_dir + "/vetting_" + str(index)
        while os.path.exists(vetting_dir) or os.path.isdir(vetting_dir):
            vetting_dir = self.data_dir + "/vetting_" + str(index)
            index = index + 1
        os.mkdir(vetting_dir)
        self.data_dir = vetting_dir
        try:
            self.__process(vetting_dir, id, sectors, t0, period, duration, depth, rp_rstar, a_rstar, ffi, run, curve, cpus)
        except Exception as e:
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
    args = ap.parse_args()
    vetter = Vetter(args.object_dir)
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
