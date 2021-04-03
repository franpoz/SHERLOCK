# from __future__ import print_function, absolute_import, division
import copy
import multiprocessing
import shutil
import types
from multiprocessing import Pool
from pathlib import Path
import traceback
import lightkurve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from astropy.time import Time
from lightkurve import TessLightCurve
from matplotlib.colorbar import Colorbar
from matplotlib import patches
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table
from astropy.io import ascii
import astropy.visualization as stretching
from argparse import ArgumentParser
import astropy.units as u
from sherlockpipe.eleanor import TargetData
from sherlockpipe import constants as const
from sherlockpipe import tpfplotter, eleanor
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
import triceratops.triceratops as tr
from matplotlib import cm, ticker
from math import floor, ceil
from triceratops.likelihoods import (simulate_TP_transit, simulate_EB_transit)
from triceratops.funcs import (Gauss2D, query_TRILEGAL, renorm_flux, stellar_relations)
from astropy import constants
import uuid

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

class Vetter:
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
        indir = self.latte_dir
        if os.path.exists(indir) and os.path.isdir(indir):
            shutil.rmtree(indir, ignore_errors=True)
        os.makedirs(indir)
        with open("{}/_config.txt".format(indir), 'w') as f:
            f.write(str(indir))
        print("\n Download the text files required ... ")
        print("\n Only the manifest text files (~325 M) will be downloaded and no TESS data.")
        print("\n This step may take a while but luckily it only has to run once... \n")
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
        candidate_df['TICID'] = candidate_df['TICID'].str.replace("TIC ", "")
        TIC_wanted = list(set(candidate_df['TICID']))
        nlc = len(TIC_wanted)
        print("nlc length: {}".format(nlc))
        print('{}/manifest.csv'.format(str(indir)))
        if exists('{}/manifest.csv'.format(str(indir))):
            print("Existing manifest file found, will skip previously processed LCs and append to end of manifest file")
        else:
            print("Creating new manifest file")
            metadata_header = ['TICID', 'Marked Transits', 'Sectors', 'RA', 'DEC', 'Solar Rad', 'TMag', 'Teff',
                               'thissector', 'TOI', 'TCE', 'TCE link', 'EB', 'Systematics', 'Background Flux',
                               'Centroids Positions', 'Momentum Dumps', 'Aperture Size', 'In/out Flux', 'Keep',
                               'Comment', 'starttime']
            with open('{}/manifest.csv'.format(str(indir)), 'w') as f:  # save in the photometry folder
                writer = csv.writer(f, delimiter=',')
                writer.writerow(metadata_header)
        return indir, candidate_df, TIC_wanted, candidate_df.iloc[0]["ffi"]

    def __process(self, indir, tic, sectors_in, transit_list, t0, period, ffi):
        """
        Performs the LATTE analysis to generate PNGs and also the TPFPlotter analysis to get the field of view
        information.
        @param indir: the vetting source and resources directory
        @param tic: the tic to be processed
        @param sectors_in: the sectors to be used for the given tic
        @param transit_list: the list of transits for the given tic
        @return: the given tic
        """
        sectors_all, ra, dec = LATTEutils.tess_point(indir, tic)
        try:
            sectors = list(set(sectors_in) & set(sectors_all))
            if len(sectors) == 0:
                print("The target was not observed in the sector(s) you stated ({}). \
                        Therefore take all sectors that it was observed in: {}".format(sectors, sectors_all))
                sectors = sectors_all
        except:
            sectors = sectors_all
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
        # TODO decide whether to use transit_list or period
        transit_list = []
        last_time = alltime[len(alltime) - 1]
        num_of_transits = int(ceil(((last_time - t0) / period)))
        transit_lists = t0 + period * range(0, num_of_transits)
        transit_lists = [transit_lists[x:x + 3] for x in range(0, len(transit_lists), 3)]
        for index, transit_list in enumerate(transit_lists):
            transit_results_dir = self.data_dir + "/" + str(index)
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
                shutil.move(vetter.latte_dir + "/" + tic, transit_results_dir)
            except Exception as e:
                traceback.print_exc()
                # see if it made any plots - often it just fails on the TPs as they are very large
                if exists("{}/{}/{}_fullLC_md.png".format(indir, tic, tic)):
                    print("couldn't download TP but continue anyway")
                    tp_downloaded = False
                    shutil.move(vetter.latte_dir + "/" + tic, transit_results_dir)
                else:
                    continue
        # check again whether the TPs downloaded - depending on where the code failed it might still have worked.
        if exists("{}/{}/{}_aperture_size.png".format(indir, tic, tic)):
            tp_downloaded = True
        else:
            tp_downloaded = False
            print("code ran but no TP -- continue anyway")
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
        # TOI -----
        TOI_planets = pd.read_csv('{}/data/TOI_list.txt'.format(indir), comment="#")
        TOIpl = TOI_planets.loc[TOI_planets['TIC'] == float(tic)]
        TOI = False
        # TODO check why TOI is useful
        # else:
        #     TOI = True
        #     TOI_name = (float(TOIpl["Full TOI ID"]))
        # -------------
        # return the tic so that it can be stored in the manifest to keep track of which files have already been produced
        # and to be able to skip the ones that have already been processed if the code has to be restarted.
        mnd = {}
        mnd['TICID'] = tic
        mnd['MarkedTransits'] = transit_list
        mnd['Sectors'] = sectors_all
        mnd['RA'] = ra
        mnd['DEC'] = dec
        mnd['SolarRad'] = srad
        mnd['TMag'] = tessmag
        mnd['Teff'] = teff
        mnd['thissector'] = sectors
        # make empty fields for the test to be checked
        if TOI == True:
            mnd['TOI'] = TOI_name
        else:
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

    def vetting(self, candidate, cpus):
        indir, df, TIC_wanted, ffi = self.__prepare(candidate)
        for tic in TIC_wanted:
            # check the existing manifest to see if I've processed this file!
            manifest_table = pd.read_csv('{}/manifest.csv'.format(str(indir)))
            # get a list of the current URLs that exist in the manifest
            urls_exist = manifest_table['TICID']
            # get the transit time list
            period = df.loc[df['TICID'] == tic]['period'].iloc[0]
            duration = df.loc[df['TICID'] == tic]['duration'].iloc[0]
            t0 = df.loc[df['TICID'] == tic]['t0'].iloc[0]
            transit_list = ast.literal_eval(((df.loc[df['TICID'] == tic]['transits']).values)[0])
            transit_depth = df.loc[df['TICID'] == tic]['depth'].iloc[0]
            candidate_row = candidate.iloc[0]
            if candidate_row["number"] is None or np.isnan(candidate_row["number"]):
                lc_file = "/" + candidate_row["lc"]
            else:
                lc_file = "/" + str(candidate_row["number"]) + "/lc_" + str(candidate_row["curve"]) + ".csv"
            lc_file = self.data_dir + lc_file
            try:
                sectors_in = ast.literal_eval(str(((df.loc[df['TICID'] == tic]['sectors']).values)[0]))
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
            try:
                res = self.__process(indir, tic, sectors, transit_list, t0, period, ffi)
                ra = res['RA']
                dec = res['DEC']
                if res['TICID'] == -99:
                    print('something went wrong with the LATTE results')
            except Exception as e:
                traceback.print_exc()
            if self.validate:
                result_dir, ra, dec = self.vetting_validation(cpus, indir, tic, sectors, lc_file, transit_depth, period,
                                                              t0, duration)
                shutil.move(result_dir, vetting_dir + "/triceratops")
            result_dir = self.vetting_field_of_view(indir, tic, ra, dec, sectors)
            shutil.move(result_dir, vetting_dir + "/tpfplot")
            # TODO improve this condition to check whether tic, sectors and transits exist
        #     if not np.isin(tic, urls_exist):
        #         # make sure the file is opened as append only
        #         with open('{}/manifest.csv'.format(str(indir)), 'a') as tic:  # save in the photometry folder
        #             writer = csv.writer(tic, delimiter=',')
        #             metadata_data = [res['TICID']]
        #             metadata_data.append(res['MarkedTransits'])
        #             metadata_data.append(res['Sectors'])
        #             metadata_data.append(res['RA'])
        #             metadata_data.append(res['DEC'])
        #             metadata_data.append(res['SolarRad'])
        #             metadata_data.append(res['TMag'])
        #             metadata_data.append(res['Teff'])
        #             metadata_data.append(res['thissector'])
        #             metadata_data.append(res['TOI'])
        #             metadata_data.append(res['TCE'])
        #             metadata_data.append(res['TCE_link'])
        #             metadata_data.append(res['EB'])
        #             metadata_data.append(res['Systematics'])
        #             metadata_data.append(res['BackgroundFlux'])
        #             metadata_data.append(res['CentroidsPositions'])
        #             metadata_data.append(res['MomentumDumps'])
        #             metadata_data.append(res['ApertureSize'])
        #             metadata_data.append(res['InoutFlux'])
        #             metadata_data.append(res['Keep'])
        #             metadata_data.append(res['Comment'])
        #             metadata_data.append(res['starttime'])
        #             writer.writerow(metadata_data)
        # return TIC_wanted

    def vetting_validation(self, cpus, indir, tic, sectors, lc_file, transit_depth, period, t0, transit_duration):
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
        @param tic: the tess object id for which the analysis will be run
        @param sectors: the sectors of the tic
        @param lc_file: the light curve source file
        @param transit_depth: the depth of the transit signal (ppts)
        @param period: the period of the transit signal /days)
        @param t0: the t0 of the transit signal (days)
        @param transit_duration: the duration of the transit signal (minutes)
        """
        save_dir = indir + "/" + str(tic) + "/triceratops_" + str(uuid.uuid4())
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sectors = np.array(sectors)
        duration = transit_duration / 60 / 24
        target = tr.target(ID=tic, sectors=sectors)
        # TODO allow user input apertures
        tpfs = lightkurve.search_targetpixelfile("TIC " + str(tic), mission="TESS", cadence="short", sector=sectors.tolist())\
            .download_all()
        star = eleanor.multi_sectors(tic=tic, sectors=sectors, tesscut_size=31, post_dir=const.USER_HOME_ELEANOR_CACHE)
        apertures = []
        sector_num = 0
        for s in star:
            tpf_idx = [data.sector if data.sector == s.sector else -1 for data in tpfs.data]
            tpf = tpfs[np.where(tpf_idx > np.zeros(len(tpf_idx)))[0][0]]
            pipeline_mask = tpfs[np.where(tpf_idx > np.zeros(len(tpf_idx)))[0][0]].pipeline_mask
            pipeline_mask = np.transpose(pipeline_mask)
            pipeline_mask_triceratops = np.zeros((len(pipeline_mask[0]), len(pipeline_mask[:][0]), 2))
            for i in range(0, len(pipeline_mask[0])):
                for j in range(0, len(pipeline_mask[:][0])):
                    pipeline_mask_triceratops[i, j] = [tpf.column + i, tpf.row + j]
            pipeline_mask_triceratops[~pipeline_mask] = None
            aperture = []
            for i in range(0, len(pipeline_mask_triceratops[0])):
                for j in range(0, len(pipeline_mask_triceratops[:][0])):
                    if not np.isnan(pipeline_mask_triceratops[i, j]).any():
                        aperture.append(pipeline_mask_triceratops[i, j])
            apertures.append(aperture)
            target.plot_field(save=True, fname=save_dir + "/field_S" + str(s.sector), sector=s.sector,
                            ap_pixels=aperture)
            sector_num = sector_num + 1
        apertures = np.array(apertures)
        depth = transit_depth / 1000
        target.calc_depths(depth, apertures)
        target.stars.to_csv(save_dir + "/stars.csv", index=False)
        lc = pd.read_csv(lc_file, header=0)
        time, flux, flux_err = lc["#time"].values, lc["flux"].values, lc["flux_err"].values
        lc_len = len(time)
        zeros_lc = np.zeros(lc_len)
        lc = TessLightCurve(time=time, flux=flux, flux_err=flux_err, quality=zeros_lc)
        lc.extra_columns = []
        lc = lc.fold(period=period, epoch_time=t0, normalize_phase=True)
        folded_plot_range = duration / 2 / period * 5
        inner_folded_range_args = np.where((0 - folded_plot_range < lc.time.value) & (lc.time.value < 0 + folded_plot_range))
        lc = lc[inner_folded_range_args]
        lc.time = lc.time * period
        sigma = np.mean(lc.flux_err)
        input_n_times = [ValidatorInput(save_dir, copy.deepcopy(target), lc.time.value, lc.flux.value, sigma, period, depth,
                                        apertures, value)
                         for value in range(0, self.validation_runs)]
        validator = Validator()
        with Pool(processes=cpus) as pool:
            validation_results = pool.map(validator.validate, input_n_times)
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
        return save_dir

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

    def vetting_field_of_view(self, indir, tic, ra, dec, sectors):
        maglim = 6
        sectors_search = None if sectors is not None and len(sectors) == 0 else sectors
        tpf_source = lightkurve.search_targetpixelfile("TIC " + str(tic), sector=sectors, mission='TESS')
        if tpf_source is None or len(tpf_source) == 0:
            ra_str = str(ra)
            dec_str = "+" + str(dec) if dec >= 0 else str(dec)
            tpf_source = lightkurve.search_tesscut(ra_str + " " + dec_str, sector=sectors_search)
        for i in range(0, len(tpf_source)):
            tpf = tpf_source[i].download(cutout_size=(12, 12))
            pipeline = True
            fig = plt.figure(figsize=(6.93, 5.5))
            gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 0.05, 0.01])
            gs.update(left=0.05, right=0.95, bottom=0.12, top=0.95, wspace=0.01, hspace=0.03)
            ax1 = plt.subplot(gs[0, 0])
            # TPF plot
            mean_tpf = np.mean(tpf.flux.value, axis=0)
            nx, ny = np.shape(mean_tpf)
            norm = ImageNormalize(stretch=stretching.LogStretch())
            division = np.int(np.log10(np.nanmax(tpf.flux.value)))
            splot = plt.imshow(np.nanmean(tpf.flux, axis=0) / 10 ** division, norm=norm, \
                               extent=[tpf.column, tpf.column + ny, tpf.row, tpf.row + nx], origin='lower', zorder=0)
            # Pipeline aperture
            if pipeline:  #
                aperture_mask = tpf.pipeline_mask
                aperture = tpf._parse_aperture_mask(aperture_mask)
                maskcolor = 'lightgray'
                print("    --> Using pipeline aperture...")
            else:
                aperture_mask = tpf.create_threshold_mask(threshold=10, reference_pixel='center')
                aperture = tpf._parse_aperture_mask(aperture_mask)
                maskcolor = 'lightgray'
                print("    --> Using threshold aperture...")

            for i in range(aperture.shape[0]):
                for j in range(aperture.shape[1]):
                    if aperture_mask[i, j]:
                        ax1.add_patch(patches.Rectangle((j + tpf.column, i + tpf.row),
                                                        1, 1, color=maskcolor, fill=True, alpha=0.4))
                        ax1.add_patch(patches.Rectangle((j + tpf.column, i + tpf.row),
                                                        1, 1, color=maskcolor, fill=False, alpha=1, lw=2))
            # Gaia sources
            gaia_id, mag = tpfplotter.get_gaia_data_from_tic(tic)
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
            plt.xlim(tpf.column, tpf.column + ny)
            plt.ylim(tpf.row, tpf.row + nx)
            plt.xlabel('Pixel Column Number', fontsize=16)
            plt.ylabel('Pixel Row Number', fontsize=16)
            plt.title('Coordinates ' + tic + ' - Sector ' + str(tpf.sector),
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
            save_dir = indir + "/tpfplot"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir + '/TPF_Gaia_TIC' + tic + '_S' + str(tpf.sector) + '.pdf')
            # Save Gaia sources info
            dist = np.sqrt((x - x[this]) ** 2 + (y - y[this]) ** 2)
            GaiaID = np.array(res['Source'])
            srt = np.argsort(dist)
            x, y, gaiamags, dist, GaiaID = x[srt], y[srt], gaiamags[srt], dist[srt], GaiaID[srt]
            IDs = np.arange(len(x)) + 1
            inside = np.zeros(len(x))
            for i in range(aperture.shape[0]):
                for j in range(aperture.shape[1]):
                    if aperture_mask[i, j]:
                        xtpf, ytpf = j + tpf.column, i + tpf.row
                        _inside = np.where((x > xtpf) & (x < xtpf + 1) &
                                           (y > ytpf) & (y < ytpf + 1))[0]
                        inside[_inside] = 1
            data = Table([IDs, GaiaID, x, y, dist, dist * 21., gaiamags, inside.astype('int')],
                         names=['# ID', 'GaiaID', 'x', 'y', 'Dist_pix', 'Dist_arcsec', 'Gmag', 'InAper'])
            ascii.write(data, save_dir + '/Gaia_TIC' + tic + '_S' + str(tpf.sector) + '.dat', overwrite=True)
        return save_dir

    def show_candidates(self):
        self.candidates = pd.read_csv(self.object_dir + "/candidates.csv")
        self.candidates.index = np.arange(1, len(self.candidates) + 1)
        print("Suggested candidates are:")
        print(self.candidates.to_markdown(index=True))
        pass

    def demand_candidate_selection(self):
        user_input = input("Select candidate number to be examined and fit: ")
        if user_input.startswith("q"):
            raise SystemExit("User quitted")
        self.candidate_selection = int(user_input)
        if self.candidate_selection < 1 or self.candidate_selection > len(self.candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        self.data_dir = self.object_dir + "/" + str(self.candidate_selection)

class Validator:
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
    ap.add_argument('--no_validate', dest='validate', action='store_false',
                    help="Whether to avoid running statistical validation")
    args = ap.parse_args()
    vetter = Vetter(args.object_dir, args.validate)
    if args.candidate is None:
        user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
        candidate = pd.DataFrame(columns=['id', 'transits', 'sectors', 'FFI'])
        candidate = candidate.append(user_properties, ignore_index=True)
        candidate = candidate.rename(columns={'id': 'TICID'})
        candidate['TICID'] = candidate["TICID"].apply(str)
        cpus = user_properties["settings"]["cpus"]
    else:
        candidate_selection = int(args.candidate)
        candidates = pd.read_csv(vetter.object_dir + "/candidates.csv")
        if candidate_selection < 1 or candidate_selection > len(candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        candidates = candidates.rename(columns={'Object Id': 'TICID'})
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate['number'] = [candidate_selection]
        vetter.data_dir = vetter.object_dir
        print("Selected signal number " + str(candidate_selection))
        if args.cpus is None:
            cpus = multiprocessing.cpu_count() - 1
        else:
            cpus = args.cpus
    vetter.vetting(candidate, cpus)
