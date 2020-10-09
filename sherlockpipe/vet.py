# from __future__ import print_function, absolute_import, division
import shutil
import types
from pathlib import Path
import traceback
import lightkurve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from matplotlib.colorbar import Colorbar
from matplotlib import patches
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table
from astropy.io import ascii
import astropy.visualization as stretching
from argparse import ArgumentParser
from sherlockpipe import tpfplotter
import six
import sys
sys.modules['astropy.extern.six'] = six

matplotlib.use('Agg')
import pandas as pd
import os
from os.path import exists
import ast
import csv
from sherlockpipe.LATTEsub import LATTEutils, LATTEbrew
from os import path

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
    def __init__(self, object_dir):
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
        num_of_transits = int((last_time - t0) / period)
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

    def vetting(self, candidate):
        indir, df, TIC_wanted, ffi = self.__prepare(candidate)
        for tic in TIC_wanted:
            # check the existing manifest to see if I've processed this file!
            manifest_table = pd.read_csv('{}/manifest.csv'.format(str(indir)))
            # get a list of the current URLs that exist in the manifest
            urls_exist = manifest_table['TICID']
            # get the transit time list
            period = df.loc[df['TICID'] == tic]['period'].iloc[0]
            t0 = df.loc[df['TICID'] == tic]['t0'].iloc[0]
            transit_list = ast.literal_eval(((df.loc[df['TICID'] == tic]['transits']).values)[0])
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
            res = self.__process(indir, tic, sectors, transit_list, t0, period, ffi)
            if res['TICID'] == -99:
                print('something went wrong')
                continue
            self.vetting_field_of_view(indir, tic, res['RA'], res['DEC'], sectors)
            shutil.move(vetter.latte_dir + "/tpfplot", vetting_dir + "/tpfplot")
            # TODO improve this condition to check whether tic, sectors and transits exist
            if not np.isin(tic, urls_exist):
                # make sure the file is opened as append only
                with open('{}/manifest.csv'.format(str(indir)), 'a') as tic:  # save in the photometry folder
                    writer = csv.writer(tic, delimiter=',')
                    metadata_data = [res['TICID']]
                    metadata_data.append(res['MarkedTransits'])
                    metadata_data.append(res['Sectors'])
                    metadata_data.append(res['RA'])
                    metadata_data.append(res['DEC'])
                    metadata_data.append(res['SolarRad'])
                    metadata_data.append(res['TMag'])
                    metadata_data.append(res['Teff'])
                    metadata_data.append(res['thissector'])
                    metadata_data.append(res['TOI'])
                    metadata_data.append(res['TCE'])
                    metadata_data.append(res['TCE_link'])
                    metadata_data.append(res['EB'])
                    metadata_data.append(res['Systematics'])
                    metadata_data.append(res['BackgroundFlux'])
                    metadata_data.append(res['CentroidsPositions'])
                    metadata_data.append(res['MomentumDumps'])
                    metadata_data.append(res['ApertureSize'])
                    metadata_data.append(res['InoutFlux'])
                    metadata_data.append(res['Keep'])
                    metadata_data.append(res['Comment'])
                    metadata_data.append(res['starttime'])
                    writer.writerow(metadata_data)
        return TIC_wanted

    def vetting_field_of_view(self, indir, tic, ra, dec, sectors):
        maglim = 6
        sectors_search = None if sectors is not None and len(sectors) == 0 else sectors
        tpf = lightkurve.search_tesscut(tic, sector=sectors_search).download(cutout_size=(12, 12))
        if tpf is None:
            ra_str = str(ra)
            dec_str = "+" + str(dec) if dec >= 0 else str(dec)
            tpf = lightkurve.search_tesscut(ra_str + " " + dec_str, sector=sectors_search).download(cutout_size=(12, 12))
        pipeline = "False"
        fig = plt.figure(figsize=(6.93, 5.5))
        gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 0.05, 0.01])
        gs.update(left=0.05, right=0.95, bottom=0.12, top=0.95, wspace=0.01, hspace=0.03)
        ax1 = plt.subplot(gs[0, 0])
        # TPF plot
        mean_tpf = np.mean(tpf.flux, axis=0)
        nx, ny = np.shape(mean_tpf)
        norm = ImageNormalize(stretch=stretching.LogStretch())
        division = np.int(np.log10(np.nanmax(tpf.flux)))
        splot = plt.imshow(np.nanmean(tpf.flux, axis=0) / 10 ** division, norm=norm, \
                           extent=[tpf.column, tpf.column + ny, tpf.row, tpf.row + nx], origin='bottom', zorder=0)
        # Pipeline aperture
        if pipeline == "True":  #
            aperture_mask = tpf.pipeline_mask
            aperture = tpf._parse_aperture_mask(aperture_mask)
            maskcolor = 'tomato'
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
        cb = Colorbar(ax=cbax, mappable=splot, orientation='vertical', ticklocation='right')
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


if __name__ == '__main__':
    ap = ArgumentParser(description='Vetting of Sherlock objects of interest')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the ABSOLUTE path", required=False)
    ap.add_argument('--candidate', type=int, default=None, help="The candidate signal to be used.", required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    args = ap.parse_args()
    vetter = Vetter(args.object_dir)
    if args.candidate is None:
        user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
        candidate = pd.DataFrame(columns=['id', 'transits', 'sectors', 'FFI'])
        candidate = candidate.append(user_properties, ignore_index=True)
        candidate = candidate.rename(columns={'id': 'TICID'})
        candidate['TICID'] = candidate["TICID"].apply(str)
    else:
        candidate_selection = int(args.candidate)
        candidates = pd.read_csv(vetter.object_dir + "/candidates.csv")
        if candidate_selection < 1 or candidate_selection > len(candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        candidates = candidates.rename(columns={'Object Id': 'TICID'})
        candidate = candidates.iloc[[candidate_selection - 1]]
        vetter.data_dir = vetter.object_dir
        print("Selected signal number " + str(candidate_selection))
        vetter.vetting(candidate)
