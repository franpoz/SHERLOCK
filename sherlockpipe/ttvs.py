import io
import logging
import os
import re
import shutil
import sys
from argparse import ArgumentParser

import alexfitter
import pandas as pd
from numpy import arange
from contextlib import redirect_stdout


def get_from_user(target, key):
    value = None
    if isinstance(target, dict) and key in target:
        value = target[key]
    return value


class TtvFitter:
    def __init__(self, only_initial, ttvs_error, fit_dir, ttvs_dir):
        self.only_initial = only_initial
        self.fit_dir = fit_dir
        self.ttvs_dir = ttvs_dir
        self.ttvs_error = ttvs_error

    def fit(self):
        shutil.copy(self.fit_dir + "/lc.csv", self.ttvs_dir + "/lc.csv")
        shutil.copy(self.fit_dir + "/params.csv", self.ttvs_dir + "/params.csv")
        shutil.copy(self.fit_dir + "/params_star.csv", self.ttvs_dir + "/params_star.csv")
        shutil.copy(self.fit_dir + "/settings.csv", self.ttvs_dir + "/settings.csv")
        self._tune_settings()
        self._fix_time_params()
        self._prepare_ttv_params()
        alexfitter.show_initial_guess(ttvs_dir)
        if not args.only_initial:
            logging.info("Running dynamic nested sampling")
            alexfitter.ns_fit(ttvs_dir)
            alexfitter.ns_output(ttvs_dir)

    def _tune_settings(self):
        logging.info("Enabling TTVs fitting setting")
        with open(ttvs_dir + "/settings.csv", 'r+') as f:
            text = f.read()
            text = re.sub('fit_ttvs,False', 'fit_ttvs,True', text)
            f.seek(0)
            f.write(text)
            f.truncate()

    def _fix_time_params(self):
        logging.info("Fixing epochs and periods")
        params_df = pd.read_csv(ttvs_dir + "/params.csv")
        ns_table_df = pd.read_csv(object_dir + "/results/ns_table.csv")
        candidates_count = len(ns_table_df[ns_table_df["#name"].str.contains("_period")])
        for i in arange(0, candidates_count):
            period_row = ns_table_df[ns_table_df["#name"].str.contains("_period")].iloc[i]
            period_row_name = period_row["#name"]
            period = period_row["median"]
            epoch_row = ns_table_df[ns_table_df["#name"].str.contains("_epoch")].iloc[i]
            epoch_row_name = epoch_row["#name"]
            epoch = epoch_row["median"]
            params_df.loc[params_df["#name"] == period_row_name, "value"] = period
            params_df.loc[params_df["#name"] == period_row_name, "fit"] = 0
            params_df.loc[params_df["#name"] == epoch_row_name, "value"] = epoch
            params_df.loc[params_df["#name"] == epoch_row_name, "fit"] = 0
        params_df.to_csv(ttvs_dir + "/params.csv", index=False)

    def _prepare_ttv_params(self):
        logging.info("Preparing TTVs params")
        alexfitter.prepare_ttv_fit(self.ttvs_dir)
        with open(self.ttvs_dir + '/ttv_preparation/ttv_initial_guess_params.csv') as f:
            ttvs_params = f.readlines()
        ttvs_error_str = str(self.ttvs_error)
        ttvs_error_str = ",uniform -" + ttvs_error_str + " " + ttvs_error_str + ","
        ttvs_initial_str = ",0.0,1,uniform"
        ttvs_params = [re.sub(r",uniform -?\d+(\.\d+)? -?\d+(\.\d+)?,", ttvs_error_str, ttvs_param) for ttvs_param in ttvs_params]
        ttvs_params = [re.sub(r",-?\d+(\.\d+)?,1,uniform", ttvs_initial_str, ttvs_param) for ttvs_param in ttvs_params]
        with open(self.ttvs_dir + "/params.csv", "a") as params_file:
            params_file.write("\n" + "".join(ttvs_params))


if __name__ == '__main__':
    ap = ArgumentParser(description='Calculation of Time Transits Variations')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--cpus', type=int, default=4, help="The number of CPU cores to be used.", required=False)
    ap.add_argument('--only_initial', dest='only_initial', action='store_true',
                    help="Whether to only run an initial guess of the ttvs")
    ap.add_argument('--ttvs_error', type=float, default=0.07, required=False,
                    help="Error window for each single transit fit")
    args = ap.parse_args()
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    index = 0
    ttvs_dir = object_dir + "/ttvs"
    if os.path.exists(ttvs_dir) or os.path.isdir(ttvs_dir):
        shutil.rmtree(ttvs_dir, ignore_errors=True)
    os.mkdir(ttvs_dir)
    file_dir = ttvs_dir + "/ttvs.log"
    if os.path.exists(file_dir):
        os.remove(file_dir)
    if not isinstance(logging.root, logging.RootLogger):
        logging.root = logging.RootLogger(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
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
    logging.info("Starting TTVs computation")
    TtvFitter(args.only_initial, args.ttvs_error, object_dir, ttvs_dir).fit()
