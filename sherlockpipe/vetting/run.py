import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from sherlockpipe.loading import common
from sherlockpipe.vetting.vetter import Vetter


def run_vet(object_dir, candidate, properties, cpus=os.cpu_count() - 1):
    object_dir = os.getcwd() if object_dir is None else object_dir
    candidates = pd.read_csv(object_dir + "/candidates.csv")
    if candidate is not None:
        vetting_dir = object_dir + "/vet_" + str(candidate)
    else:
        vetting_dir = object_dir + "/vet_" + str(Path(properties).stem)
    if os.path.exists(vetting_dir) or os.path.isdir(vetting_dir):
        shutil.rmtree(vetting_dir, ignore_errors=True)
    os.mkdir(vetting_dir)
    vetter = Vetter(object_dir, vetting_dir, candidate is not None, candidates)
    file_dir = vetter.watson.object_dir + "/vetting.log"
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
    logging.info("Starting vetting")
    star_df = pd.read_csv(vetter.object_dir() + "/params_star.csv")
    transits_df = None
    if candidate is None:
        user_properties = common.load_from_yaml(properties)
        candidate = pd.DataFrame(columns=['id', 'period', 'depth', 't0', 'sectors', 'number', 'lc'])
        candidate = candidate.append(user_properties, ignore_index=True)
        candidate['id'] = star_df.iloc[0]["obj_id"]
    else:
        candidate_selection = int(candidate)
        if candidate_selection < 1 or candidate_selection > len(candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        candidates = candidates.rename(columns={'Object Id': 'id'})
        candidates['number'] = 1
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate.iloc[:]['number'] = candidate_selection
        transits_df_file = vetter.object_dir() + "/transits_stats.csv"
        if os.path.exists(transits_df_file):
            transits_df = pd.read_csv(vetter.object_dir() + "/transits_stats.csv")
            transits_df = transits_df[transits_df["candidate"] == candidate_selection - 1]
            if len(transits_df) == 0:
                logging.info("Not NAN transits found for candidate in transits_stats.csv file")
                transits_df = None
        # watson.data_dir = watson.object_dir
        logging.info("Selected signal number " + str(candidate_selection))
    transits_mask = []
    for i in range(0, int(candidate['number']) - 1):
        transits_mask.append({"P": candidates.iloc[i]["period"], "T0": candidates.iloc[i]["t0"],
                              "D": candidates.iloc[i]["duration"] * 2})
    vetter.run(cpus, candidate=candidate, star_df=star_df.iloc[0], transits_df=transits_df, transits_mask=transits_mask)
