import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import sys

from sherlockpipe.loading import common
from sherlockpipe.validation.validator import Validator


def run_validate(args):
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    candidates = pd.read_csv(object_dir + "/candidates.csv")
    if args.candidate is not None:
        validation_dir = object_dir + "/validate_" + str(args.candidate)
    else:
        validation_dir = object_dir + "/validate_" + str(Path(args.properties).stem)
    if os.path.exists(validation_dir) or os.path.isdir(validation_dir):
        shutil.rmtree(validation_dir, ignore_errors=True)
    os.mkdir(validation_dir)
    file_dir = validation_dir + "/validation.log"
    if os.path.exists(file_dir):
        os.remove(file_dir)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if not isinstance(logging.root, logging.RootLogger):
        logging.root = logging.RootLogger(logging.INFO)
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
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    if args.candidate is None:
        logging.info("Reading validation input from properties file: %s", args.properties)
        user_properties = common.load_from_yaml(args.properties)
        candidate = pd.DataFrame(columns=['id', 'period', 'depth', 't0', 'sectors', 'ffi', 'number', 'lc'])
        candidate = candidate.append(user_properties, ignore_index=True)
        candidate['id'] = star_df.iloc[0]["obj_id"]
    else:
        candidate_selection = int(args.candidate)
        if candidate_selection < 1 or candidate_selection > len(candidates.index):
            raise SystemExit("User selected a wrong candidate number.")
        candidates = candidates.rename(columns={'Object Id': 'id'})
        candidates['number'] = 1
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate.iloc[:]['number'] = candidate_selection
        logging.info("Selected signal number " + str(candidate_selection))
    if args.sectors is not None:
        candidate['sectors'] = args.sectors
    validator = Validator(object_dir, validation_dir, len(candidate) == 1, candidates)
    validator.validate(candidate, star_df.iloc[0], args.cpus, args.contrast_curve, args.bins, args.scenarios,
                       args.sigma_mode)
