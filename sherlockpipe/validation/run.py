import logging
import os
import pandas as pd
import sys

from sherlockpipe.loading import common
from sherlockpipe.validation.validator import Validator


def run_validate(args):
    index = 0
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    candidates = pd.read_csv(object_dir + "/candidates.csv")
    validation_dir = object_dir + "/validation_" + str(index)
    while os.path.exists(validation_dir) or os.path.isdir(validation_dir):
        validation_dir = object_dir + "/validation_" + str(index)
        index = index + 1
    os.mkdir(validation_dir)
    file_dir = validation_dir + "/validation.log"
    if os.path.exists(file_dir):
        os.remove(file_dir)
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
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate['number'] = [candidate_selection]
        logging.info("Selected signal number " + str(candidate_selection))
    validator = Validator(object_dir, validation_dir, len(candidate) == 1, candidates)
    validator.validate(candidate, star_df.iloc[0], args.cpus, args.contrast_curve, args.bins, args.scenarios,
                       args.sigma_mode)
