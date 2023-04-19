import logging
import math
import os
import sys
from argparse import ArgumentParser
import pandas as pd

from sherlockpipe.loading import common


def tidal_args_parse(args=None):
    ap = ArgumentParser(description='Validation of system stability')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--candidate', type=int, default=None, help="The candidate signal to be used.", required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    return ap.parse_args(args)


def run_tidal(object_dir, candidate, properties):
    object_dir = os.getcwd() if object_dir is None else object_dir
    candidates = pd.read_csv(object_dir + "/candidates.csv")
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
    logging.info("Starting tidal locking time computation")
    star_df = pd.read_csv(object_dir + "/params_star.csv")
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
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate['number'] = [candidate_selection]
        logging.info("Selected signal number " + str(candidate_selection))
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    star_mass = star_df.iloc[0]["M_star"]
    star_mass_low_err = star_df.iloc[0]["M_star_lerr"]
    star_mass_up_err = star_df.iloc[0]["M_star_uerr"]
    logger.info("Star mass: %.2f", star_mass)
    logger.info("Lowest star mass: %.2f", star_mass - star_mass_low_err)
    logger.info("Highest star mass: %.2f", star_mass + star_mass_up_err)
    G = 6.67259e-11  # gravitational constant
    Q = 100  # planet's tidal dissipation function
    semimajor_axis = candidate.iloc[0]['a']
    rp = candidate.iloc[0]["rad_p"]
    planet_mean_motion = math.sqrt((G * (star_mass + 1)) / (semimajor_axis ** 3)) #angular velocity of the planet in radians per second
    T_rot_planet = (2 * math.pi) / planet_mean_motion
    # "Tidal Evolution of Close-in Planets" by Matthew J. Holman and Scott J. Tremaine, The Astrophysical Journal, vol. 552, pp. 693–718 (2001)
    # This paper provides a detailed derivation of the tidal evolution equations, including the time scale for a planet to become tidally locked.
    T_lock = (3 * math.sqrt(T_rot_planet) * semimajor_axis ** 6) / (Q * G * star_mass * rp ** 5)
    print("Tidal Locking Time: {:.2e} years".format(T_lock))


if __name__ == '__main__':
    args = tidal_args_parse()
    run_tidal(args.object_dir, args.candidate, args.properties)
