import logging
import os
import sys
from argparse import ArgumentParser

import json
import yaml
import pandas as pd
from sherlockpipe.nbodies.megno import MegnoStabilityCalculator
from sherlockpipe.nbodies.planet_input import PlanetInput
from sherlockpipe.nbodies.spock import SpockStabilityCalculator


def get_from_user(target, key):
    value = None
    if isinstance(target, dict) and key in target:
        value = target[key]
    return value


if __name__ == '__main__':
    ap = ArgumentParser(description='Validation of system stability')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--planets', type=str, default=None,
                    help="Comma separated string of assumed existing planets.",
                    required=False)
    ap.add_argument('--max_ecc', type=float, default=0.1,
                    help="Upper limit for the eccentricity grid.",
                    required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    ap.add_argument('--cpus', type=int, default=4, help="The number of CPU cores to be used.", required=False)
    ap.add_argument('--ecc_bins', type=int, default=None, help="The number of eccentricity bins to use.", required=False)
    ap.add_argument('--mass_bins', type=int, default=None, help="The number of mass bins to use.", required=False)
    ap.add_argument('--star_mass_bins', type=int, default=3, help="The number of star mass bins to use.",
                    required=False)
    ap.add_argument('--megno', dest='use_megno', action='store_true',
                    help="Whether to force the usage of megno even for multiplanetary systems.")
    args = ap.parse_args()
    planets = args.planets.split(",") if args.planets is not None else None
    object_dir = os.getcwd() if args.object_dir is None else args.object_dir
    if planets is None and args.properties is None:
        print("Can't execute validation without assumed signals or properties file")
    index = 0
    stability_dir = object_dir + "/stability_" + str(index)
    while os.path.exists(stability_dir) or os.path.isdir(stability_dir):
        stability_dir = object_dir + "/stability_" + str(index)
        index = index + 1
    os.mkdir(stability_dir)
    file_dir = stability_dir + "/stability.log"
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
    logging.info("Starting stability validation")
    star_df = pd.read_csv(object_dir + "/params_star.csv")
    star_mass = star_df.iloc[0]["M_star"]
    star_mass_low_err = star_df.iloc[0]["M_star_lerr"]
    star_mass_up_err = star_df.iloc[0]["M_star_uerr"]
    candidates = pd.read_csv(object_dir + "/candidates.csv")
    planets_params = []
    if planets is not None:
        planets = map(int, planets)
        planets = [planets - 1 for planets in planets]
        planets_df = candidates.iloc[planets]
        planets_params = [PlanetInput(planet["period"], planet["rad_p"], eccentricity_low=0,
                                      eccentricity_up=args.max_ecc)
                                 for index, planet in planets_df.iterrows()]
    star_mass_low = star_mass if star_mass_low_err is None else star_mass - star_mass_low_err
    star_mass_up = star_mass if star_mass_up_err is None else star_mass + star_mass_up_err
    star_mass_bins = args.star_mass_bins
    if args.properties is not None:
        user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
        user_planet_params = []
        for planet in user_properties["BODIES"]:
            ecc_bins = args.ecc_bins if "E_BINS" not in planet or args.ecc_bins is not None else planet["E_BINS"]
            mass_bins = args.mass_bins if "M_BINS" not in planet or args.mass_bins is not None else planet["M_BINS"]
            user_planet_params.append(PlanetInput(period=get_from_user(planet, "P"),
                                                  radius=get_from_user(planet, "R"),
                                                  mass_low=get_from_user(planet, "M_LOW"),
                                                  mass_max=get_from_user(planet, "M_UP"),
                                                  eccentricity_low=get_from_user(planet, "E_LOW"),
                                                  eccentricity_up=get_from_user(planet, "E_UP"),
                                                  mass_bins=mass_bins,
                                                  ecc_bins=ecc_bins))
        planets_params = planets_params + user_planet_params
        if "STAR" in user_properties:
            star_mass_low = star_mass_low if "M_LOW" not in user_properties["STAR"] else user_properties["STAR"]["M_LOW"]
            star_mass_up = star_mass_up if "M_UP" not in user_properties["STAR"] else user_properties["STAR"]["M_UP"]
            star_mass_bins = args.star_mass_bins if "M_BINS" not in user_properties["STAR"] \
                                                    or args.star_mass_bins is not None \
                                                 else user_properties["STAR"]["M_BINS"]
    stability_calculator = MegnoStabilityCalculator() if len(planets_params) < 3 or args.use_megno \
        else SpockStabilityCalculator()
    logger.info("%.0f planets to be simulated. %s will be used", len(planets_params),
                type(stability_calculator).__name__)
    logger.info("Lowest star mass: %.2f", star_mass_low)
    logger.info("Highest star mass: %.2f", star_mass_up)
    logger.info("Star mass bins: %.0f", star_mass_bins)
    for key, body in enumerate(planets_params):
        logger.info("Body %.0f: %s", key, json.dumps(body.__dict__))
    stability_calculator.run(stability_dir, star_mass_low, star_mass_up, star_mass_bins, planets_params, args.cpus)

