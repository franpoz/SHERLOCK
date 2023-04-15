from argparse import ArgumentParser
from sherlockpipe.system_stability.run import run_stability


def stability_args_parse(args=None):
    ap = ArgumentParser(description='Validation of system stability')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--max_ecc', type=float, default=0.1,
                    help="Upper limit for the eccentricity grid.",
                    required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    ap.add_argument('--cpus', type=int, default=4, help="The number of CPU cores to be used.", required=False)
    ap.add_argument('--period_bins', type=int, default=1, help="The number of period bins to use.", required=False)
    ap.add_argument('--ecc_bins', type=int, default=1, help="The number of eccentricity bins to use.", required=False)
    ap.add_argument('--inc_bins', type=int, default=1, help="The number of inclination bins to use.", required=False)
    ap.add_argument('--omega_bins', type=int, default=1, help="The number of argument of periastron bins to use.",
                    required=False)
    ap.add_argument('--mass_bins', type=int, default=1, help="The number of mass bins to use.", required=False)
    ap.add_argument('--star_mass_bins', type=int, default=1, help="The number of star mass bins to use.",
                    required=False)
    ap.add_argument('--years', type=int, default=500, help="The number of years to integrate (for MEGNO).",
                    required=False)
    ap.add_argument('--free_params', type=str, default=None, help="The parameters to be entirely sampled, separated by "
                                                                  "commas. E.g. 'eccentricity,omega'", required=False)
    return ap.parse_args(args)


if __name__ == '__main__':
    args = stability_args_parse()
    run_stability(args)
