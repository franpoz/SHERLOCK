from argparse import ArgumentParser
from sherlockpipe.bayesian_fit.run import run_fit


def fit_parse_args(args=None):
    ap = ArgumentParser(description='Fitting of Sherlock objects of interest')
    ap.add_argument('--object_dir',
                    help="If the object directory is not your current one you need to provide the ABSOLUTE path",
                    required=False)
    ap.add_argument('--candidate', type=str, default=None, help="The CSV candidate signals to be used.", required=False)
    ap.add_argument('--only_initial', dest='only_initial', action='store_true',
                    help="Whether to only run an initial guess of the transit")
    ap.set_defaults(only_initial=False)
    ap.add_argument('--cpus', type=int, default=None, help="The number of CPU cores to be used.", required=False)
    ap.add_argument('--tolerance', type=float, default=0.01, help="The tolerance of the nested sampling algorithm.",
                    required=False)
    ap.add_argument('--mcmc', dest='mcmc', action='store_true', help="Whether to run using mcmc or ns. Default is ns.")
    ap.add_argument('--detrend', dest='detrend', default="hybrid_spline", help="Type of detrending to be used",
                    required=False, choices=['no', 'gp'])
    ap.add_argument('--fit_orbit', dest='fit_orbit', action='store_true', help="Whether to fit eccentricity and "
                                                                               "argument of periastron")
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    return ap.parse_args(args)


if __name__ == '__main__':
    args = fit_parse_args()
    run_fit(args)
