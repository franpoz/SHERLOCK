from argparse import ArgumentParser
from sherlockpipe.observation_plan.run import run_plan

if __name__ == '__main__':
    ap = ArgumentParser(description='Planning of observations from Sherlock objects of interest')
    ap.add_argument('--object_dir',
                    help="If the object directory is not your current one you need to provide the ABSOLUTE path",
                    required=False)
    ap.add_argument("--observatories", help="Csv file containing the observatories coordinates", required=False)
    ap.add_argument('--lat', help="Observer latitude", required=False)
    ap.add_argument('--lon', help="Observer longitude", required=False)
    ap.add_argument('--alt', help="Observer altitude", required=False)
    ap.add_argument('--tz', help="Time zone (in hours offset from UTC)", required=False)
    ap.add_argument('--min_altitude', help="Minimum altitude of the target to be observable", default=23,
                    required=False)
    ap.add_argument('--moon_min_dist', help="Minimum required moon distance for moon minimum illumination.", default=30,
                    required=False)
    ap.add_argument('--moon_max_dist', help="Minimum required moon distance for moon maximum illumination.", default=55,
                    required=False)
    ap.add_argument('--max_days', help="Maximum number of days for the plan to take.", type=int, default=365, required=False)
    ap.add_argument('--transit_fraction', help="Minimum transit fraction to be observable.", type=float, default=0.5,
                    required=False)
    ap.add_argument('--baseline', help="Required baseline (in hours) for the observation.", type=float, default=0.5,
                    required=False)
    ap.add_argument('--since', help="yyyy-mm-dd date since when you want to start the plan (defaults to today).",
                    type=str, default=None,
                    required=False)
    ap.add_argument('--error_sigma', help="Sigma to be used for epoch and period errors.", type=int, default=2,
                    required=False)
    ap.add_argument('--no_error_alert', help="Will not block imprecise observations to be plotted.",
                    action='store_true', required=False)
    ap.add_argument('--time_unit', help="Time unit to be used for the light curve measurements.", default=None,
                    required=False)
    args = ap.parse_args()
    run_plan(args)
