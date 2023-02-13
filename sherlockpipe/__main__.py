from argparse import ArgumentParser
from sherlockpipe.search.run import run_search


if __name__ == '__main__':
    ap = ArgumentParser(description='Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-based seeKers')
    ap.add_argument('--properties', help="Additional properties to be loaded into Sherlock run ", required=True)
    ap.add_argument('--results_dir', help="Directory where results should be written to", required=False, default=None)
    ap.add_argument('--explore', dest='explore', action='store_true',
                    help="Whether to run using mcmc or ns. Default is ns.")
    args = ap.parse_args()
    run_search(args.properties, args.explore, args.results_dir)
