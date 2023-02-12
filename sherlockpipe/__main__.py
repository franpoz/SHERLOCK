from argparse import ArgumentParser
from os import path
from sherlockpipe.loading.run import run_search


if __name__ == '__main__':
    ap = ArgumentParser(description='Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-based seeKers')
    ap.add_argument('--properties', help="Additional properties to be loaded into Sherlock run ", required=True)
    ap.add_argument('--explore', dest='explore', action='store_true',
                    help="Whether to run using mcmc or ns. Default is ns.")
    args = ap.parse_args()
    run_search(args.properties, args.explore)
