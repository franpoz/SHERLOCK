from argparse import ArgumentParser

from sherlockpipe.single_transits.report import MoriartyReport
from sherlockpipe.single_transits.run import run_moriarty

if __name__ == '__main__':
    ap = ArgumentParser(description='Vetting of Sherlock objects of interest')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--mask_candidate', type=str, default=[], help="The candidate signal to be used.", required=False)
    ap.add_argument('--batch_size', type=int, default=256, help="The candidate signal to be used.", required=False)
    ap.add_argument('--threshold', type=float, default=0.5, help="The candidate signal to be used.", required=False)
    args = ap.parse_args()
    run_moriarty(args.object_dir, [int(candidate) - 1 for candidate in args.mask_candidate.split(',')], args.batch_size, args.threshold)
