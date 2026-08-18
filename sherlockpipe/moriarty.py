from argparse import ArgumentParser

from sherlockpipe.single_transits.report import MoriartyReport
from sherlockpipe.single_transits.run import run_moriarty

if __name__ == '__main__':
    ap = ArgumentParser(description='Vetting of Sherlock objects of interest')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--mask_candidate', type=str, default='', help="The candidate signals to be masked.", required=False)
    ap.add_argument('--lc_file', type=str, default='lc_predetrends.csv', help="The light curve csv to use.", required=False)
    ap.add_argument('--batch_size', type=int, default=256, help="The candidate signal to be used.", required=False)
    ap.add_argument('--threshold', type=float, default=0.5, help="The candidate signal to be used.", required=False)
    ap.add_argument('--disable_snr_projected', action='store_true', default=False,
                    help="Disable the projected SNR check that filters phase-folded detectable periods.", required=False)
    args = ap.parse_args()
    run_moriarty(args.object_dir,
                 [int(candidate) - 1 for candidate in args.mask_candidate.split(',')] if args.mask_candidate is not '' else [],
                 args.batch_size,
                 args.threshold,
                 args.disable_snr_projected,
                 args.lc_file)
