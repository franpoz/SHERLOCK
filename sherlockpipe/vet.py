from argparse import ArgumentParser

from sherlockpipe.vetting.run import run_vet

if __name__ == '__main__':
    ap = ArgumentParser(description='Vetting of Sherlock objects of interest')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--candidate', type=int, default=None, help="The candidate signal to be used.", required=False)
    ap.add_argument('--ml', action='store_true', default=False, help="Whether tu run WATSON-NET.", required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    ap.add_argument('--cpus', type=int, default=None, help="The number of CPU cores to be used.", required=False)
    ap.add_argument('--only_summary', action='store_true', default=False, help="Whether only the summary report should be created.", required=False)
    ap.add_argument('--gpt', action='store_true', default=False, help="Whether the GPT analysis should be run.", required=False)
    ap.add_argument('--gpt_key', type=str, default=None, help="The GPT api key.", required=False)
    ap.add_argument('--triceratops_bins', type=int, default=200, help="The number of bins to be used for the folded curve "
                                                          "triceratops validation.", required=False)
    ap.add_argument('--triceratops_sigma_mode', type=str, default='flux_err', help="The way to calculate the sigma value for the "
                                                                       "triceratops validation. [flux_err|binning]", required=False)
    ap.add_argument('--triceratops_scenarios', type=int, default=5, help="The number of scenarios to be used for the triceratops validation",
                    required=False)
    ap.add_argument('--triceratops_light_curve', type=str, default=None, help="The light curve in csv format for the triceratops validation.",
                    required=False)
    ap.add_argument('--triceratops_contrast_curve', type=str, default=None, help="The contrast curve in csv format for the triceratops validation.",
                    required=False)
    ap.add_argument("--triceratops_additional_stars", type=str, default=None,
                    help="The additional stars to use in the triceratops nearby list.",
                    required=False)
    ap.add_argument('--triceratops_ignore_ebs', action='store_true', default=False, help="Whether EB scenarios should "
                                                                             "be ruled out (this is only recommended when follow-up observations discard them)"
                                                                             "for the triceratops validation.",
                    required=False)
    ap.add_argument('--triceratops_ignore_background_stars', action='store_true', default=False,
                    help="Whether background star scenarios should "
                         "be ruled out (this is only recommended when archive imaging discards them)"
                         "for the triceratops validation.", required=False)
    ap.add_argument('--triceratops_resolved_companion', type=str, default=None,
                    help="Whether an unresolved companion is spotted or discarded for the triceratops validation."
                         " 'yes' means there is a resolved companion, 'no' means there are NO resolved companions. "
                         "This flag should not be set if the companion status is not known on beforehand",
                    required=False)
    ap.add_argument('--sectors', type=str, default=None, help="The sectors to be used.", required=False)
    args = ap.parse_args()
    run_vet(args.object_dir, args.candidate, args.properties, args.cpus,
            run_iatson=args.ml, run_gpt=args.gpt, gpt_key=args.gpt_key, only_summary=args.only_summary,
            triceratops_bins=args.triceratops_bins, triceratops_sigma_mode=args.triceratops_sigma_mode,
            triceratops_scenarios=args.triceratops_scenarios, triceratops_curve_file=args.triceratops_light_curve,
            triceratops_contrast_curve_file=args.triceratops_contrast_curve, triceratops_additional_stars_file=args.triceratops_additional_stars,
            triceratops_ignore_ebs=args.triceratops_ignore_ebs, triceratops_ignore_background_stars=args.triceratops_ignore_background_stars,
            triceratops_resolved_companion=args.triceratops_resolved_companion, sectors=args.sectors)
