from argparse import ArgumentParser

from sherlockpipe.validation.run import run_validate


def validation_args_parse(arguments=None):
    ap = ArgumentParser(description='Validation of Sherlock objects of interest')
    ap.add_argument('--object_dir', help="If the object directory is not your current one you need to provide the "
                                         "ABSOLUTE path", required=False)
    ap.add_argument('--candidate', type=int, default=None, help="The candidate signal to be used.", required=False)
    ap.add_argument('--properties', help="The YAML file to be used as input.", required=False)
    ap.add_argument('--cpus', type=int, default=None, help="The number of CPU cores to be used.", required=False)
    ap.add_argument('--bins', type=int, default=100, help="The number of bins to be used for the folded curve "
                                                          "validation.", required=False)
    ap.add_argument('--sigma_mode', type=str, default='flux_err', help="The way to calculate the sigma value for the "
                                                                       "validation. [flux_err|binning]", required=False)
    ap.add_argument('--scenarios', type=int, default=5, help="The number of scenarios to be used for the validation",
                    required=False)
    ap.add_argument('--contrast_curve', type=str, default=None, help="The contrast curve in csv format.",
                    required=False)
    ap.add_argument('--sectors', type=str, default=None, help="The sectors to be used.",
                    required=False)
    return ap.parse_args(arguments)


if __name__ == '__main__':
    args = validation_args_parse()
    run_validate(args)
