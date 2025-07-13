from watson.watson import Watson

from sherlockpipe.loading.tool_with_candidate import ToolWithCandidate


class Vetter(ToolWithCandidate):
    watson = None

    def __init__(self, object_dir, vetting_dir, is_candidate_from_search, candidates_df) -> None:
        super().__init__(is_candidate_from_search, candidates_df)
        self.watson = Watson(object_dir, vetting_dir)

    def run(self, cpus, **kwargs):
        self.watson.vetting_with_data(kwargs['candidate'], kwargs['star_df'], kwargs['transits_df'],
                                      cpus, transits_mask=kwargs["transits_mask"],
                                      iatson_enabled=kwargs['iatson_enabled'], iatson_inputs_save=True,
                                      gpt_enabled=kwargs['gpt_enabled'], gpt_api_key=kwargs['gpt_api_key'],
                                      only_summary=kwargs['only_summary'], triceratops_bins=kwargs['triceratops_bins'],
                                      triceratops_scenarios=kwargs['triceratops_scenarios'],
                                      triceratops_curve_file=kwargs['triceratops_curve_file'],
                                      triceratops_contrast_curve_file=kwargs['triceratops_contrast_curve_file'],
                                      triceratops_additional_stars_file=kwargs['triceratops_additional_stars_file'],
                                      triceratops_sigma_mode=kwargs['triceratops_sigma_mode'],
                                      triceratops_ignore_ebs=kwargs['triceratops_ignore_ebs'],
                                      triceratops_resolved_companion=kwargs['triceratops_resolved_companion'],
                                      triceratops_ignore_background_stars=kwargs['triceratops_ignore_background_stars'],
                                      sectors=kwargs['sectors']
                                      )

    def object_dir(self):
        return self.watson.object_dir
