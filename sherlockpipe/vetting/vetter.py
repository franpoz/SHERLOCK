import logging
import os

from watson.watson import Watson

from sherlockpipe.loading.tool_with_candidate import ToolWithCandidate


class Vetter(ToolWithCandidate):
    """
    Coordinates the vetting of transit candidates using the WATSON pipeline.

    Extends ToolWithCandidate to wrap the Watson vetting tool, providing
    convenient methods for running vetting with pre-loaded data.
    """

    watson = None

    def __init__(self, object_dir, vetting_dir, is_candidate_from_search, candidates_df) -> None:
        """
        Initialize the vetter.

        Parameters
        ----------
        object_dir : str
            Directory containing the object's data files.
        vetting_dir : str
            Directory where vetting outputs will be written.
        is_candidate_from_search : bool
            Whether the candidate comes from a SHERLOCK search.
        candidates_df : pandas.DataFrame
            DataFrame of candidate information.
        """
        super().__init__(is_candidate_from_search, candidates_df)
        self.watson = Watson(object_dir, vetting_dir)

    def run(self, cpus, **kwargs):
        """
        Execute the vetting analysis via WATSON.

        Parameters
        ----------
        cpus : int
            Number of CPUs to use.
        \*\*kwargs : dict
            Vetting parameters passed to watson.vetting_with_data, including
            candidate, star_df, transits_df, transits_mask, and TRICERATOPS options.
        """
        if kwargs['iatson_enabled']:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            try:
                import tensorflow as tf
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError:
                logging.warning("TensorFlow GPU state was already initialized; IATSON may not run on CPU.")

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
                                      sectors=kwargs['sectors'],
                                      triceratops_external_lc_files=kwargs.get('triceratops_external_lc_files'),
                                      triceratops_filt_lcs=kwargs.get('triceratops_filt_lcs'),
                                      main_fit_dir=kwargs.get('main_fit_dir'),
                                      odd_fit_dir=kwargs.get('odd_fit_dir'),
                                      even_fit_dir=kwargs.get('even_fit_dir')
                                      )

    def object_dir(self):
        """
        Return the object directory used by the internal WATSON instance.

        Returns
        -------
        str
            Path to the object's data directory.
        """
        return self.watson.object_dir
