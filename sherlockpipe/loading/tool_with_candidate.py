import logging

import foldedleastsquares
import numpy as np
from lcbuilder.helper import LcbuilderHelper


class ToolWithCandidate:
    """
    Base class to be extended by tools that process candidates information.
    """

    def __init__(self, is_candidate_from_search, candidates_df) -> None:
        super().__init__()
        self.is_candidate_from_search = is_candidate_from_search
        self.search_candidates_df = candidates_df

    def mask_previous_candidates(self, time, flux, flux_err, candidate_id):
        """
        Masks all the candidates found in previous runs in the SHERLOCK search.

        :param time: the time array
        :param flux: the flux measurements array
        :param flux_err: the flux error measurements array
        :param candidate_id: the candidate number
        :return: time, flux and flux_err with previous candidates in-transit data masked
        """
        if self.is_candidate_aware():
            for index in np.arange(0, candidate_id - 1):
                candidate_row = self.search_candidates_df.iloc[index]
                period = candidate_row["period"]
                duration = candidate_row["duration"]
                duration = duration / 60 / 24
                t0 = candidate_row["t0"]
                logging.info("Masking candidate number %.0f with P=%.3fd, T0=%.2f and D=%.2fd", index + 1, period, t0,
                             duration)
                time, flux, flux_err = LcbuilderHelper.mask_transits(time, flux, period, duration * 2, t0, flux_err)
        return time, flux, flux_err

    def is_candidate_aware(self):
        """
        Boolean return to inform whether the candidate to be processed comes from SHERLOCK searches or is user-given.

        :return: boolean with the value
        """
        return self.is_candidate_from_search
