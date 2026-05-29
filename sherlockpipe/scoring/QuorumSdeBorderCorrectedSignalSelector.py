import math

import numpy as np
from lcbuilder.HarmonicSelector import HarmonicSelector

from sherlockpipe.scoring.BasicSdeSignalSelector import BasicSdeSignalSelector
from sherlockpipe.scoring.SdeBorderCorrectedSignalSelector import CorrectedBorderSdeSignalSelection


class QuorumSdeBorderCorrectedSignalSelector(BasicSdeSignalSelector):
    """
    Given the same correction applied in SnrBorderCorrectedSignalSelector, the best signal is selected by a voting
    mechanism. Those signals with same epoch and period vote for the same signal, whose SDE gets proportionally
    increased by the number of votes.
    """
    def __init__(self, strength=1, min_quorum=0):
        """Initialize the quorum SDE border-corrected signal selector.

        Parameters
        ----------
        strength : int, optional
            The strength factor applied to the voting correction, by default 1.
        min_quorum : int, optional
            The minimum fraction of votes required for a signal to be selected, by default 0.
        """
        super().__init__()
        self.strength = strength
        self.min_quorum = min_quorum
        self.zero_epsilon = 1e-6

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        """Select the best signal using quorum voting with SDE border correction.

        Signals with the same period and epoch vote for the same candidate,
        increasing its SDE proportionally by the number of votes.

        Parameters
        ----------
        id_run : int
            The current SHERLOCK run number.
        sherlock_target : SherlockTarget
            The target configuration.
        star_info : StarInfo
            The star information.
        transits_min_count : int
            The minimum number of transits required.
        time : ndarray
            The time array of the light curve.
        lcs : dict
            Dictionary of detrended light curve flux arrays.
        transit_results : dict
            Dictionary mapping curve indices to TransitResult objects.
        wl : float
            The window length used for detrending.
        cadence : float
            The cadence of the observations.

        Returns
        -------
        CorrectedQuorumBorderSdeSignalSelection
            The selection result with vote counts and corrected values.
        """
        basic_signal_selection = super().select(id_run, sherlock_target, star_info, transits_min_count, time, lcs,
                                                transit_results, wl, cadence)
        index_sde_period_t0_array = [[key, transit_result.sde * (transit_result.border_score + self.zero_epsilon),
                                        transit_result.period, transit_result.t0]
                                        for key, transit_result in transit_results.items()]
        # index_snr_period_t0_array = sorted(index_snr_period_t0_array, key=lambda k: k[2])
        i = 0
        number_of_voters = len(index_sde_period_t0_array)
        votes = list(range(0, number_of_voters))
        while i < len(index_sde_period_t0_array):
            if i != votes[i]:  # we already compared the vote
                i = i + 1
                continue
            period = index_sde_period_t0_array[i][2]
            t0 = index_sde_period_t0_array[i][3]
            j = i + 1
            while j < len(index_sde_period_t0_array):
                compared_period = index_sde_period_t0_array[j][2]
                compared_t0 = index_sde_period_t0_array[j][3]
                votes[j] = i if HarmonicSelector.is_harmonic(t0, compared_t0, period, compared_period) else votes[j]
                j = j + 1
            i = i + 1
        votes_counts = [votes.count(vote) for vote in votes]
        corrected_sdes = [index_sde_period_t0_array[key][1] +
                          index_sde_period_t0_array[key][1] * (votes_count - 1) / number_of_voters * self.strength
                for key, votes_count in enumerate(votes_counts)]
        number_corrected_sdes_length = len(np.where(~np.isnan(corrected_sdes)))
        if number_corrected_sdes_length == 0:
            best_signal_score = 0
            selected_signal = 0
            best_signal_sde_index = 0
            best_signal_sde = 0
        else:
            best_signal_sde = np.nanmax(corrected_sdes)
            best_signal_sde_index = np.nanargmax(corrected_sdes)
            selected_signal = transit_results[best_signal_sde_index]
            selected_signal_snr = selected_signal.snr
            max_votes_rate = max(votes_counts) / len(votes)
            if best_signal_sde > sherlock_target.sde_min and selected_signal_snr > sherlock_target.snr_min and \
                    max_votes_rate >= self.min_quorum:
                best_signal_score = 1
            else:
                best_signal_score = 0
        return CorrectedQuorumBorderSdeSignalSelection(best_signal_score, best_signal_sde,
                                                    basic_signal_selection.curve_index,
                                                    transit_results[basic_signal_selection.curve_index],
                                                    best_signal_sde_index, selected_signal,
                                                    votes_counts[best_signal_sde_index])


class CorrectedQuorumBorderSdeSignalSelection(CorrectedBorderSdeSignalSelection):
    """Container for the result of a quorum SDE border-corrected signal selection."""

    def __init__(self, score, corrected_sde, original_curve_index, original_transit_result, final_curve_index,
                 final_transit_result, votes):
        """Initialize the corrected quorum border SDE signal selection.

        Parameters
        ----------
        score : float
            The selection score.
        corrected_sde : float
            The quorum-corrected SDE value of the final selection.
        original_curve_index : int
            The curve index of the original (non-corrected) selection.
        original_transit_result : TransitResult
            The transit result of the original (non-corrected) selection.
        final_curve_index : int
            The curve index of the final (corrected) selection.
        final_transit_result : TransitResult
            The transit result of the final (corrected) selection.
        votes : int
            The number of votes received by the selected signal.
        """
        super().__init__(score, corrected_sde, original_curve_index, original_transit_result, final_curve_index,
                         final_transit_result)
        self.votes = votes

    def get_message(self):
        """Return a human-readable summary of the quorum SDE signal selection.

        Returns
        -------
        str
            A string describing the elected signal with its vote count,
            corrected SDE, and the original signal proposed by the basic algorithm.
        """
        curve_name = "PDCSAP_FLUX" if self.curve_index == 0 else str(self.curve_index - 1)
        original_curve_name = "PDCSAP_FLUX" if self.original_curve_index == 0 else str(self.original_curve_index - 1)
        return "Elected signal with QUORUM algorithm from " + str(self.votes) + " VOTES --> NAME: " + curve_name + \
               "\tPeriod:" + str(self.transit_result.period) + \
               "\tCORR_SDE: " + str(self.corrected_sde) + \
               "\tSNR: " + str(self.transit_result.snr) + \
               "\tSDE: " + str(self.transit_result.sde) + \
               "\tFAP: " + str(self.transit_result.fap) + \
               "\tBORDER_SCORE: " + str(self.transit_result.border_score) + \
               "\nProposed selection with BASIC algorithm was --> NAME: " + original_curve_name + \
               "\tPeriod:" + str(self.original_transit_result.period) + \
               "\tSNR: " + str(self.original_transit_result.snr)
