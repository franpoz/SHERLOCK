import math

import numpy as np
from lcbuilder.HarmonicSelector import HarmonicSelector

from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector
from sherlockpipe.scoring.SnrBorderCorrectedSignalSelector import CorrectedBorderSignalSelection


class QuorumSnrBorderCorrectedSignalSelector(BasicSignalSelector):
    """
    Given the same correction applied in SnrBorderCorrectedSignalSelector, the best signal is selected by a voting
    mechanism. Those signals with same epoch and period vote for the same signal, whose SNR gets proportionally
    increased by the number of votes.
    """
    def __init__(self, strength=1, min_quorum=0):
        super().__init__()
        self.strength = strength
        self.min_quorum = min_quorum
        self.zero_epsilon = 1e-6

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        basic_signal_selection = super().select(id_run, sherlock_target, star_info, transits_min_count, time, lcs,
                                                transit_results, wl, cadence)
        index_snr_period_t0_array = [[key, transit_result.snr * (transit_result.border_score * self.zero_epsilon),
                                        transit_result.period, transit_result.t0]
                                        for key, transit_result in transit_results.items()]
        # index_snr_period_t0_array = sorted(index_snr_period_t0_array, key=lambda k: k[2])
        i = 0
        number_of_voters = len(index_snr_period_t0_array)
        votes = list(range(0, number_of_voters))
        while i < len(index_snr_period_t0_array):
            if i != votes[i]:  # we already compared the vote
                i = i + 1
                continue
            period = index_snr_period_t0_array[i][2]
            t0 = index_snr_period_t0_array[i][3]
            j = i + 1
            while j < len(index_snr_period_t0_array):
                compared_period = index_snr_period_t0_array[j][2]
                compared_t0 = index_snr_period_t0_array[j][3]
                votes[j] = i if HarmonicSelector.is_harmonic(t0, compared_t0, period, compared_period) else votes[j]
                j = j + 1
            i = i + 1
        votes_counts = [votes.count(vote) for vote in votes]
        corrected_snrs = [index_snr_period_t0_array[key][1] +
                          index_snr_period_t0_array[key][1] * (votes_count - 1) / number_of_voters * self.strength
                for key, votes_count in enumerate(votes_counts)]
        number_corrected_snrs_length = len(np.where(~np.isnan(corrected_snrs)))
        if number_corrected_snrs_length == 0:
            best_signal_score = 0
            selected_signal = 0
            best_signal_snr_index = 0
            best_signal_snr = 0
        else:
            best_signal_snr = np.nanmax(corrected_snrs)
            best_signal_snr_index = np.nanargmax(corrected_snrs)
            selected_signal = transit_results[best_signal_snr_index]
            selected_signal_sde = selected_signal.sde
            max_votes_rate = max(votes_counts) / len(votes)
            if best_signal_snr > sherlock_target.snr_min and selected_signal_sde > sherlock_target.sde_min and \
                    max_votes_rate >= self.min_quorum:
                best_signal_score = 1
            else:
                best_signal_score = 0
        return CorrectedQuorumBorderSignalSelection(best_signal_score, best_signal_snr,
                                                    basic_signal_selection.curve_index,
                                                    transit_results[basic_signal_selection.curve_index],
                                                    best_signal_snr_index, selected_signal,
                                                    votes_counts[best_signal_snr_index])

class CorrectedQuorumBorderSignalSelection(CorrectedBorderSignalSelection):
    def __init__(self, score, corrected_snr, original_curve_index, original_transit_result, final_curve_index,
                 final_transit_result, votes):
        super().__init__(score, corrected_snr, original_curve_index, original_transit_result, final_curve_index,
                         final_transit_result)
        self.votes = votes

    def get_message(self):
        curve_name = "PDCSAP_FLUX" if self.curve_index == 0 else str(self.curve_index - 1)
        original_curve_name = "PDCSAP_FLUX" if self.original_curve_index == 0 else str(self.original_curve_index - 1)
        return "Elected signal with QUORUM algorithm from " + str(self.votes) + " VOTES --> NAME: " + curve_name + \
               "\tPeriod:" + str(self.transit_result.period) + \
               "\tCORR_SNR: " + str(self.corrected_snr) + \
               "\tSNR: " + str(self.transit_result.snr) + \
               "\tSDE: " + str(self.transit_result.sde) + \
               "\tFAP: " + str(self.transit_result.fap) + \
               "\tBORDER_SCORE: " + str(self.transit_result.border_score) + \
               "\nProposed selection with BASIC algorithm was --> NAME: " + original_curve_name + \
               "\tPeriod:" + str(self.original_transit_result.period) + \
               "\tSNR: " + str(self.original_transit_result.snr)
