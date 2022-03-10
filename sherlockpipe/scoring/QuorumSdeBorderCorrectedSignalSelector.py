import math

import numpy as np

from sherlockpipe.scoring.BasicSdeSignalSelector import BasicSdeSignalSelector
from sherlockpipe.scoring.SdeBorderCorrectedSignalSelector import CorrectedBorderSdeSignalSelection


class QuorumSdeBorderCorrectedSignalSelector(BasicSdeSignalSelector):
    """
    Given the same correction applied in SnrBorderCorrectedSignalSelector, the best signal is selected by a voting
    mechanism. Those signals with same epoch and period vote for the same signal, whose SNR gets proportionally
    increased by the number of votes.
    """
    def __init__(self, strength=1, min_quorum=0):
        super().__init__()
        self.strength = strength
        self.min_quorum = min_quorum

    def select(self, transit_results, snr_min, sde_min, detrend_method, wl):
        basic_signal_selection = super().select(transit_results, snr_min, sde_min, detrend_method, wl)
        index_sde_period_t0_array = [[key, transit_result.sde * transit_result.border_score,
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
                votes[j] = i if self.is_harmonic(t0, compared_t0, period, compared_period) else votes[j]
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
            if best_signal_sde > sde_min and selected_signal_snr > snr_min and max_votes_rate >= self.min_quorum:
                best_signal_score = 1
            else:
                best_signal_score = 0
        return CorrectedQuorumBorderSdeSignalSelection(best_signal_score, best_signal_sde,
                                                    basic_signal_selection.curve_index,
                                                    transit_results[basic_signal_selection.curve_index],
                                                    best_signal_sde_index, selected_signal,
                                                    votes_counts[best_signal_sde_index])

    def is_harmonic(self, a_t0, b_t0, a_period, b_period):
        multiplicity = self.multiple_of(a_period, b_period, 0.01 * a_period)
        return multiplicity != 0 and self.matches_t0(a_t0, b_t0, a_period, multiplicity, 0.01 * a_t0)

    def multiple_of(self, a, b, tolerance=0.05):
        a = np.float(a)
        b = np.float(b)
        mod_ab = a % b
        mod_ba = b % a
        is_a_multiple_of_b = a >= b and a < b * 3 + tolerance * 3 and (
            (mod_ab < 1 and abs(mod_ab % 1) <= tolerance) or ((b - mod_ab) < 1 and abs((b - mod_ab) % 1) <= tolerance))
        if is_a_multiple_of_b:
            return round(a / b)
        is_b_multiple_of_a = b >= a and a > b / 3 - tolerance / 3 and (
            (mod_ba < 1 and abs(mod_ba % 1) <= tolerance) or ((a - mod_ba) < 1 and abs((a - mod_ba) % 1) <= tolerance))
        if is_b_multiple_of_a:
            return - round(b / a)
        return 0

    def matches_t0(self, a_t0, b_t0, a_period, multiplicity, tolerance=0.02):
        if multiplicity == 1:
            return abs(b_t0 - a_t0) < tolerance
        elif multiplicity < 0:
            allowed_t0s_centers = np.linspace(a_t0 - a_period * (-multiplicity), a_t0 + a_period * (-multiplicity), -2 * multiplicity + 1)
        elif multiplicity > 0:
            allowed_t0s_centers = np.linspace(a_t0 - a_period, a_t0 + a_period, 2 * multiplicity + 1)
        else:
            return False
        matching_t0s = [abs(b_t0 - allowed_t0_center) <= tolerance for allowed_t0_center in allowed_t0s_centers]
        return True in matching_t0s

class CorrectedQuorumBorderSdeSignalSelection(CorrectedBorderSdeSignalSelection):
    def __init__(self, score, corrected_sde, original_curve_index, original_transit_result, final_curve_index,
                 final_transit_result, votes):
        super().__init__(score, corrected_sde, original_curve_index, original_transit_result, final_curve_index,
                         final_transit_result)
        self.votes = votes

    def get_message(self):
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
