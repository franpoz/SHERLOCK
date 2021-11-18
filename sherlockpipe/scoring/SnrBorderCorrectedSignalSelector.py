import numpy as np

from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector
from sherlockpipe.scoring.SignalSelector import SignalSelection


class SnrBorderCorrectedSignalSelector(BasicSignalSelector):
    """
    Selects the best signal among the SNR border-corrected signals. The applied correction consists in calculating how
    many transits occur in times very close to observation starts/ends.
    """
    def __init__(self):
        super().__init__()

    def select(self, transit_results, snr_min, sde_min, detrend_method, wl):
        basic_signal_selection = super().select(transit_results, snr_min, sde_min, detrend_method, wl)
        signals_snr = np.nan_to_num([transit_result.snr * transit_result.border_score
                                     for key, transit_result in transit_results.items()])
        best_signal_snr = np.nanmax(signals_snr)
        best_signal_snr_index = np.nanargmax(signals_snr)
        selected_signal_sde = transit_results[best_signal_snr_index].sde
        selected_signal = transit_results[best_signal_snr_index]
        if best_signal_snr > snr_min and selected_signal_sde > sde_min:  # and SDE[a] > SDE_min and FAP[a] < FAP_max):
            best_signal_score = 1
        else:
            best_signal_score = 0
        return CorrectedBorderSignalSelection(best_signal_score, best_signal_snr, basic_signal_selection.curve_index,
                                              transit_results[basic_signal_selection.curve_index],
                                              best_signal_snr_index, selected_signal)


class CorrectedBorderSignalSelection(SignalSelection):
    def __init__(self, score, corrected_snr, original_curve_index, original_transit_result, final_curve_index,
                 final_transit_result):
        super().__init__(score, final_curve_index, final_transit_result)
        self.original_curve_index = original_curve_index
        self.original_transit_result = original_transit_result
        self.corrected_snr = corrected_snr

    def get_message(self):
        curve_name = "PDCSAP_FLUX" if self.curve_index == 0 else str(self.curve_index - 1)
        original_curve_name = "PDCSAP_FLUX" if self.original_curve_index == 0 else str(self.original_curve_index - 1)
        return "Chosen signal with BORDER_CORRECT algorithm --> NAME: " + curve_name + \
               "\tPeriod:" + str(self.transit_result.period) + \
               "\tCORR_SNR: " + str(self.corrected_snr) + \
               "\tSNR: " + str(self.transit_result.snr) + \
               "\tSDE: " + str(self.transit_result.sde) + \
               "\tFAP: " + str(self.transit_result.fap) + \
               "\tBORDER_SCORE: " + str(self.transit_result.border_score) + \
               "\nProposed selection with BASIC algorithm was --> NAME: " + original_curve_name + \
               "\tPeriod:" + str(self.original_transit_result.period) + \
               "\tSNR: " + str(self.original_transit_result.snr)
