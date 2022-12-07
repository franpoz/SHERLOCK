import numpy as np

from sherlockpipe.scoring.SignalSelector import SignalSelector, SignalSelection


class BasicSignalSelector(SignalSelector):
    """
    Selects the signal with best SNR
    """
    def __init__(self):
        super().__init__()

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        detrends_snr = np.nan_to_num([transit_result.snr
                                      for key, transit_result in transit_results.items()])
        best_signal_snr = np.nanmax(detrends_snr)
        best_signal_snr_index = np.nanargmax(detrends_snr)
        selected_signal_sde = transit_results[best_signal_snr_index].sde
        selected_signal = transit_results[best_signal_snr_index]
        if best_signal_snr > sherlock_target.snr_min and selected_signal_sde > sherlock_target.sde_min:  # and SDE[a] > SDE_min and FAP[a] < FAP_max):
            best_signal_score = 1
        else:
            best_signal_score = 0
        return SignalSelection(best_signal_score, best_signal_snr_index, selected_signal)
