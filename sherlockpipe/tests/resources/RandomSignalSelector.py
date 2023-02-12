import random
from sherlockpipe.scoring.SignalSelector import SignalSelector, SignalSelection


class RandomSignalSelector(SignalSelector):
    """
    Implements a random signal selection from a list of detrends. Only for testing.
    """
    def __init__(self):
        super().__init__()

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        best_signal_snr_index = random.randrange(0, len(transit_results) - 1, 1)
        best_signal = transit_results[best_signal_snr_index]
        best_signal_snr = best_signal.snr
        if best_signal_snr > sherlock_target.snr_min:  # and SDE[a] > SDE_min and FAP[a] < FAP_max):
            best_signal_score = 1
        else:
            best_signal_score = 0
        # You could also extend SignalSelector class to provide more information about your choice.
        return SignalSelection(best_signal_score, best_signal_snr_index, best_signal)