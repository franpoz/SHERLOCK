import numpy as np

from sherlockpipe.scoring.SignalSelector import SignalSelector, SignalSelection


class BasicSignalSelector(SignalSelector):
    """
    Selects the signal with best SNR
    """
    def __init__(self):
        """Initialize the Basic signal selector."""
        super().__init__()

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        """Select the signal with the highest SNR among all detrended curves.

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
        SignalSelection
            The selection result with score, curve index, and transit result.
        """
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
