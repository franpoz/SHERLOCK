import numpy as np

from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector
from sherlockpipe.scoring.SignalSelector import SignalSelection


class SdeBorderCorrectedSignalSelector(BasicSignalSelector):
    """
    Selects the best signal among the SDE border-corrected signals. The applied correction consists in calculating how
    many transits occur in times very close to observation starts/ends.
    """
    def __init__(self):
        """Initialize the SDE border-corrected signal selector."""
        super().__init__()
        self.zero_epsilon = 1e-6

    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        """Select the best signal applying a border-score correction to the SDE.

        The SDE of each transit result is multiplied by its border score to
        penalize signals with transits near observation edges.

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
        CorrectedBorderSdeSignalSelection
            The selection result with the original and corrected selections.
        """
        basic_signal_selection = super().select(id_run, sherlock_target, star_info, transits_min_count, time, lcs,
                                                transit_results, wl, cadence)
        signals_sde = np.nan_to_num([transit_result.sde * (transit_result.border_score + self.zero_epsilon)
                                     for key, transit_result in transit_results.items()])
        best_signal_sde = np.nanmax(signals_sde)
        best_signal_sde_index = np.nanargmax(signals_sde)
        selected_signal_snr = transit_results[best_signal_sde_index].snr
        selected_signal = transit_results[best_signal_sde_index]
        if best_signal_sde > sherlock_target.sde_min and selected_signal_snr > sherlock_target.snr_min:  # and SDE[a] > SDE_min and FAP[a] < FAP_max):
            best_signal_score = 1
        else:
            best_signal_score = 0
        return CorrectedBorderSdeSignalSelection(best_signal_score, best_signal_sde, basic_signal_selection.curve_index,
                                              transit_results[basic_signal_selection.curve_index],
                                              best_signal_sde_index, selected_signal)


class CorrectedBorderSdeSignalSelection(SignalSelection):
    """Container for the result of an SDE border-corrected signal selection."""

    def __init__(self, score, corrected_sde, original_curve_index, original_transit_result, final_curve_index,
                 final_transit_result):
        """Initialize the corrected border SDE signal selection.

        Parameters
        ----------
        score : float
            The selection score.
        corrected_sde : float
            The border-corrected SDE value of the final selection.
        original_curve_index : int
            The curve index of the original (non-corrected) selection.
        original_transit_result : TransitResult
            The transit result of the original (non-corrected) selection.
        final_curve_index : int
            The curve index of the final (corrected) selection.
        final_transit_result : TransitResult
            The transit result of the final (corrected) selection.
        """
        super().__init__(score, final_curve_index, final_transit_result)
        self.original_curve_index = original_curve_index
        self.original_transit_result = original_transit_result
        self.corrected_sde = corrected_sde

    def get_message(self):
        """Return a human-readable summary of the corrected border SDE signal selection.

        Returns
        -------
        str
            A string describing the chosen signal with its corrected SDE and
            the original signal that would have been chosen without correction.
        """
        curve_name = "PDCSAP_FLUX" if self.curve_index == 0 else str(self.curve_index - 1)
        original_curve_name = "PDCSAP_FLUX" if self.original_curve_index == 0 else str(self.original_curve_index - 1)
        return "Chosen signal with BORDER_CORRECT algorithm --> NAME: " + curve_name + \
               "\tPeriod:" + str(self.transit_result.period) + \
               "\tCORR_SDE: " + str(self.corrected_sde) + \
               "\tSNR: " + str(self.transit_result.snr) + \
               "\tSDE: " + str(self.transit_result.sde) + \
               "\tFAP: " + str(self.transit_result.fap) + \
               "\tBORDER_SCORE: " + str(self.transit_result.border_score) + \
               "\nProposed selection with BASIC algorithm was --> NAME: " + original_curve_name + \
               "\tPeriod:" + str(self.original_transit_result.period) + \
               "\tSNR: " + str(self.original_transit_result.snr)
