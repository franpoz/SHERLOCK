from abc import ABC, abstractmethod


class SignalSelector(ABC):
    """
    Base class to perform a selection of the best candidate signal among a list of them.
    """
    def __init__(self):
        pass

    @abstractmethod
    def select(self, id_run, sherlock_target, star_info, transits_min_count, time, lcs, transit_results, wl, cadence):
        """
        Selects the best signal given the implementation of this method.

        :param transit_result_detrends: a list of transit results to be explored.
        :param snr_min: the minimum snr to allow a signal to be considered good enough.
        :param sde_min: the minimum sde to allow a signal to be considered good enough.
        :param detrend_method: the detrend method used to obtain the candidate transit results
        :param wl: the window length used by the detrend method used.
        :return: a SignalSelection with the parameters that characterize the best selection.
        """
        pass

class SignalSelection:
    def __init__(self, score, curve_index, transit_result):
        self.score = score
        self.curve_index = curve_index
        self.transit_result = transit_result

    def get_message(self):
        curve_name = "PDCSAP_FLUX" if self.curve_index == 0 else str(self.curve_index - 1)
        return "Chosen signal with BASIC algorithm --> NAME: " + curve_name + \
               "\tPeriod:" + str(self.transit_result.period) + \
               "\tSNR: " + str(self.transit_result.snr) + \
               "\tSDE: " + str(self.transit_result.sde) + \
               "\tFAP: " + str(self.transit_result.fap) + \
               "\tBORDER_SCORE: " + str(self.transit_result.border_score)
