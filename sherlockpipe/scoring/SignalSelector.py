from abc import ABC, abstractmethod


class SignalSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select(self, transit_result_detrends, snr_min, detrend_method, wl):
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
