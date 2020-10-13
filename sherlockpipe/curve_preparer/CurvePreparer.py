from abc import ABC, abstractmethod


class CurvePreparer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def prepare(self, object_info, time, flux, flux_err):
        pass