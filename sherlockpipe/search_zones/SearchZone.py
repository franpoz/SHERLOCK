from abc import ABC, abstractmethod

from lcbuilder.star.starinfo import StarInfo


class SearchZone(ABC):
    """
    Abstract class to be implemented for calculating minimum and maximum search periods for an input star.
    """
    def __init__(self):
        pass

    @abstractmethod
    def calculate_period_range(self, star_info: StarInfo):
        """
        Calculates the minimum and maximum periods for the given star_info
        @param star_info: the star where the range should be calculated
        @return: a tuple of minimum_period and maximum_period
        """
        pass
