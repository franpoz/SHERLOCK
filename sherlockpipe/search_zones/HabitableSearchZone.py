import math
from sherlockpipe.search_zones.SearchZone import SearchZone
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from lcbuilder.star.starinfo import StarInfo


class HabitableSearchZone(SearchZone):
    """
    Implementation for calculating minimum and maximum search periods for the habitable zone of an input star.
    """
    def __init__(self):
        super().__init__()

    def calculate_period_range(self, star_info: StarInfo):
        """
        Calculates the minimum and maximum periods for the Habitable Zone (from kopparapu et al 2013)
        given the star_info.

        :param star_info: the star where the range should be calculated
        :return: a tuple of minimum_period and maximum_period
        """
        habitability_calc = HabitabilityCalculator()
        hz_periods = habitability_calc.calculate_hz_periods(star_info.teff, star_info.lum, star_info.mass)
        if hz_periods is None:
            return None
        return hz_periods[1], hz_periods[2]
