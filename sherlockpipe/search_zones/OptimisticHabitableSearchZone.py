import math
from sherlockpipe.search_zones.SearchZone import SearchZone
from sherlockpipe.star.HabitabilityCalculator import HabitabilityCalculator
from sherlockpipe.star.starinfo import StarInfo


class OptimisticHabitableSearchZone(SearchZone):
    """
    Implementation for calculating minimum and maximum search periods for the habitable zone of an input star.
    """
    def __init__(self):
        super().__init__()

    def calculate_period_range(self, star_info: StarInfo):
        hz_aus = HabitabilityCalculator().calculate_hz(star_info.teff, star_info.lum)
        GM = 7.496e-6 * star_info.mass
        period_min = math.sqrt((hz_aus[0] ** 3) * GM)
        period_max = math.sqrt((hz_aus[3] ** 3) * GM)
        return period_min, period_max
