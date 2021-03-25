import math
from sherlockpipe.search_zones.SearchZone import SearchZone
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from lcbuilder.star.starinfo import StarInfo


class NeptunianDesertSearchZone(SearchZone):
    def __init__(self):
        super().__init__()

    def calculate_period_range(self, star_info: StarInfo):
        # TODO create a proper calculation of the periods given the effective temperatures which would define the Neptunian desert
        return 0.5, 3
