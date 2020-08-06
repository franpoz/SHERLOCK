from abc import ABC, abstractmethod


class StarCatalog(ABC):
    sun_teff = 5700

    def __init__(self):
        pass

    @abstractmethod
    def catalog_info(self, id):
        pass

    def star_luminosity(self, t_eff, star_radius):
        return (star_radius ** 2) * (t_eff ** 4) / (self.sun_teff ** 4)
