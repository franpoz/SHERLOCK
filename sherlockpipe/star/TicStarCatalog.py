import numpy as np
from sherlockpipe.star.StarCatalog import StarCatalog
from astroquery.mast import Catalogs
import transitleastsquares as tls


class TicStarCatalog(StarCatalog):
    def __init__(self):
        super().__init__()

    def catalog_info(self, id):
        """Takes TIC_ID, returns stellar information from online catalog using Vizier"""
        if type(id) is not int:
            raise TypeError('TIC_ID ID must be of type "int"')
        result = Catalogs.query_criteria(catalog="Tic", ID=id).as_array()
        Teff = result[0][64]
        lum = result[0]['lum']
        logg = result[0][66]
        radius = result[0][70]
        radius_max = result[0][71]
        radius_min = result[0][71]
        mass = result[0][72]
        mass_max = result[0][73]
        mass_min = result[0][73]
        if lum is None or np.isnan(lum):
            lum = self.star_luminosity(Teff, radius)
        ld, mass, mass_min, mass_max, radius, radius_min, radius_max = tls.catalog_info(TIC_ID=id)
        return (ld, Teff, lum, logg, radius, radius_min, radius_max, mass, mass_min, mass_max)
