from sherlockpipe.star.StarCatalog import StarCatalog
from astroquery.vizier import Vizier
import transitleastsquares as tls


class KicStarCatalog(StarCatalog):
    def __init__(self):
        super().__init__()

    def catalog_info(self, id):
        """Takes KIC_ID, returns stellar information from online catalog using Vizier"""
        if type(id) is not int:
            raise TypeError('KIC_ID ID must be of type "int"')
        columns = ["Teff", "log(g)", "Rad", "E_Rad", "e_Rad", "Mass", "E_Mass", "e_Mass"]
        catalog = "J/ApJS/229/30/catalog"
        result = (
            Vizier(columns=columns)
                .query_constraints(KIC=id, catalog=catalog)[0]
                .as_array()
        )
        Teff = result[0][0]
        logg = result[0][1]
        radius = result[0][2]
        radius_max = result[0][3]
        radius_min = result[0][4]
        mass = result[0][5]
        mass_max = result[0][6]
        mass_min = result[0][7]
        lum = self.star_luminosity(Teff, radius)
        ld, mass, mass_min, mass_max, radius, radius_min, radius_max = tls.catalog_info(KIC_ID=id)
        return (ld, Teff, lum, logg, radius, radius_min, radius_max, mass, mass_min, mass_max)
