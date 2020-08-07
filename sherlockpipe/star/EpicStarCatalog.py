from astroquery.vizier import Vizier
from sherlockpipe.star.StarCatalog import StarCatalog
import transitleastsquares as tls


class EpicStarCatalog(StarCatalog):
    def __init__(self):
        super().__init__()

    def catalog_info(self, id):
        """Takes EPIC_ID, returns stellar information from online catalog using Vizier"""
        if type(id) is not int:
            raise TypeError('EPIC_ID ID must be of type "int"')
        if (id < 201000001) or (id > 251813738):
            raise TypeError("EPIC_ID ID must be in range 201000001 to 251813738")
        columns = ["Teff", "logg", "Rad", "E_Rad", "e_Rad", "Mass", "E_Mass", "e_Mass"]
        catalog = "IV/34/epic"
        result = (
            Vizier(columns=columns)
                .query_constraints(ID=id, catalog=catalog)[0]
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
        ld, mass, mass_min, mass_max, radius, radius_min, radius_max = tls.catalog_info(EPIC_ID=id)
        return (ld, Teff, lum, logg, radius, radius_min, radius_max, mass, mass_min, mass_max)
