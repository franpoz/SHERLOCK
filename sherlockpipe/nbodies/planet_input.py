class PlanetInput:
    """
    Defines the planet parameters for system stability calculations
    """
    def __init__(self, period, radius, mass_low=None, mass_max=None, eccentricity_low=0, eccentricity_up=0, omega=0,
                 mass_bins=3, ecc_bins=3):
        self.period = period
        self.radius = radius
        self.eccentricity_low = eccentricity_low
        self.eccentricity_up = eccentricity_up
        self.omega = omega
        self.mass_low = mass_low
        self.mass_up = mass_max
        self.mass_bins = mass_bins if mass_bins is not None else 3
        self.ecc_bins = ecc_bins if ecc_bins is not None else 3
