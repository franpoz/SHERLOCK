class PlanetInput:
    """
    Defines the planet parameters for system stability calculations
    """
    def __init__(self, period, period_low_err, period_up_err,
                 radius, radius_low_err, radius_up_err, eccentricity, ecc_low_err, ecc_up_err,
                 inclination, inc_low_err, inc_up_err, omega, omega_low_err, omega_up_err,
                 mass=None, mass_low_err=None, mass_up_err=None, period_bins=3,
                 mass_bins=3, ecc_bins=3, inc_bins=3, omega_bins=3):
        self.period = period
        self.period_low_err = period_low_err
        self.period_up_err = period_up_err
        self.radius = radius
        self.radius_low_err = radius_low_err
        self.radius_up_err = radius_up_err
        self.eccentricity = eccentricity
        self.ecc_low_err = ecc_low_err
        self.ecc_up_err = ecc_up_err
        self.inclination = inclination
        self.inc_low_err = inc_low_err
        self.inc_up_err = inc_up_err
        self.omega = omega
        self.omega_low_err = omega_low_err
        self.omega_up_err = omega_up_err
        self.mass = mass
        self.mass_low_err = mass_low_err
        self.mass_up_err = mass_up_err
        self.mass_bins = mass_bins if mass_bins is not None else 3
        self.period_bins = period_bins if period_bins is not None else 3
        self.ecc_bins = ecc_bins if ecc_bins is not None else 3
        self.inc_bins = inc_bins if inc_bins is not None else 3
        self.omega_bins = omega_bins if omega_bins is not None else 3
