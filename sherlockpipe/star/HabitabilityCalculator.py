import numpy as np

'''Calculates the Habitability Zone of a star based on the Kopparapu et al (2003) equations. These equations are only
valid for stars between 2600 K < Teff < 7200 K.'''
class HabitabilityCalculator:
    sun_luminosity = 1.
    G = 6.674e-11  # m3 kg-1 s-2
    ##
    ## Stellar flux coefficients (table 3 from kopparapu et al 2013)
    ##
    e_end = 1.00
    Nout = 10000
    e = np.linspace(0.0001, e_end, Nout, endpoint=False)

    def __init__(self):
        pass

    # equation (2) from kopparapu et al 2013, for non ecentric orbits
    def __seff(self, Seff_sun, a, b, c, d, t_star):
        Seff = Seff_sun + a * t_star + b * t_star ** 2 + c * t_star ** 3 + d * t_star ** 4
        return Seff

    # equation (3) from kopparapu et al 2013, for non ecentric orbits
    def __dis(self, Seff, luminosity):
        return (luminosity / self.sun_luminosity / Seff) ** 0.5

    # having equations (3) y (4) for eccentric orbits from kopparapu et al 2013
    def __seff_prime(self, A, e):
        return A / np.sqrt(1. - e ** 2)

    def calculate_hz(self, t_eff, luminosity):
        """
        Calculates the habitable zone ranges in astronomical units
        @param t_eff: the stellar effective temperature
        @param luminosity: the stellar luminosity in sun luminosities
        @return: a tuple of size 4 with the recent venus, moist greenhouse, maximum greenhouse and early mars orbital
        semi-major axises for the given star parameters.
        """
        if t_eff < 2600 or t_eff > 7200:
            return None
        t_star = t_eff - 5700
        # Recent Venus
        s_eff_rv = 1.7763
        a_rv = 1.4335e-4
        b_rv = 3.3954e-9
        c_rv = -7.6364e-12
        d_rv = -1.1950e-15
        Seff_rv = self.__seff(s_eff_rv, a_rv, b_rv, c_rv, d_rv, t_star)
        Dis_rv = self.__dis(Seff_rv, luminosity)
        Seff_prime_rv = self.__seff_prime(Seff_rv, self.e)
        Dis_prime_rv = self.__dis(Seff_prime_rv, t_star)
        # Runaway Greenhouse
        s_eff_rg = 1.0385
        a_rg = 1.2456e-4
        b_rg = 1.4612e-8
        c_rg = -7.6345e-12
        d_rg = -1.7511e-15
        Seff_rg = self.__seff(s_eff_rg, a_rg, b_rg, c_rg, d_rg, t_star)
        Dis_rg = self.__dis(Seff_rg, luminosity)
        Seff_prime_rg = self.__seff_prime(Seff_rg, self.e)
        Dis_prime_rg = self.__dis(Seff_prime_rg, luminosity)
        # Moist Greenhouse
        s_eff_mog = 1.0146
        a_mog = 8.1884e-5
        b_mog = 1.9394e-9
        c_mog = -4.3618e-12
        d_mog = -6.8260e-16
        Seff_mog = self.__seff(s_eff_mog, a_mog, b_mog, c_mog, d_mog, t_star)
        Dis_mog = self.__dis(Seff_mog, luminosity)
        Seff_prime_mog = self.__seff_prime(Seff_mog, self.e)
        Dis_prime_mog = self.__dis(Seff_prime_mog, luminosity)
        # Maximun Greenhouse
        s_eff_mag = 0.3507
        a_mag = 5.9578e-5
        b_mag = 1.6707e-9
        c_mag = -3.0058e-12
        d_mag = -5.1925e-16
        Seff_mag = self.__seff(s_eff_mag, a_mag, b_mag, c_mag, d_mag, t_star)
        Dis_mag = self.__dis(Seff_mag, luminosity)
        Seff_prime_mag = self.__seff_prime(Seff_mag, self.e)
        Dis_prime_mag = self.__dis(Seff_prime_mag, luminosity)
        # Early Mars
        s_eff_em = 0.3207
        a_em = 5.4471e-5
        b_em = 1.5275e-9
        c_em = -2.1709e-12
        d_em = -3.8282e-16
        Seff_em = self.__seff(s_eff_em, a_em, b_em, c_em, d_em, t_star)
        Dis_em = self.__dis(Seff_em, luminosity)
        Seff_prime_em = self.__seff_prime(Seff_em, self.e)
        Dis_prime_em = self.__dis(Seff_prime_em, luminosity)
        return [Dis_rv, Dis_mog, Dis_mag, Dis_em]

    def calculate_hz_periods(self, t_eff, luminosity, mass):
        """
        Calculates the habitable zone ranges in periods
        @param t_eff: the stellar effective temperature
        @param luminosity: the stellar luminosity in sun luminosities
        @param mass: the stellar mass in sun masses
        @return: a tuple of size 4 with the recent venus, moist greenhouse, maximum greenhouse and early mars orbital
        periods for the given star parameters.
        """
        aus = self.calculate_hz(t_eff, luminosity)
        if aus is None:
            return None
        return [self.au_to_period(mass, au) for au in aus]

    def au_to_period(self, mass, au):
        """
        Calculates the orbital period for the semi-major axis assuming a circular orbit.
        @param mass: the stellar mass
        @param au: the semi-major axis in astronomical units.
        @return: the period in days
        """
        mass_kg = mass * 2.e30
        a = au * 1.496e11
        return ((a ** 3) * 4 * (np.pi ** 2) / self.G / mass_kg) ** (1. / 2.) / 3600 / 24

    def calculate_semi_major_axis(self, period, star_mass):
        period_seconds = period * 24. * 3600.
        mass_kg = star_mass * 2.e30
        a1 = (self.G * mass_kg*period_seconds ** 2/4. / (np.pi ** 2)) ** (1. / 3.)
        return a1 / 1.496e11

    def calculate_hz_score(self, t_eff, star_mass, luminosity, period):
        """
        Returns the semi-major axis and the HZ Area [I=Inner, HZ-IO=Habitable Zone (Inner Optimistic),
        HZ=Habitable Zone, HZ-OO=Habitable Zone (Outer Optimistic)
        @param t_eff: the star effective temperature
        @param star_mass: the star mass in sun masses
        @param luminosity: the star luminosity in sun luminosities
        @param period: the period to guess the semi-major axis
        @return: a tuple of semi-major axis and hz position string.
        """
        hz = self.calculate_hz(t_eff, luminosity) if t_eff is not None and not np.isnan(t_eff) else None
        a1_au = self.calculate_semi_major_axis(period, star_mass) if star_mass is not None and not np.isnan(star_mass)\
                else np.nan
        if np.isnan(a1_au) or hz is None:
            hz_position = "-"
        elif a1_au < hz[0]:
            hz_position = 'I'
        elif a1_au >= hz[0] and a1_au < hz[1]:
            hz_position = 'HZ-IO'
        elif a1_au >= hz[1] and a1_au < hz[2]:
            hz_position = 'HZ'
        elif a1_au >= hz[2] and a1_au < hz[3]:
            hz_position = 'HZ-OC'
        else:
            hz_position = 'O'
        return a1_au, hz_position
