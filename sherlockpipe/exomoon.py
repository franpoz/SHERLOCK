import ellc
import numpy as np
import astropy.constants as ac
import astropy.units as u

class ExoMoonLeastSquares:
    G = 6.674e-11  # m3 kg-1 s-2
    au_to_Rsun = 215.032
    Msolar_to_kg = 2.e30

    def compute_semimajor_axis(self, major_mass, minor_period):
        period_seconds = minor_period * 24. * 3600.
        mass_kg = major_mass * self.Msolar_to_kg
        a1 = (self.G * mass_kg * period_seconds ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
        return a1 / 1.496e11

    def compute_hill_radius(self, major_mass, minor_mass, semimajor_axis, eccentricity=0):
        return semimajor_axis * (1 - eccentricity) * (major_mass / (3 * minor_mass) ** (1 / 3))

    def compute_moon_min_max_orbit_periods(self, star_mass, planet_mass, planet_ecc, planet_semimajor_axis):
        planet_hill_r = self.compute_hill_radius(star_mass, planet_mass, planet_semimajor_axis, planet_ecc)
        return 0.5, planet_hill_r

    def compute_moon_orbit_transit_length(self, star_radius, planet_radius,
                                          planet_semimajor_axis, planet_period, moon_semimajor_axis,
                                          moon_eccentrictiy=0, moon_arg_periastron=0, moon_inclination=90,
                                          planet_eccentricity=0, planet_arg_periastron=0, planet_inclination=90):
        #planet_period / np.pi * np.asin(np.sqrt((star_radius + planet_radius) * 2 - (b * star_radius) * 2) / 2)
        return 2 * moon_semimajor_axis / (planet_semimajor_axis * 2 * np.pi) * planet_period


    def compute_moon_period_grid(self, min, max, mode="log", samples=20):
        if "log" == mode:
            return np.logspace(min, max, samples)
        else:
            return np.linspace(min, max, samples)

    def tokenize_transits_moon_orbit_ranges(self, time, flux, flux_err, star_mass, star_radius, planet_mass, planet_radius,
                                            planet_period, planet_t0, moon_semimajor_axis, planet_semimajor_axis,
                                            moon_period, moon_eccentrictiy=0,
                                            moon_arg_periastron=0, moon_inclination=90,
                                            planet_eccentricity=0, planet_arg_periastron=0, planet_inclination=90):
        moon_semimajor_axis = self.compute_semimajor_axis(planet_mass, moon_period)
        planet_semimajor_axis = self.compute_semimajor_axis(star_mass, planet_period)
        moon_orbit_transit_length = self.compute_moon_orbit_transit_length(star_radius, planet_radius,
                                                                           planet_semimajor_axis, planet_period,
                                                                           moon_semimajor_axis)
        # TODO we probably need to define left_transit_length and right_transit_length depending on moon orbit parameters
        all_t0s = np.linspace(planet_t0, time[len(time) - 1], planet_period)
        moon_orbit_tokens = [[t0, t0 - moon_orbit_transit_length / 2, t0 + moon_orbit_transit_length / 2] for t0 in all_t0s]
        return moon_orbit_tokens

    def subtract_planet_transit(self, ab, star_radius, star_mass, time, flux, planet_radius, planet_t0,
                                planet_period, planet_inc=90):
        P1 = planet_period * u.day
        a = np.cbrt((ac.G * star_mass * P1 ** 2) / (4 * np.pi ** 2)).to(u.au)
        model = ellc.lc(
            t_obs=time,
            radius_1=star_radius.to(u.au) / a,  # star radius convert from AU to in units of a
            radius_2=planet_radius.to(u.au) / a,
            # convert from Rearth (equatorial) into AU and then into units of a
            sbratio=0,
            incl=planet_inc,
            light_3=0,
            t_zero=planet_t0,
            period=planet_period,
            a=None,
            q=1e-6,
            f_c=None, f_s=None,
            ldc_1=ab, ldc_2=None,
            gdc_1=None, gdc_2=None,
            didt=None,
            domdt=None,
            rotfac_1=1, rotfac_2=1,
            hf_1=1.5, hf_2=1.5,
            bfac_1=None, bfac_2=None,
            heat_1=None, heat_2=None,
            lambda_1=None, lambda_2=None,
            vsini_1=None, vsini_2=None,
            t_exp=None, n_int=None,
            grid_1='default', grid_2='default',
            ld_1='quad', ld_2=None,
            shape_1='sphere', shape_2='sphere',
            spots_1=None, spots_2=None,
            exact_grav=False, verbose=1)
        return flux - model + 1

    def compute_moon_transit_scenarios(self, time, flux, planet_t0, moon_period, moon_orbit_ranges, moon_transit_length):
        #TODO need to take into account "prograde" or "retrograde" orbit
        orbit_scenarios = []
        tau = moon_transit_length / 2
        for moon_orbit_range in moon_orbit_ranges:
            time_args = np.argwhere(time[(time > moon_orbit_range[1]) & (time < moon_orbit_range[2])])
            #TODO we'd need to fill measurement gaps (detected from the time array)
            time_orbit_range = time[time_args]
            flux_orbit_range = flux[time_args]
            tau1 = time_orbit_range - planet_t0
            flux_alpha = np.arccos(tau1 / tau)
            flux_alpha_comp = 2 * np.pi - flux_alpha
            phase_delta = (moon_orbit_range[0] - planet_t0) / moon_period
            flux_alpha_0 = flux_alpha + phase_delta
            flux_alpha_comp_0 = flux_alpha_comp + phase_delta
            orbit_scenarios = orbit_scenarios.append([time_orbit_range, tau1, flux_orbit_range, flux_alpha_0, flux_alpha_comp_0])
        return orbit_scenarios


    def run(self):
        min_period = 0.5 # TODO compute this value somehow
        max_period = self.compute_hill_radius(self.star_mass, self.planet_mass, self.planet_semimajor_axis)
        period_grid = self.compute_moon_period_grid(min_period, max_period)
        planet_mass_grid = self.planet_mass_grid
        moon_inc_grid = self.moon_inc_grid
        moon_ecc_grid = self.moon_ecc_grid
        moon_arg_periastron_grid = self.moon_arg_periastron_grid
        flux = self.subtract_planet_transit(self.ab, self.star_radius, self.star_mass, self.time, self.flux,
                                            self.planet_radius, self.planet_t0, self.planet_period, self.planet_inc)
        for planet_mass in planet_mass_grid:
            for moon_inc in moon_inc_grid:
                for moon_ecc in moon_ecc_grid:
                    for moon_arg_periastron in moon_arg_periastron_grid:
                        for moon_period in period_grid:
                            #TODO moon_orbit_ranges should use moon_radius ?
                            moon_semimajor_axis = self.moon_semimajor_axis = self.compute_semimajor_axis(planet_mass, moon_period)
                            planet_semimajor_axis = self.planet_semimajor_axis = self.compute_semimajor_axis(self.star_mass, self.planet_period)
                            moon_transit_length = self.compute_moon_orbit_transit_length(self.star_radius, self.planet_radius,
                                                                   planet_semimajor_axis, self.planet_period,
                                                                   moon_semimajor_axis, moon_ecc, moon_arg_periastron, moon_inc,
                                                                   self.planet_ecc, self.planet_arg_periastron, self.planet_inc)
                            moon_orbit_ranges = self.tokenize_transits_moon_orbit_ranges(self.time, flux, self.flux_err,
                                                                     self.star_mass, self.star_radius, planet_mass,
                                                                     self.planet_radius, self.planet_period,
                                                                     self.planet_t0, moon_semimajor_axis, planet_semimajor_axis,
                                                                     moon_period, moon_ecc,
                                                                     moon_arg_periastron, moon_inc, self.planet_ecc,
                                                                     self.planet_arg_periastron, self.planet_inc)
                            moon_transit_scenarios = self.compute_moon_transit_scenarios(self.time, flux, self.planet_t0, moon_period, moon_orbit_ranges, moon_transit_length)


