import multiprocessing
from abc import ABC, abstractmethod
from multiprocessing import Pool

import numpy as np
import pandas as pd


class StabilityCalculator(ABC):
    """
    Template class for system stability calculation algorithms
    """
    def __init__(self):
        pass

    def mass_from_radius(self, radius):
        return radius ** (1 / 0.55) if radius <= 12.1 else radius ** (1 / 0.01)

    def prepare_star_masses(self, star_mass_low, star_mass_up):
        return np.linspace(star_mass_low, star_mass_up, 3) if star_mass_low != star_mass_up \
            else np.linspace(star_mass_low, star_mass_up, 1)

    def prepare_planet_params(self, planet_params):
        for planet_param in planet_params:
            guessed_mass = self.mass_from_radius(planet_param.radius)
            guessed_mass_low = guessed_mass - 0.2 if guessed_mass - 0.2 > 0 else guessed_mass
            planet_param.mass_low = planet_param.mass_low if planet_param.mass_low is not None else guessed_mass_low
            planet_param.mass_up = planet_param.mass_up if planet_param.mass_up is not None else guessed_mass + 0.2
        return planet_params

    def run(self, results_dir, star_mass_low, star_mass_up, planet_params, cpus=multiprocessing.cpu_count() - 1):
        planet_params = self.prepare_planet_params(planet_params)
        star_masses = self.prepare_star_masses(star_mass_low, star_mass_up)
        planet_masses = []
        planet_periods = []
        planet_ecc = []
        for planet_param in planet_params:
            planet_masses.append(np.linspace(planet_param.mass_low, planet_param.mass_up, planet_param.mass_bins))
            planet_periods.append(planet_param.period)
            planet_ecc.append(np.linspace(planet_param.eccentricity_low, planet_param.eccentricity_up,
                                          planet_param.ecc_bins))
        masses_grid = np.array(np.meshgrid(*np.array(planet_masses))).T.reshape(-1, len(planet_masses))
        ecc_grid = np.array(np.meshgrid(*np.array(planet_ecc))).T.reshape(-1, len(planet_ecc))
        simulation_inputs = []
        for star_mass in star_masses:
            for mass_key, mass_arr in enumerate(masses_grid):
                for ecc_key, ecc_arr in enumerate(ecc_grid):
                    simulation_inputs.append(SimulationInput(star_mass, mass_arr, planet_periods, ecc_arr))
        with Pool(processes=cpus) as pool:
            simulation_results = pool.map(self.run_simulation, simulation_inputs)
        self.store_simulation_results(simulation_results, results_dir)

    @abstractmethod
    def run_simulation(self, simulation_input):
        pass

    @abstractmethod
    def store_simulation_results(self, simulation_results, results_dir):
        pass

class SimulationInput:
    def __init__(self, star_mass, mass_arr, planet_periods, ecc_arr):
        self.star_mass = star_mass
        self.mass_arr = mass_arr
        self.planet_periods = planet_periods
        self.ecc_arr = ecc_arr