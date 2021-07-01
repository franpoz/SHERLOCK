import json
import logging
import multiprocessing
from abc import ABC, abstractmethod
from multiprocessing import Pool

import numpy as np
import rebound


class StabilityCalculator(ABC):
    """
    Template class for system stability calculation algorithms
    """
    EARTH_TO_SUN_MASS = 0.000003003

    def __init__(self):
        pass

    @staticmethod
    def mass_from_radius(radius):
        """
        Computation of mass-radius relationship from
        Bashi D., Helled R., Zucker S., Mordasini C., 2017, A&A, 604, A83. doi:10.1051/0004-6361/201629922
        @param radius: the radius value in earth radius
        @return: the mass in earth masses
        """
        return radius ** (1 / 0.55) if radius <= 12.1 else radius ** (1 / 0.01)

    @staticmethod
    def prepare_star_masses(star_mass_low, star_mass_up, star_mass_bins):
        """
        Creates a star masses grid
        @param star_mass_low: the lowest star mass value
        @param star_mass_up: the highest star mass value
        @param star_mass_bins: the number of star masses to sample. It will be ignored if star_mass_low == star_mass_up.
        @return: the star masses grid
        """
        return np.linspace(star_mass_low, star_mass_up, star_mass_bins) if star_mass_low != star_mass_up \
            else np.linspace(star_mass_low, star_mass_up, 1)

    @staticmethod
    def prepare_planet_params(planet_params):
        """
        Fills the planet masses if missing
        @param planet_params: the planet inputs
        @return: the planet inputs with the filled masses
        """
        for planet_param in planet_params:
            if planet_param.radius is None and (planet_param.mass_low is None or planet_param.mass_up is None):
                raise ValueError("There is one body without either radius or mass information: " +
                                 json.dumps(planet_param.__dict__))
            if planet_param.radius is not None:
                guessed_mass = StabilityCalculator.mass_from_radius(planet_param.radius)
                guessed_mass_low = guessed_mass - 0.2 if guessed_mass - 0.2 > 0 else guessed_mass
                planet_param.mass_low = planet_param.mass_low if planet_param.mass_low is not None else guessed_mass_low
                planet_param.mass_up = planet_param.mass_up if planet_param.mass_up is not None else guessed_mass + 0.2
        return planet_params

    def init_rebound_simulation(self, simulation_input):
        """
        Initializes the simulation for rebound-based algorithms
        @param simulation_input: the input data for the simulation scenario
        @return: the rebound initialized simulation scenario
        """
        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0
        sim.dt = 1e-2
        sim.add(m=simulation_input.star_mass)
        for planet_key, mass in enumerate(simulation_input.mass_arr):
            period = simulation_input.planet_periods[planet_key]
            ecc = simulation_input.ecc_arr[planet_key]
            sim.add(m=mass * self.EARTH_TO_SUN_MASS / simulation_input.star_mass, P=period, e=ecc, omega=0)
        # sim.status()
        sim.move_to_com()
        return sim

    def run(self, results_dir, star_mass_low, star_mass_up, star_mass_bins, planet_params, cpus=multiprocessing.cpu_count() - 1):
        """
        Creates possible scenarios of stellar masses, planet masses and planet eccentricities. Afterwards a stability
        analysis is run for each of the scenarios and the results are stored in a file.
        @param results_dir: the directory where the results will be written into
        @param star_mass_low: the lowest star mass
        @param star_mass_up: the highest star mass
        @param star_mass_bins: the number of star masses to sample
        @param planet_params: the planet inputs containing the planets parameters
        @param cpus: the number of cpus to be used
        """
        planet_params = StabilityCalculator.prepare_planet_params(planet_params)
        star_masses = StabilityCalculator.prepare_star_masses(star_mass_low, star_mass_up, star_mass_bins)
        planet_masses = []
        planet_periods = []
        planet_ecc = []
        for planet_param in planet_params:
            if planet_param.mass_bins == 1:
                mass_grid = np.full(1, (planet_param.mass_low + planet_param.mass_up) / 2)
            elif planet_param.mass_low == planet_param.mass_up:
                mass_grid = np.full(1, planet_param.mass_low)
            else:
                mass_grid = np.linspace(planet_param.mass_low, planet_param.mass_up, planet_param.mass_bins)
            planet_masses.append(mass_grid)
            if planet_param.ecc_bins == 1:
                ecc_grid = np.full(1, (planet_param.eccentricity_low + planet_param.eccentricity_up) / 2)
            elif planet_param.eccentricity_low == planet_param.eccentricity_up:
                ecc_grid = np.full(1, planet_param.eccentricity_low)
            else:
                ecc_grid = np.linspace(planet_param.eccentricity_low, planet_param.eccentricity_up,
                                        planet_param.ecc_bins)
            planet_ecc.append(ecc_grid)
            planet_periods.append(planet_param.period)
        masses_grid = np.array(np.meshgrid(*np.array(planet_masses))).T.reshape(-1, len(planet_masses))
        ecc_grid = np.array(np.meshgrid(*np.array(planet_ecc))).T.reshape(-1, len(planet_ecc))
        simulation_inputs = []
        i = 0
        for star_mass in star_masses:
            for mass_key, mass_arr in enumerate(masses_grid):
                for ecc_key, ecc_arr in enumerate(ecc_grid):
                    simulation_inputs.append(SimulationInput(star_mass, mass_arr, planet_periods, ecc_arr, i))
                    i = i + 1
        logging.info("%.0f star mass scenarios.", len(star_masses))
        logging.info("%.0f bodies mass scenarios.", len(masses_grid))
        logging.info("%.0f eccentricity scenarios.", len(ecc_grid))
        logging.info("%.0f x %.0f x %.0f = %.0f total scenarios to be computed.", len(star_masses), len(masses_grid),
                     len(ecc_grid), len(simulation_inputs))
        with Pool(processes=cpus) as pool:
            simulation_results = pool.map(self.log_and_run_simulation, simulation_inputs)
        self.store_simulation_results(simulation_results, results_dir)

    def log_and_run_simulation(self, simulation_input):
        logging.info("Running scenario number %.0f: %s", simulation_input.index, json.dumps(simulation_input.__dict__,
                                                                                            cls=NumpyEncoder))
        self.run_simulation(simulation_input)

    @abstractmethod
    def run_simulation(self, simulation_input):
        """
        Runs one stability scenario
        @param simulation_input: the simulation scenario parameters
        """
        pass

    @abstractmethod
    def store_simulation_results(self, simulation_results, results_dir):
        """
        Writes into disk all the final simulation results
        @param simulation_results: the results of the simulation for all the scenarios
        @param results_dir: the output directory where results will be written into
        """
        pass


class SimulationInput:
    """
    Used as input for the simulations done for each scenario
    """
    def __init__(self, star_mass, mass_arr, planet_periods, ecc_arr, index):
        self.star_mass = star_mass
        self.mass_arr = mass_arr
        self.planet_periods = planet_periods
        self.ecc_arr = ecc_arr
        self.index = index

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types. Got from https://stackoverflow.com/a/49677241/4198726"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
