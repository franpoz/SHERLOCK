import json
import logging
import multiprocessing
import os

import imageio
import matplotlib.pyplot as plt
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
                planet_param.mass = StabilityCalculator.mass_from_radius(planet_param.radius)
                planet_param.mass_low_err = (planet_param.mass - StabilityCalculator.mass_from_radius(planet_param.radius - planet_param.radius_low_err)) * 2
                planet_param.mass_up_err = (StabilityCalculator.mass_from_radius(planet_param.radius + planet_param.radius_up_err) - planet_param.mass) * 2
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
            inc = np.deg2rad(simulation_input.inc_arr[planet_key])
            omega = np.deg2rad(simulation_input.omega_arr[planet_key])
            sim.add(m=mass * self.EARTH_TO_SUN_MASS / simulation_input.star_mass, P=period, e=ecc, omega=omega, inc=inc)
        # sim.status()
        sim.move_to_com()
        return sim

    def run(self, results_dir, star_mass_low, star_mass_up, star_mass_bins, planet_params,
            cpus=multiprocessing.cpu_count() - 1, free_params=None):
        """
        Creates possible scenarios of stellar masses, planet masses and planet eccentricities. Afterwards a stability
        analysis is run for each of the scenarios and the results are stored in a file.
        @param results_dir: the directory where the results will be written into
        @param star_mass_low: the lowest star mass
        @param star_mass_up: the highest star mass
        @param star_mass_bins: the number of star masses to sample
        @param planet_params: the planet inputs containing the planets parameters
        @param cpus: the number of cpus to be used
        @param free_params: the parameters to be sampled entirely
        """
        if free_params is None:
            free_params = []
        planet_params = StabilityCalculator.prepare_planet_params(planet_params)
        star_masses = StabilityCalculator.prepare_star_masses(star_mass_low, star_mass_up, star_mass_bins)
        planet_masses = []
        planet_period = []
        planet_ecc = []
        planet_inc = []
        planet_omega = []
        for planet_param in planet_params:
            if planet_param.period_bins == 1 or planet_param.period_low_err == planet_param.period_up_err == 0:
                period_grid = np.full(1, planet_param.period)
            else:
                period_grid = np.linspace(planet_param.period - planet_param.period_low_err,
                                        planet_param.period + planet_param.period_up_err,
                                        planet_param.period_bins)
            planet_period.append(period_grid)
            if planet_param.mass_bins == 1 or planet_param.mass_low_err == planet_param.mass_up_err == 0:
                mass_grid = np.full(1, planet_param.mass)
            else:
                mass_grid = np.linspace(planet_param.mass - planet_param.mass_low_err,
                                        planet_param.mass + planet_param.mass_up_err,
                                        planet_param.mass_bins)
            planet_masses.append(mass_grid)
            if "eccentricity" in free_params:
                ecc_grid = np.linspace(0, 0.5, planet_param.ecc_bins)
            elif planet_param.ecc_bins == 1 or planet_param.ecc_low_err == planet_param.ecc_up_err == 0:
                ecc_grid = np.full(1, planet_param.eccentricity)
            else:
                low_ecc = planet_param.eccentricity - planet_param.ecc_low_err
                low_ecc = low_ecc if low_ecc > 0 else 0
                up_ecc = planet_param.eccentricity + planet_param.ecc_up_err
                up_ecc = up_ecc if up_ecc < 1 else 1
                ecc_grid = np.linspace(low_ecc, up_ecc, planet_param.ecc_bins)
            planet_ecc.append(ecc_grid)
            if planet_param.inc_bins == 1 or planet_param.inc_low_err == planet_param.inc_up_err == 0:
                inc_grid = np.full(1, planet_param.inclination)
            else:
                inc_grid = np.linspace(planet_param.inclination - planet_param.inc_low_err,
                                       planet_param.inclination + planet_param.inc_up_err,
                                       planet_param.inc_bins)
            planet_inc.append(inc_grid)
            if "omega" in free_params:
                # using arange instead of linspace because 0 and 360 are the same, so we exclude 360
                omega_grid = np.arange(0, 360, 360 // planet_param.omega_bins)
            elif planet_param.omega_bins == 1 or planet_param.omega_low_err == planet_param.omega_up_err == 0:
                omega_grid = np.full(1, planet_param.omega)
            else:
                omega_grid = np.linspace(planet_param.omega - planet_param.omega_low_err,
                                         planet_param.omega + planet_param.omega_up_err,
                                         planet_param.omega_bins)
            planet_omega.append(omega_grid)
        period_grid = np.array(np.meshgrid(*np.array(planet_period))).T.reshape(-1, len(planet_period))
        masses_grid = np.array(np.meshgrid(*np.array(planet_masses))).T.reshape(-1, len(planet_masses))
        ecc_grid = np.array(np.meshgrid(*np.array(planet_ecc))).T.reshape(-1, len(planet_ecc))
        inc_grid = np.array(np.meshgrid(*np.array(planet_inc))).T.reshape(-1, len(planet_inc))
        omega_grid = np.array(np.meshgrid(*np.array(planet_omega))).T.reshape(-1, len(planet_omega))
        simulation_inputs = []
        i = 0
        star_masses_scenario_num = len(star_masses)
        masses_scenario_num = len(masses_grid)
        period_scenario_num = len(period_grid)
        inc_scenario_num = len(inc_grid)
        ecc_scenario_num = len(ecc_grid)
        omega_scenario_num = len(omega_grid)
        scenarios_num = star_masses_scenario_num * masses_scenario_num * period_scenario_num * inc_scenario_num * \
                        ecc_scenario_num * omega_scenario_num
        logging.info("Preparing system values for all scenarios")
        logging.info("%.0f star mass scenarios.", len(star_masses))
        logging.info("%.0f bodies mass scenarios.", len(masses_grid))
        logging.info("%.0f period scenarios.", len(period_grid))
        logging.info("%.0f inclination scenarios.", len(inc_grid))
        logging.info("%.0f eccentricity scenarios.", len(ecc_grid))
        logging.info("%.0f arg of periastron scenarios.", len(omega_grid))
        logging.info("%.0f x %.0f x %.0f x %.0f x %.0f x %.0f = %.0f total scenarios.", star_masses_scenario_num,
                     masses_scenario_num, period_scenario_num, inc_scenario_num, ecc_scenario_num, omega_scenario_num,
                     scenarios_num)
        for star_mass in star_masses:
            for period_key, period_arr in enumerate(period_grid):
                for mass_key, mass_arr in enumerate(masses_grid):
                    for inc_key, inc_arr in enumerate(inc_grid):
                        for ecc_key, ecc_arr in enumerate(ecc_grid):
                            for omega_key, omega_arr in enumerate(omega_grid):
                                simulation_inputs.append(SimulationInput(star_mass, mass_arr, period_arr, ecc_arr, inc_arr, omega_arr, i))
                                i = i + 1
        logging.info("Finished preparing scenarios")
        with Pool(processes=cpus) as pool:
            simulation_results = pool.map(self.log_and_run_simulation, simulation_inputs)
        self.store_simulation_results(simulation_results, results_dir)

    def log_and_run_simulation(self, simulation_input):
        logging.info("Running scenario number %.0f: %s", simulation_input.index, json.dumps(simulation_input.__dict__,
                                                                                            cls=NumpyEncoder))
        return self.run_simulation(simulation_input)

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

    def plot_simulation(self, simulation_input_df, save_dir, scenario_name, xlim=None, ylim=None):
        eccs = simulation_input_df["eccentricities"].split(",")
        eccs = [float(i) for i in eccs]
        incs = simulation_input_df["inclinations"].split(",")
        incs = [float(i) - 90 for i in incs]
        omegas = simulation_input_df["arg_periastron"].split(",")
        omegas = [float(i) for i in omegas]
        periods = simulation_input_df["periods"].split(",")
        periods = [float(i) for i in periods]
        masses = simulation_input_df["masses"].split(",")
        masses = [float(i) for i in masses]
        star_mass = simulation_input_df["star_mass"]
        sim = self.init_rebound_simulation(SimulationInput(star_mass, masses, periods, eccs, incs, omegas, 1))
        filenames = []
        for i in range(1):
            sim.integrate(sim.t + i / 100 * 2 * np.pi)
            fig, ax = rebound.OrbitPlot(sim, color=True, unitlabel="[AU]", xlim=xlim, ylim=ylim)
            filename = save_dir + "scenario_" + scenario_name + "_" + str(i) + ".png"
            plt.savefig(filename)
            filenames.append(filename)
            plt.close(fig)
        # with imageio.get_writer(save_dir + scenario_name + '.gif', mode='I') as writer:
        #     for filename in filenames:
        #         image = imageio.imread(filename)
        #         writer.append_data(image)
        # # Remove files
        # for filename in set(filenames):
        #     os.remove(filename)


class SimulationInput:
    """
    Used as input for the simulations done for each scenario
    """
    def __init__(self, star_mass, mass_arr, planet_periods, ecc_arr, inc_arr, omega_arr, index):
        self.star_mass = star_mass
        self.mass_arr = mass_arr
        self.planet_periods = planet_periods
        self.ecc_arr = ecc_arr
        self.inc_arr = inc_arr
        self.omega_arr = omega_arr
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
