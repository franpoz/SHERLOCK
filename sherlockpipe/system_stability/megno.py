from typing import Dict, List

import rebound
import numpy as np
from sherlockpipe.system_stability.stability_calculator import StabilityCalculator, SimulationInput
import pandas as pd


class MegnoStabilityCalculator(StabilityCalculator):

    """
    Runs the stability computations by calculating the MEGNO score.
    """
    def __init__(self, years):
        super().__init__()
        self.years = years

    def run_simulation(self, simulation_input: SimulationInput) -> dict:
        """
        Runs one simulation scenario using the megno module of rebound.

        :param SimulationInput simulation_input: the scenario parameters
        :return dict: the results of the simulation with the rebound-specific megno metric.
        """
        sim = self.init_rebound_simulation(simulation_input)
        sim.init_megno()
        sim.exit_max_distance = 20.
        try:
            sim.integrate(self.years * 2. * np.pi, exact_finish_time=0)  # integrate for 500 years, integrating to the nearest
            # for i in range(500):
            #     sim.integrate(sim.t + i * 2 * np.pi)
            #     fig, ax = rebound.OrbitPlot(sim, color=True, unitlabel="[AU]", xlim=[-0.1, 0.1], ylim=[-0.1, 0.1])
            #     plt.show()
            #     plt.close(fig)
            # clear_output(wait=True)
            # timestep for each output to keep the timestep constant and preserve WHFast's symplectic nature
            megno = sim.calculate_megno()
            megno = megno if megno < 10 else 10
        except rebound.Escape:
            megno = 10
        return {"star_mass": simulation_input.star_mass,
                "periods": ",".join([str(planet_period) for planet_period in simulation_input.planet_periods]),
                "masses": ",".join([str(mass_value) for mass_value in simulation_input.mass_arr]),
                "inclinations": ",".join([str(ecc_value) for ecc_value in simulation_input.inc_arr]),
                "eccentricities": ",".join([str(ecc_value) for ecc_value in simulation_input.ecc_arr]),
                "arg_periastron": ",".join([str(ecc_value) for ecc_value in simulation_input.omega_arr]),
                "megno": megno}

    def store_simulation_results(self, simulation_results: List[Dict], results_dir: str):
        """
        Stores the megno results in the given directory

        :param List[Dict] simulation_results: the list of results dictionaries
        :param str results_dir: directory to store the file
        """
        result_file = results_dir + "/stability_megno.csv"
        results_df = pd.DataFrame(columns=['star_mass', 'periods', 'masses', 'inclinations', 'eccentricities',
                                           'arg_periastron', 'megno'])
        results_df = results_df.append(simulation_results, ignore_index=True)
        results_df = results_df.sort_values('megno')
        results_df.to_csv(result_file, index=False)
