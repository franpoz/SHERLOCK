from typing import Dict, List

import rebound
import numpy as np
from sherlockpipe.system_stability.stability_calculator import StabilityCalculator, SimulationInput
import pandas as pd


class MegnoStabilityCalculator(StabilityCalculator):

    """
    Runs the stability computations by calculating the MEGNO score.
    """
    def __init__(self, years, dt, repetitions):
        super().__init__(dt)
        self.years = years
        self.repetitions = repetitions

    def run_simulation(self, simulation_input: SimulationInput) -> dict:
        """
        Runs one simulation scenario using the megno module of rebound.

        :param SimulationInput simulation_input: the scenario parameters
        :return dict: the results of the simulation with the rebound-specific megno metric.
        """
        final_megno = []
        final_year = []
        for i in range(self.repetitions):
            sim = self.init_rebound_simulation(simulation_input)
            sim.init_megno()
            sim.exit_max_distance = 20.
            for year in np.logspace(0, np.log10(self.years), int(np.log10(self.years / sim.dt))):
                try:
                    sim.integrate(year, exact_finish_time=0)
                    # import matplotlib.pyplot as plt
                    # sim.integrate(1, exact_finish_time=0)
                    # op = rebound.OrbitPlot(sim, color=True)
                    # plt.show()

                    # for i in range(500):
                    #     sim.integrate(sim.t + i * 2 * np.pi)
                    #     fig, ax = rebound.OrbitPlot(sim, color=True, unitlabel="[AU]", xlim=[-0.1, 0.1], ylim=[-0.1, 0.1])
                    #     plt.show()
                    #     plt.close(fig)
                    # clear_output(wait=True)
                    # timestep for each output to keep the timestep constant and preserve WHFast's symplectic nature
                    megno = sim.megno()
                    megno = megno if megno < 5 else 5
                except rebound.Escape:
                    megno = 5
                except rebound.Encounter:
                    megno = 5
                if megno == 5:
                    break
            final_megno = np.append(final_megno, megno)
            final_year = np.append(final_year, year)
        return {"star_mass": simulation_input.star_mass,
                "periods": ",".join([str(planet_period) for planet_period in simulation_input.planet_periods]),
                "masses": ",".join([str(mass_value) for mass_value in simulation_input.mass_arr]),
                "inclinations": ",".join([str(inc_value) for inc_value in simulation_input.inc_arr]),
                "eccentricities": ",".join([str(ecc_value) for ecc_value in simulation_input.ecc_arr]),
                "arg_periastron": ",".join([str(om_value) for om_value in simulation_input.omega_arr]),
                "long_asc_node": ",".join([str(om_big_value) for om_big_value in simulation_input.omega_big_arr]),
                "last_year": np.nanmedian(final_year),
                "megno": np.nanmedian(final_megno)}

    def store_simulation_results(self, simulation_results: List[Dict], results_dir: str):
        """
        Stores the megno results in the given directory

        :param List[Dict] simulation_results: the list of results dictionaries
        :param str results_dir: directory to store the file
        """
        result_file = results_dir + "/stability_megno.csv"
        results_df = pd.DataFrame(columns=['star_mass', 'periods', 'masses', 'inclinations', 'eccentricities',
                                           'arg_periastron', 'long_asc_node', 'last_year', 'megno'])
        results_df = pd.concat([results_df, pd.DataFrame(simulation_results)], ignore_index=True)
        results_df = results_df.sort_values('megno')
        results_df.to_csv(result_file, index=False)
