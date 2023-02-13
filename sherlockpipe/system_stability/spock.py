from typing import Dict, List

from spock import FeatureClassifier, DeepRegressor
from sherlockpipe.system_stability.stability_calculator import StabilityCalculator, SimulationInput
import pandas as pd


class SpockStabilityCalculator(StabilityCalculator):
    """
    Runs the stability computation by computing the stability probability and the median expected instability time for
    each scenario
    """
    def run_simulation(self, simulation_input: SimulationInput) -> dict:
        """
        Runs one stability scenario using Spock.

        :param SimulationInput simulation_input:
        :return dict: the result with the spock-specific metrics stability_probability and the median_expected_instability_time
        """
        sim = self.init_rebound_simulation(simulation_input)
        feature_classifier_model = FeatureClassifier()
        deep_regressor_model = DeepRegressor()
        stability_probability = feature_classifier_model.predict_stable(sim)
        median, lower, upper = deep_regressor_model.predict_instability_time(sim, samples=10000)
        return {"star_mass": simulation_input.star_mass,
                "periods": ",".join([str(planet_period) for planet_period in simulation_input.planet_periods]),
                "masses": ",".join([str(mass_value) for mass_value in simulation_input.mass_arr]),
                "inclinations": ",".join([str(ecc_value) for ecc_value in simulation_input.inc_arr]),
                "eccentricities": ",".join([str(ecc_value) for ecc_value in simulation_input.ecc_arr]),
                "arg_periastron": ",".join([str(ecc_value) for ecc_value in simulation_input.omega_arr]),
                "stability_probability": stability_probability, "median_expected_instability_time": median}

    def store_simulation_results(self, simulation_results: List[Dict], results_dir: str):
        """
        Stores the spock results in the given directory

        :param List[Dict] simulation_results: the list of results dictionaries
        :param str results_dir: directory to store the file
        """
        result_file = results_dir + "/stability_spock.csv"
        results_df = pd.DataFrame(columns=['star_mass', 'periods', 'masses', 'inclinations', 'eccentricities',
                                           'arg_periastron', 'stability_probability',
                                           'median_expected_instability_time'])
        results_df = results_df.append(simulation_results, ignore_index=True)
        results_df = results_df.sort_values('stability_probability', ascending=False)
        results_df.to_csv(result_file, index=False)
