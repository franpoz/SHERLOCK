import multiprocessing
from multiprocessing import Pool

import rebound
import numpy as np
from rebound import InterruptiblePool

from sherlockpipe.nbodies.stability_calculator import StabilityCalculator
import pandas as pd


class MegnoStabilityCalculator(StabilityCalculator):
    def run_simulation(self, simulation_input):
        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0
        sim.dt = 1e-2
        sim.add(m=simulation_input.star_mass)
        for planet_key, mass in enumerate(simulation_input.mass_arr):
            period = simulation_input.planet_periods[planet_key]
            ecc = simulation_input.ecc_arr[planet_key]
            sim.add(m=mass * 0.000003003 / simulation_input.star_mass, P=period, e=ecc, omega=0)
        # sim.status()
        sim.move_to_com()
        sim.init_megno()
        sim.exit_max_distance = 20.
        megno = 10
        try:
            sim.integrate(5e2 * 2. * np.pi, exact_finish_time=0)  # integrate for 500 years, integrating to the nearest
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
            megno = 10  # At least one particle got ejected, returning large MEGNO
        return {"star_mass": simulation_input.star_mass,
                "periods": ",".join([str(planet_period) for planet_period in simulation_input.planet_periods]),
                "masses": ",".join([str(mass_value) for mass_value in simulation_input.mass_arr]),
                "eccentricities": ",".join([str(ecc_value) for ecc_value in simulation_input.ecc_arr]),
                "megno": megno}

    def store_simulation_results(self, simulation_results, results_dir):
        result_file = results_dir + "/stability_megno.csv"
        results_df = pd.DataFrame(columns=['star_mass', 'periods', 'masses', 'eccentricities', 'megno'])
        results_df = results_df.append(simulation_results, ignore_index=True)
        results_df = results_df.sort_values('megno')
        results_df.to_csv(result_file, index=False)


# parameters.append(PlanetInput(5.43440, 1.68792, 0))
# parameters.append(PlanetInput(1.74542, 1.12207, 0))
# parameters.append(PlanetInput(4.02382, 1.34990, 0))
# parameters.append(PlanetInput(2.8611, 1.17643, 0))
# parameters.append(PlanetInput(1.58834, 1.07459, 0))
# result = StabilityCalculator(0.211299).run(parameters)
# print("MEGNO: " + str(result))

# pool = InterruptiblePool()
# results = pool.map(StabilityCalculator(0.211299).run, parameters)
# results2d = np.array(results).reshape(grid, grid)
# fig = plt.figure(figsize=(7, 5))
# ax = plt.subplot(111)
# extent = [min(par_e), max(par_e), min(par_e1), max(par_e1)]
# ax.set_xlim(extent[0], extent[1])
# ax.set_xlabel("ecc1 $e$")
# ax.set_ylim(extent[2], extent[3])
# ax.set_ylabel("ecc2 $e1$")
# im = ax.imshow(results2d, interpolation="none", vmin=1.9, vmax=10, cmap="RdYlGn_r", origin="lower", aspect='auto', extent=extent)
# cb = plt.colorbar(im, ax=ax)
# cb.set_label("MEGNO $\\langle Y \\rangle$")
# plt.show()
