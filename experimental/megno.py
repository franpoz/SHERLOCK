import rebound
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from sherlockpipe.nbodies.PlanetInput import PlanetInput


class StabilityCalculator:
    def __init__(self, star_mass):
        self.star_mass = star_mass

    def mass_from_radius(self, radius):
        return radius ** (1 / 0.55) if radius <= 12.1 else radius ** (1 / 0.01)

    def run(self, planet_params):
        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0
        sim.dt = 1e-2
        sim.add(m=1.0)
        for planet_param in planet_params:
            sim.add(m=self.mass_from_radius(planet_param.r) * 0.000003003 / self.star_mass, P=planet_param.P, e=planet_param.e, omega=planet_param.omega)
        #sim.status()
        sim.move_to_com()
        sim.init_megno()
        sim.exit_max_distance = 20.
        try:
            sim.integrate(5e2 * 2. * np.pi, exact_finish_time=0) # integrate for 500 years, integrating to the nearest
            # for i in range(500):
            #     sim.integrate(sim.t + i * 2 * np.pi)
            #     fig, ax = rebound.OrbitPlot(sim, color=True, unitlabel="[AU]", xlim=[-0.1, 0.1], ylim=[-0.1, 0.1])
            #     plt.show()
            #     plt.close(fig)
                #clear_output(wait=True)
            #timestep for each output to keep the timestep constant and preserve WHFast's symplectic nature
            megno = sim.calculate_megno()
            megno = megno if megno < 10 else 10
            return megno
        except rebound.Escape:
            return 10. # At least one particle got ejected, returning large MEGNO

planet_params = []
parameters = []
# grid = 20
# par_e = np.linspace(0.0, 0.7, grid)
# par_e1 = np.linspace(0.0, 0.7, grid)
# for i in par_e:
#     for j in par_e1:
#         parameters.append((PlanetInput(1.74542, 0.01606, 1.12207, 0), PlanetInput(0.03088, 2.97, j)))
from rebound.interruptible_pool import InterruptiblePool


parameters.append(PlanetInput(5.43440, 1.68792, 0))
parameters.append(PlanetInput(1.74542, 1.12207, 0))
parameters.append(PlanetInput(4.02382, 1.34990, 0))
parameters.append(PlanetInput(2.8611, 1.17643, 0))
parameters.append(PlanetInput(1.58834, 1.07459, 0))
result = StabilityCalculator(0.211299).run(parameters)
print("MEGNO: " + str(result))

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
