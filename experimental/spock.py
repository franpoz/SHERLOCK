import numpy as np
from spock import FeatureClassifier
import rebound
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
            sim.add(m=self.mass_from_radius(planet_param.r) * 0.000003003 / self.star_mass, P=planet_param.P,
                    e=planet_param.e, omega=planet_param.omega)
        #sim.status()
        sim.move_to_com()
        model = FeatureClassifier()
        print("SPOCK=" + str(model.predict_stable(sim)))

grid = 5
sc = StabilityCalculator(0.53)
par_e = np.linspace(0.0, 0.7, grid)
par_e1 = np.linspace(0.0, 0.7, grid)
for i in par_e:
    for j in par_e1:
        sc.run([PlanetInput(1.17, 0.01749, 11.76943, i), PlanetInput(1.37, 0.03088, 2.97, j), PlanetInput(2.45, 0, 3.9, 0)])

