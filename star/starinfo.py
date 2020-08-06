class StarInfo:
    def __init__(self, object_id, ld_coefficients=None, teff=None, lum=None, logg=None, mass=None, mass_min=None,
                 mass_max=None, radius=None, radius_min=None, radius_max=None):
        self.object_id = object_id
        self.ld_coefficients = ld_coefficients
        self.teff = teff
        self.lum = lum
        self.logg = logg
        self.mass = mass
        self.radius = radius
        self.mass = mass
        self.mass_min = None if mass_min is None else mass - mass_min
        self.mass_max = None if mass_max is None else mass + mass_max
        self.radius = radius
        self.radius_min = None if radius_min is None else radius - radius_min
        self.radius_max = None if radius_max is None else radius + radius_max
        self.mass_assumed = False
        self.radius_assumed = False

    def assume_model_mass(self, mass=0.1):
        self.mass = mass
        self.mass_min = mass
        self.mass_max = mass
        self.mass_assumed = True

    def assume_model_radius(self, radius=0.1):
        self.radius = radius
        self.radius_min = radius
        self.radius_max = radius
        self.radius_assumed = True
