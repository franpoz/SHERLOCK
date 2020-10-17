class TransitResult:
    def __init__(self, results, period, per_err, duration,
                 t0, depths, depth, count, snr, sde, fap, border_score, in_transit, harmonic=None):
        self.results = results
        self.period = period
        self.per_err = per_err
        self.duration = duration
        self.t0 = t0
        self.depths = depths
        self.depth = depth
        self.count = count
        self.snr = snr
        self.sde = sde
        self.fap = fap
        self.border_score = border_score
        self.in_transit = in_transit
        self.harmonic = harmonic
