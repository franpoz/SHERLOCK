class TransitResult:
    def __init__(self, results, period, per_err, duration,
                 t0, t0s, depths, depths_err, depth, odd_even_mismatch, depth_mean_even, depth_mean_odd, count, snr,
                 sde, fap, border_score, in_transit, harmonic=None):
        self.results = results
        self.period = period
        self.per_err = per_err
        self.duration = duration
        self.t0 = t0
        self.t0s = t0s
        self.depths = depths
        self.depths_err = depths_err
        self.depth = depth
        self.odd_even_mismatch = odd_even_mismatch
        self.depth_mean_even = depth_mean_even
        self.depth_mean_odd = depth_mean_odd
        self.count = count
        self.snr = snr
        self.sde = sde
        self.fap = fap
        self.border_score = border_score
        self.in_transit = in_transit
        self.harmonic = harmonic
