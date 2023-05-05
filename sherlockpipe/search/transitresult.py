class TransitResult:
    """
    Attributes-only class to store the results of transit search from :class:`sherlockpipe.sherlock.Sherlock`
    """
    def __init__(self, power_args, results, period, per_err, duration,
                 t0, t0s, depths, depths_err, depth, depth_err, odd_even_mismatch, depth_mean_even, depth_mean_odd,
                 count, snr, sde, fap, border_score, in_transit, harmonic=None, harmonic_spectrum=None):
        self.power_args = power_args
        self.results = results
        self.period = period
        self.per_err = per_err
        self.duration = duration
        self.t0 = t0
        self.t0s = t0s
        self.depths = depths
        self.depths_err = depths_err
        self.depth = depth
        self.depth_err = depth_err
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
        self.harmonic_spectrum = harmonic_spectrum
