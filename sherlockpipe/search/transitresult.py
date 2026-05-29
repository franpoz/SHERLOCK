class TransitResult:
    """
    Attributes-only class to store the results of transit search from :class:`sherlockpipe.sherlock.Sherlock`
    """
    def __init__(self, power_args, results, period, per_err, duration,
                 t0, t0s, depths, depths_err, depth, depth_err, odd_even_mismatch, depth_mean_even, depth_mean_odd,
                 count, snr, sde, fap, border_score, in_transit, harmonic=None, harmonic_spectrum=None, mode='tls'):
        """
        Parameters
        ----------
        power_args : dict
            The arguments passed to the TLS/BLS power search.
        results : object
            The raw results object from the search engine.
        period : float
            The detected orbital period in days.
        per_err : float
            The period uncertainty in days.
        duration : float
            The transit duration in days.
        t0 : float
            The transit epoch (time of conjunction) in days.
        t0s : numpy.ndarray
            Array of individual transit times.
        depths : numpy.ndarray
            Array of transit depths in ppt for each transit.
        depths_err : numpy.ndarray
            Array of transit depth uncertainties in ppt.
        depth : float
            The mean transit depth in ppt.
        depth_err : float
            The uncertainty on the mean depth.
        odd_even_mismatch : float
            Odd-even transit depth mismatch significance.
        depth_mean_even : float
            Mean depth of even-numbered transits in ppt.
        depth_mean_odd : float
            Mean depth of odd-numbered transits in ppt.
        count : int
            Number of distinct transits detected.
        snr : float
            Signal-to-noise ratio of the detection.
        sde : float
            Signal Detection Efficiency.
        fap : float
            False-alarm probability.
        border_score : float
            Score measuring proximity to data borders.
        in_transit : numpy.ndarray
            Boolean array indicating in-transit data points.
        harmonic : str or None
            Harmonic relationship string, if detected.
        harmonic_spectrum : numpy.ndarray or None
            The harmonic spectrum array.
        mode : str
            The search mode ('tls' or 'bls').
        """
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
        self.mode = mode
