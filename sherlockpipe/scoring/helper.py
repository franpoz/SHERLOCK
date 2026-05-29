import numpy as np
from scipy.ndimage.interpolation import shift


def compute_border_score(time, result, intransit, cadence):
    """Compute a border score penalizing transits that fall near observation edges.

    The score ranges from 0 (all transits near edges) to 1 (no transits near edges).

    Parameters
    ----------
    time : ndarray
        The time array of the light curve.
    result : TransitResult
        The transit result containing transit depths.
    intransit : ndarray
        Boolean array indicating in-transit points.
    cadence : float
        The cadence in seconds.

    Returns
    -------
    float
        The border score between 0 and 1, where 1 means no transits near edges.
    """
    shift_cadences = 3600 / cadence
    edge_limit_days = 0.05
    transit_depths = np.nan_to_num(result.transit_depths)
    transit_depths = np.zeros(1) if type(transit_depths) is not np.ndarray else transit_depths
    transit_depths = transit_depths[transit_depths > 0] if len(transit_depths) > 0 else []
    # a=a[np.where([i for i, j in groupby(intransit)])]
    border_score = 0
    if len(transit_depths) > 0:
        shifted_transit_points = shift(intransit, shift_cadences, cval=np.nan)
        inverse_shifted_transit_points = shift(intransit, -shift_cadences, cval=np.nan)
        intransit_shifted = intransit | shifted_transit_points | inverse_shifted_transit_points
        time_edge_indexes = np.where(abs(time[:-1] - time[1:]) > edge_limit_days)[0]
        time_edge = np.full(len(time), False)
        time_edge[time_edge_indexes] = True
        time_edge[0] = True
        time_edge[len(time_edge) - 1] = True
        transits_in_edge = intransit_shifted & time_edge
        transits_in_edge_count = len(transits_in_edge[transits_in_edge])
        border_score = 1 - transits_in_edge_count / len(transit_depths)
    return border_score if border_score >= 0 else 0

def find_nearest(array, value):
    """Find the index and value nearest to a given value in an array.

    Parameters
    ----------
    array : ndarray
        The array to search in.
    value : float
        The target value to find the nearest neighbor for.

    Returns
    -------
    tuple
        A tuple of (index, nearest_value) where index is the position in the
        array and nearest_value is the closest array element to the input value.
    """
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def harmonic_spectrum(periods, spectrum):
    """Compute a harmonic spectrum by averaging SDE at harmonic periods.

    For each period in the input grid, the average SDE at harmonic ratios
    (1/4, 1/3, 1/2, 1, 2, 3, 4) is computed to detect harmonic patterns.

    Parameters
    ----------
    periods : ndarray
        The array of periods from the search grid.
    spectrum : ndarray
        The SDE/power spectrum values corresponding to each period.

    Returns
    -------
    ndarray
        The harmonic spectrum with the same length as the input periods.
    """
    harmonics = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]
    harmonic_spectrum = np.zeros(len(periods))
    for index, period in enumerate(periods):
        harmonic_sde = 0
        non_nan_harmonics_count = 0
        for harmonic in harmonics:
            expected_harmonic_period = period * harmonic
            harmonic_index, harmonic_period = find_nearest(periods, expected_harmonic_period)
            harmonic_index = harmonic_index if np.abs(1 - harmonic_period / expected_harmonic_period) < 0.01 else np.nan
            if not np.isnan(harmonic_index):
                harmonic_sde = harmonic_sde + spectrum[harmonic_index]
                non_nan_harmonics_count = non_nan_harmonics_count + 1
        harmonic_spectrum[index] = harmonic_sde / non_nan_harmonics_count if non_nan_harmonics_count > 0 else 0
    return harmonic_spectrum
