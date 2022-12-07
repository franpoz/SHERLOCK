import numpy as np
from scipy.ndimage.interpolation import shift


def compute_border_score(time, result, intransit, cadence):
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
