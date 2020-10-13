import wotan
from astropy.stats import sigma_clip
from scipy import stats


class Flattener:
    def __init__(self) -> None:
        super().__init__()

    def flatten_bw(self, flatten_input):
        flatten_lc, trend = wotan.flatten(flatten_input.time, flatten_input.flux, window_length=flatten_input.wl,
                                    return_trend=True, method="biweight", break_tolerance=0.5)
        flatten_lc = sigma_clip(flatten_lc, sigma_lower=20, sigma_upper=3)
        bin_centers_i, bin_means_i, bin_width_i, bin_edges_i, bin_stds_i = \
            self.__compute_flatten_stats(flatten_input.time, flatten_lc, flatten_input.bin_minutes)
        return flatten_lc, trend, bin_centers_i, bin_means_i, flatten_input.wl

    def flatten_gp(self, flatten_input):
        flatten_lc, trend = wotan.flatten(flatten_input.time, flatten_input.flux, method="gp", kernel='matern',
                                               kernel_size=flatten_input.wl, return_trend=True, break_tolerance=0.5)
        flatten_lc = sigma_clip(flatten_lc, sigma_lower=20, sigma_upper=3)
        bin_centers_i, bin_means_i, bin_width_i, bin_edges_i, bin_stds_i = \
            self.__compute_flatten_stats(flatten_input.time, flatten_lc, flatten_input.bin_minutes)
        return flatten_lc, trend, bin_centers_i, bin_means_i, flatten_input.wl

    def __compute_flatten_stats(self, time, flux, bin_minutes):
        bins_i = len(time) * 2 / bin_minutes
        bin_means_i, bin_edges_i, binnumber_i = stats.binned_statistic(time, flux, statistic='mean', bins=bins_i)
        bin_stds_i, _, _ = stats.binned_statistic(time, flux, statistic='std', bins=bins_i)
        bin_width_i = (bin_edges_i[1] - bin_edges_i[0])
        bin_centers_i = bin_edges_i[1:] - bin_width_i / 2
        return bin_centers_i, bin_means_i, bin_width_i, bin_edges_i, bin_stds_i

class FlattenInput:
    def __init__(self, time, flux, wl, bin_minutes):
        self.time = time
        self.flux = flux
        self.wl = wl
        self.bin_minutes = bin_minutes