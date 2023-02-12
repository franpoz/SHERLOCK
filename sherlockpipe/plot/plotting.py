import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sherlockpipe.search.transitresult import TransitResult


def save_transit_plot(object_id: str, title: str, plot_dir: str, file: str, time, lc, transit_result: TransitResult,
                      cadence, run_no: int, plot_harmonics: bool = False):
    """
    Stores the search results plot with: 1) The entire curve with the transit model 2)The folded curve and the transit
    model 3) The power spectrum of the TLS search 4) Only if the flag is enabled, the TLS search harmonics power
    spectrum.

    :param str object_id: the target id
    :param str title: title for the plot
    :param str plot_dir: directory where the plot should be stored
    :param str file: the file name of the plot
    :param time: the time array of the light curve
    :param lc: the flux value of the light curve
    :param TransitResult transit_result: the TransitResult object containing the search results
    :param cadence: the cadence of the curve in days
    :param int run_no: the SHERLOCK run of the results
    :param bool plot_harmonics: whether the harmonics power spectrum should be plotted
    """
    # start the plotting
    rows = 3 if not plot_harmonics else 4
    figsize = 10 if not plot_harmonics else 14
    fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(10, figsize), constrained_layout=True)
    fig.suptitle(title)
    # 1-Plot all the transits
    in_transit = transit_result.in_transit
    tls_results = transit_result.results
    axs[0].scatter(time[in_transit], lc[in_transit], color='red', s=2, zorder=0)
    axs[0].scatter(time[~in_transit], lc[~in_transit], color='black', alpha=0.05, s=2, zorder=0)
    axs[0].plot(tls_results.model_lightcurve_time, tls_results.model_lightcurve_model, alpha=1, color='red', zorder=1)
    # plt.scatter(time_n, flux_new_n, color='orange', alpha=0.3, s=20, zorder=3)
    plt.xlim(time.min(), time.max())
    # plt.xlim(1362.0,1364.0)
    axs[0].set(xlabel='Time (days)', ylabel='Relative flux')
    # phase folded plus binning
    bins_per_transit = 8
    half_duration_phase = transit_result.duration / 2 / transit_result.period
    if np.isnan(transit_result.period) or np.isnan(transit_result.duration):
        bins = 200
        folded_plot_range = 0.05
    else:
        bins = transit_result.period / transit_result.duration * bins_per_transit
        folded_plot_range = half_duration_phase * 10
    binning_enabled = True
    axs[1].plot(tls_results.model_folded_phase, tls_results.model_folded_model, color='red')
    scatter_measurements_alpha = 0.05 if binning_enabled else 0.8
    axs[1].scatter(tls_results.folded_phase, tls_results.folded_y, color='black', s=10,
                alpha=scatter_measurements_alpha, zorder=2)
    lower_x_limit = 0.5 - folded_plot_range
    upper_x_limit = 0.5 + folded_plot_range
    axs[1].set_xlim(lower_x_limit, upper_x_limit)
    axs[1].set(xlabel='Phase', ylabel='Relative flux')
    folded_phase_zoom_mask = np.argwhere((tls_results.folded_phase > lower_x_limit) &
                                         (tls_results.folded_phase < upper_x_limit)).flatten()
    if isinstance(tls_results.folded_phase, (list, np.ndarray)):
        folded_phase = tls_results.folded_phase[folded_phase_zoom_mask]
        folded_y = tls_results.folded_y[folded_phase_zoom_mask]
        axs[1].set_ylim(np.min([np.min(folded_y), np.min(tls_results.model_folded_model)]),
                     np.max([np.max(folded_y), np.max(tls_results.model_folded_model)]))
        plt.ticklabel_format(useOffset=False)
        bins = 80
        if binning_enabled and tls_results.SDE != 0 and bins < len(folded_phase):
            bin_means, bin_edges, binnumber = stats.binned_statistic(folded_phase, folded_y, statistic='mean',
                                                                     bins=bins)
            bin_stds, _, _ = stats.binned_statistic(folded_phase, folded_y, statistic='std', bins=bins)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            bin_size = int(folded_plot_range * 2 / bins * transit_result.period * 24 * 60)
            bin_means_data_mask = np.isnan(bin_means)
            axs[1].errorbar(bin_centers[~bin_means_data_mask], bin_means[~bin_means_data_mask],
                         yerr=bin_stds[~bin_means_data_mask] / 2, xerr=bin_width / 2, marker='o', markersize=4,
                         color='darkorange', alpha=1, linestyle='none', label='Bin size: ' + str(bin_size) + "m")
            axs[1].legend(loc="upper right")
    axs[2].axvline(transit_result.period, alpha=0.4, lw=3)
    axs[2].set_xlim(np.min(tls_results.periods), np.max(tls_results.periods))
    for n in range(2, 10):
        axs[2].axvline(n * tls_results.period, alpha=0.4, lw=1, linestyle="dashed")
        axs[2].axvline(tls_results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    axs[2].set(xlabel='Period (days)', ylabel='SDE')
    axs[2].plot(tls_results.periods, tls_results.power, color='black', lw=0.5)
    if plot_harmonics:
        max_harmonic_power_index = np.argmax(transit_result.harmonic_spectrum)
        harmonics = [1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]
        for harmonic in harmonics:
            axs[3].axvline(harmonic * tls_results.periods[max_harmonic_power_index], alpha=0.4, lw=1, linestyle="dashed")
        axs[3].axvline(tls_results.periods[max_harmonic_power_index], alpha=0.4, lw=3)
        axs[3].set_xlim(np.min(tls_results.periods), np.max(tls_results.periods))
        axs[3].set(xlabel='Period (days)', ylabel='Harmonics Power')
        axs[3].plot(tls_results.periods, transit_result.harmonic_spectrum, color='black', lw=0.5)
    plt.savefig(plot_dir + file, bbox_inches='tight', dpi=200)
    fig.clf()
    plt.close(fig)
