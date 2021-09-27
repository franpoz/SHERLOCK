import logging

import foldedleastsquares
import lightkurve as lk
import numpy as np
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import pylab
import wotan
from astropy.stats import sigma_clip
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.ObjectInfo import ObjectInfo
from scipy import stats, signal
from scipy.signal import argrelextrema, savgol_filter
from wotan import flatten
from contextlib import contextmanager
from timeit import default_timer
from scipy.ndimage import uniform_filter1d
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

class SherlockExplorer:
    VERSION = 1
    OBJECT_ID_REGEX = "^(KIC|TIC|EPIC)[-_ ]([0-9]+)$"
    MISSION_ID_KEPLER = "KIC"
    MISSION_ID_KEPLER_2 = "EPIC"
    MISSION_ID_TESS = "TIC"

    def explore_object(self, object_id, input_lc_file=None, time_units=1, auto_detrend_periodic_signals=False,
                       auto_detrend_ratio=1/4, detrend_method="cosine", smooth=False, sectors=None, cadence=300):
        lc_builder = LcBuilder()
        object_info = lc_builder.build_object_info(object_id, None, sectors, input_lc_file, cadence, None,
                                                   [{"P": 0.59925, "D": 200, "T0": 1354.57}], None, None, None)
        lc, lc_data, star_info, transits_min_count, cadence, detrend_period, quarters = \
            lc_builder.build(object_info, ".")
        lc = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper=3)  # remove outliers over 3sigma
        flux = lc.flux.value
        flux_err = lc.flux_err.value
        if time_units == 0:
            time = lc.astropy_time.jd
        else:
            time = lc.time.value.astype('<f4')
        if smooth:
            flux = savgol_filter(flux, 11, 2)
            flux = uniform_filter1d(flux, 11)
        lc_df = pd.DataFrame(columns=['#time', 'flux', 'flux_err'])
        lc_df['#time'] = lc.time.value.astype('<f4')
        lc_df['flux'] = lc.flux.value.astype('<f4')
        lc_df['flux_err'] = lc.flux_err.value.astype('<f4')
        lc_df = pd.DataFrame(columns=['flux'])
        lc_df['flux'] = acf(lc.flux.value.astype('<f4'), nlags=7200)
        test_df = pd.DataFrame(lc_df)
        ax = test_df.plot()
        ax.set_ylim(lc_df["flux"].min(), lc_df["flux"].max())
        #ax.set_xticks(list(range(0, len(ax.get_xticks()))), ax.get_xticks() / 30 / 24)
        plt.show()
        #periodogram = lc.to_periodogram(oversample_factor=10)
        #fig = px.line(x=periodogram.period.astype('<f4'), y=periodogram.power.astype('<f4'), log_x=True)
        #fig.show()
        fig = px.scatter(x=lc.time.value.astype('<f4'), y=lc.flux.value.astype('<f4'))
        fig.show()
        if auto_detrend_periodic_signals:
            detrend_period = self.__calculate_max_significant_period(lc, periodogram)
            if detrend_period is not None:
                flatten_lc, trend_lc = self.__detrend_by_period(lc, detrend_period * auto_detrend_ratio, detrend_method)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=lc.time.value.astype('<f4'), y=flux.astype('<f4'), mode='markers', name='Flux'))
                fig.add_trace(go.Scatter(x=lc.time.value.astype('<f4'), y=trend_lc.astype('<f4'), mode='lines+markers',
                                         name='Main Trend'))
                fig.show()
                if auto_detrend_periodic_signals:
                    fig = px.line(x=lc.time.value.astype('<f4'), y=flatten_lc.astype('<f4'))
                    fig.show()
                    lc.flux = flatten_lc
        bin_means, bin_edges, binnumber = stats.binned_statistic(lc.time.value, lc.flux.value, statistic='mean',bins=len(time) / 5)
        bin_stds, _, _ = stats.binned_statistic(time, flux, statistic='std', bins=len(time) / 3)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2
        time_binned = bin_centers.astype('<f4')
        flux_binned = bin_means.astype('<f4')
        lc_binned = lk.LightCurve(time=time_binned, flux=flux_binned)
        fig = px.scatter(x=lc.time.value.astype('<f4'), y=lc.flux.value.astype('<f4'))
        fig.show()
        while True:
            try:
                user_input = input("Select the period to fold the light curve: ")
                if user_input.startswith("q"):
                    break
                period = float(user_input)
            except ValueError:
                print("Wrong number.")
                continue
            try:
                user_input = input("Select the t0 to fold the light curve: ")
                if user_input.startswith("q"):
                    break
                t0 = float(user_input)
            except ValueError:
                print("Wrong number.")
                continue
            # flux = lc.flux
            # args_flux_outer = np.argwhere(flux > 1 + depth * 0.001)
            # flux[args_flux_outer] = np.nan
            # time[args_flux_outer] = np.nan
            # args_flux_outer = np.argwhere(flux < 1 + depth * 0.001)
            # flux[args_flux_outer] = np.nan
            # time[args_flux_outer] = np.nan

            j = 0
            fig_transit, axs = plt.subplots(4, 4, figsize=(16, 16))
            #fig_transit.suptitle("Transit centered plots for TIC " + str(object_id))
            for i in np.arange(t0, t0 + period * 16, period)[0:16]:
                args_plot = np.argwhere((i - 0.1 < time) & (time < i + 0.1)).flatten()
                plot_time = time[args_plot]
                plot_flux = flux[args_plot]
                axs[j // 4][j % 4].scatter(plot_time, plot_flux, color='gray', alpha=1, rasterized=True, label="Flux Transit " + str(j))
                j = j + 1
            folded_lc = lk.LightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
            folded_lc.remove_nans()
            # folded_lc.flux_err = np.nan
            # folded_lc = folded_lc.bin(bins=500)
            folded_lc.fold(period=period).scatter()
            plt.title("Phase-folded period: " + format(period, ".2f") + " days")
            plt.show()
            fig_transit.show()
            fig = px.line(x=folded_lc.time.value.astype('<f4'), y=folded_lc.flux.value.astype('<f4'))
            fig.show()

    def __parse_object_id(self, object_id):
        object_id_parsed = re.search(self.OBJECT_ID_REGEX, object_id)
        mission_prefix = object_id[object_id_parsed.regs[1][0]:object_id_parsed.regs[1][1]]
        id = object_id[object_id_parsed.regs[2][0]:object_id_parsed.regs[2][1]]
        if mission_prefix == self.MISSION_ID_KEPLER:
            mission = "Kepler"
        elif mission_prefix == self.MISSION_ID_KEPLER_2:
            mission = "K2"
        elif mission_prefix == self.MISSION_ID_TESS:
            mission = "TESS"
        else:
            raise ValueError("Invalid object id " + object_id)
        return mission, mission_prefix, int(id)

    def __calculate_max_significant_period(self, lc, periodogram):
        max_accepted_period = (lc.time[len(lc.time) - 1] - lc.time[0]) / 4
        accepted_power_indexes = np.transpose(np.argwhere(periodogram.power > 0.0008))[0]
        period_values = [p.value for p in periodogram.period]
        accepted_period_indexes = np.transpose(np.argwhere(period_values < max_accepted_period))[0]
        accepted_indexes = [i for i in accepted_period_indexes if i in accepted_power_indexes]
        local_extrema = argrelextrema(periodogram.power[accepted_indexes], np.greater)[0]
        accepted_indexes = np.array(accepted_indexes)[local_extrema]
        period = None
        if len(accepted_indexes) > 0:
            relevant_periods = periodogram.period[accepted_indexes]
            period = min(relevant_periods)
            period = period.value
            logging.info("Auto-Detrend found the next important periods: " + str(relevant_periods) + ". Period used " +
                         "for auto-detrend will be " + str(period) + " days")
        else:
            logging.info("Auto-Detrend did not find relevant periods.")
        return period

    def __detrend_by_period(self, lc, period_window, detrend_method):
        if detrend_method == 'gp':
            flatten_lc, lc_trend = flatten(lc.time.value, lc.flux.value, method=detrend_method, kernel='matern',
                                   kernel_size=period_window, return_trend=True, break_tolerance=0.5)
        else:
            flatten_lc, lc_trend = flatten(lc.time.value, lc.flux.value, window_length=period_window, return_trend=True,
                                   method=detrend_method, break_tolerance=0.5)
        flatten_lc = sigma_clip(flatten_lc, sigma_lower=20, sigma_upper=3)
        return flatten_lc, lc_trend

    @contextmanager
    def __elapsed_timer(self):
        start = default_timer()
        elapser = lambda: str(default_timer())
        yield lambda: elapser()
        end = default_timer()
        elapser = lambda: str(end - start)
