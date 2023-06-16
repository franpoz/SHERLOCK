import alexfitter
import batman
import foldedleastsquares
import pandas as pd
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

class Plotter:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def plot(self):
        alles = alexfitter.allesclass(self.results_dir)
        inst = 'lc'
        key = 'flux'
        time = alles.data[inst]['time']
        flux = alles.data[inst][key]
        flux_err = alles.data[inst]['err_scales_' + key] * alles.posterior_params_median['err_' + key + '_' + inst]
        ns_table_df = pd.read_csv(self.results_dir + '/results/ns_table.csv')
        ns_derived_df = pd.read_csv(self.results_dir + '/results/ns_derived_table.csv')
        star_df = pd.read_csv(self.results_dir + '/params_star.csv')
        star_radius = star_df['R_star'].iloc[0]
        ld_a = ns_derived_df.loc[ns_derived_df['#property'] == 'Limb darkening; $u_\\mathrm{1; lc}$', 'value'].iloc[0]
        ld_b = ns_derived_df.loc[ns_derived_df['#property'] == 'Limb darkening; $u_\\mathrm{2; lc}$', 'value'].iloc[0]
        for companion in alles.settings['companions_phot']:
            period = ns_table_df.loc[ns_table_df['#name'] == companion + '_period', 'median'].iloc[0]
            period_err_l = ns_table_df.loc[ns_table_df['#name'] == companion + '_period', 'lower_error'].iloc[0]
            period_err_u = ns_table_df.loc[ns_table_df['#name'] == companion + '_period', 'upper_error'].iloc[0]
            epoch = ns_table_df.loc[ns_table_df['#name'] == companion + '_epoch', 'median'].iloc[0]
            epoch_err_l = ns_table_df.loc[ns_table_df['#name'] == companion + '_epoch', 'lower_error'].iloc[0]
            epoch_err_u = ns_table_df.loc[ns_table_df['#name'] == companion + '_epoch', 'upper_error'].iloc[0]
            depth = ns_derived_df.loc[ns_derived_df['#property'] == 'Transit depth (undil.) ' + companion + '; $\\delta_\\mathrm{tr; undil; ' + companion + '; lc}$ (ppt)', 'value'].iloc[0]
            depth_err_l = ns_derived_df.loc[ns_derived_df['#property'] == 'Transit depth (undil.) ' + companion + '; $\\delta_\\mathrm{tr; undil; ' + companion + '; lc}$ (ppt)', 'lower_error'].iloc[0]
            depth_err_u = ns_derived_df.loc[ns_derived_df['#property'] == 'Transit depth (undil.) ' + companion + '; $\\delta_\\mathrm{tr; undil; ' + companion + '; lc}$ (ppt)', 'upper_error'].iloc[0]
            full_duration = ns_derived_df.loc[ns_derived_df['#property'] == 'Full-transit duration ' + companion + '; $T_\\mathrm{full;' + companion + '}$ (h)', 'value'].iloc[0]
            full_duration_err_l = ns_derived_df.loc[ns_derived_df['#property'] == 'Full-transit duration ' + companion + '; $T_\\mathrm{full;' + companion + '}$ (h)', 'lower_error'].iloc[0]
            full_duration_err_u = ns_derived_df.loc[ns_derived_df[ '#property'] == 'Full-transit duration ' + companion + '; $T_\\mathrm{full;' + companion + '}$ (h)', 'upper_error'].iloc[0]
            total_duration = ns_derived_df.loc[ns_derived_df['#property'] == 'Total transit duration ' + companion + '; $T_\\mathrm{tot;' + companion + '}$ (h)', 'value'].iloc[0]
            total_duration_err_l = ns_derived_df.loc[ns_derived_df['#property'] == 'Total transit duration ' + companion + '; $T_\\mathrm{tot;' + companion + '}$ (h)', 'lower_error'].iloc[0]
            total_duration_err_u = ns_derived_df.loc[ns_derived_df['#property'] == 'Total transit duration ' + companion + '; $T_\\mathrm{tot;' + companion + '}$ (h)', 'upper_error'].iloc[0]
            inclination = ns_derived_df.loc[ns_derived_df['#property'] == 'Inclination ' + companion + '; $i_\\mathrm{' + companion + '}$ (deg)', 'value'].iloc[0]
            semi_a = ns_derived_df.loc[ns_derived_df['#property'] == 'Semi-major axis ' + companion + ' over host radius; $a_\\mathrm{' + companion + '}/R_\\star$', 'value'].iloc[0]
            radius_earth = ns_derived_df.loc[ns_derived_df['#property'] == 'Companion radius ' + companion + '; $R_\\mathrm{' + companion + '}$ ($\\mathrm{R_{\\oplus}}$)', 'value'].iloc[0]
            radius_star_units = ((radius_earth * u.R_earth).to(u.R_sun) / (star_radius * u.R_sun)).value
            params = batman.TransitParams()  # object to store transit parameters
            params.t0 = epoch  # time of inferior conjunction
            params.per = period  # orbital period
            params.rp = radius_star_units  # planet radius (in units of stellar radii)
            params.a = semi_a  # semi-major axis (in units of stellar radii)
            params.inc = inclination  # orbital inclination (in degrees)
            params.ecc = 0.  # eccentricity
            params.w = 90.  # longitude of periastron (in degrees)
            params.limb_dark = "quadratic"  # limb darkening model
            params.u = [ld_a, ld_b]  # limb da  # times at which to calculate light curve
            m = batman.TransitModel(params, time)  # initializes model
            model = m.light_curve(params)
            total_duration_over_period = (total_duration / 24) / period
            time_folded = foldedleastsquares.fold(time, period, epoch + period / 2)
            data_df = pd.DataFrame(columns=['time', 'time_folded', 'flux', 'model'])
            data_df['time'] = time
            data_df['time_folded'] = time_folded
            data_df['flux'] = flux
            data_df['model'] = model
            data_df = data_df.loc[(data_df['time_folded'] > 0.5 - total_duration_over_period * 3) & (
                        data_df['time_folded'] < 0.5 + total_duration_over_period * 3)]
            time_sub = data_df['time'].to_numpy()
            time_folded_sub = data_df['time_folded'].to_numpy()
            flux_sub = data_df['flux'].to_numpy()
            model_sub = data_df['model'].to_numpy()
            data_df = data_df.sort_values(by=['time_folded'], ascending=True)
            bin_means, bin_edges, binnumber = binned_statistic(time_folded_sub, flux_sub, statistic='mean', bins=40)
            bin_stds, _, _ = binned_statistic(time_folded_sub, flux_sub, statistic='std', bins=40)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            time_binned = bin_centers
            flux_binned = bin_means
            plt.scatter(data_df['time_folded'].to_numpy(), data_df['flux'].to_numpy(), color='gray', alpha=0.1)
            plt.errorbar(time_binned, flux_binned, yerr=bin_stds / 2, xerr=bin_width / 2, marker='o', markersize=4,
                         color='blue', alpha=1, linestyle='none')
            plt.plot(data_df['time_folded'].to_numpy(), data_df['model'].to_numpy(), color='red')
            plt.xlim([0.5 - total_duration_over_period * 3, 0.5 + total_duration_over_period * 3])
            plt.ylim(
                [np.nanmin(flux_binned) - 2 * np.nanmax(bin_stds), np.nanmax(flux_binned) + 2 * np.nanmax(bin_stds)])
            plt.xlabel(r'Phase', fontsize='small')
            plt.ylabel(r'Flux norm.', fontsize='small')
            plt.title(r'')
            plt.show()

Plotter("/home/martin/git_repositories/SHERLOCK-data/TOI 696/TIC77156829_all_p10/fit_3/").plot()