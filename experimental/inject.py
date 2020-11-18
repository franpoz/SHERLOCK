#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import os, sys
import ellc
from transitleastsquares import catalog_info
import astropy.constants as ac
import astropy.units as u
import lightkurve as lk
import pandas as pd


np.random.seed(42)

#::: load data and set the units correctly
TIC_ID = 85400193    # TIC_ID of our candidate
lcf= lk.search_lightcurvefile('TIC '+str(TIC_ID), mission="tess").download_all()
ab, mass, massmin, massmax, radius, radiusmin, radiusmax = catalog_info(TIC_ID=TIC_ID)

#units for ellc
rstar=radius*u.R_sun
mstar=mass*u.M_sun
#mass and radius for the TLS
#rstar=radius
#mstar=mass
mstar_min = mass-massmin
mstar_max = mass+massmax
rstar_min = radius-radiusmin
rstar_max = radius+radiusmax

#uncomment the following lines to check that the parameters used are correct.

#print('\n STELLAR PROPERTIES FOR THE SIGNAL SEARCH')
#print('================================================\n')
#print('limb-darkening estimates using quadratic LD (a,b)=', ab)
#print('mass =', format(mstar,'0.5f'))
#print('mass_min =', format(mstar_min,'0.5f'))
#print('mass_max =', format(mstar_max,'0.5f'))
#print('radius =', format(rstar,'0.5f'))
#print('radius_min =', format(rstar_min,'0.5f'))
#print('radius_max =', format(rstar_max,'0.5f'))


lc=lcf.PDCSAP_FLUX.stitch().remove_nans() # remove of the nans
lc_new=lk.LightCurve(time=lc.time, flux=lc.flux,flux_err=lc.flux_err)
clean=lc_new.remove_outliers(sigma_lower=float('inf'), sigma_upper=3) #remove outliers over 3sigma
flux0=clean.flux
time=clean.time
flux_err = clean.flux_err
#period_maximum=(max(time)-min(time))/2.
#time, flux0 = np.genfromtxt('TESS_phot.csv', delimiter=',', unpack=True)
#rstar = 0.211257 * 41.46650444642 #in Rearth

#::: make model        
def make_model(epoch, period, rplanet):
    #a = (7.495e-6 * period**2)**(1./3.)*u.au #in AU
    P1=period*u.day
    a = np.cbrt((ac.G*mstar*P1**2)/(4*np.pi**2)).to(u.au)
    #print("radius_1 =", rstar.to(u.au) / a) #star radius convert from AU to in units of a 
    #print("radius_2 =", rplanet.to(u.au) / a)
    texpo=2./60./24.
    #print("T_expo = ", texpo,"dy")
    #tdur=t14(R_s=radius, M_s=mass,P=period,small_planet=False) #we define the typical duration of a small planet in this star
    #print("transit_duration= ", tdur*24*60,"min" )
    model = ellc.lc(
           t_obs = time,
           radius_1 = rstar.to(u.au) / a, #star radius convert from AU to in units of a
           radius_2 = rplanet.to(u.au) / a, #convert from Rearth (equatorial) into AU and then into units of a
           sbratio = 0,
           incl = 90,
           light_3 = 0,
           t_zero = epoch,
           period = period,
           a = None,
           q = 1e-6,
           f_c = None, f_s = None,
           ldc_1=[0.2755,0.5493], ldc_2 = None,
           gdc_1 = None, gdc_2 = None,
           didt = None,
           domdt = None,
           rotfac_1 = 1, rotfac_2 = 1,
           hf_1 = 1.5, hf_2 = 1.5,
           bfac_1 = None, bfac_2 = None,
           heat_1 = None, heat_2 = None,
           lambda_1 = None, lambda_2 = None,
           vsini_1 = None, vsini_2 = None,
           t_exp=texpo, n_int=None,
           grid_1='default', grid_2='default',
           ld_1='quad', ld_2=None,
           shape_1='sphere', shape_2='sphere',
           spots_1=None, spots_2=None,
           exact_grav=False, verbose=1)

    flux_t = flux0 + model - 1.
    if model[0] > 0:
        flux = flux_t
        flux_err_model = flux_err
        time_custom = time
    else:
        flux = []
        time_custom = []
        flux_err_model = []
    return time_custom, flux, flux_err_model
    #minutes=10
    #print(len(time))
    #print(min(time),max(time))
    #bins=len(time)*2./minutes
    #print(bins)
    #bin_means, bin_edges, binnumber = stats.binned_statistic(time, flux, statistic='mean', bins=bins)
    #bin_stds, _, _ = stats.binned_statistic(time, flux, statistic='std', bins=bins)
    #bin_width = (bin_edges[1] - bin_edges[0])
    #bin_centers = bin_edges[1:] - bin_width/2
    #print('RMS PDCSAP flux (ppm): ',np.std(flux0[~np.isnan(flux0)])*1e6)
    #print('RMS model (ppm): ',np.std(flux[~np.isnan(flux)])*1e6)
    #print('RMS 10min bin detrended (ppm): ',np.std(bin_means[~np.isnan(bin_means)])*1e6)
    
    #fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(10,5), constrained_layout=True)
    ##ax1
    #ax1.plot(time, flux0, linewidth=0.05 ,color='black', alpha=0.4)
    ##ax1.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.,fontsize=8)
    #ax1.set_ylabel("Normalized flux")
    #ax1.set_xlim(1766,1769)
    ##ax2
    #ax2.plot(time, flux0, linewidth=0.05 ,color='black', alpha=0.4)
    ##ax2.plot(time, model, linewidth=0.9 ,color='firebrick', alpha=1)
    #ax2.errorbar(time, model, marker='.', markersize=2, color='firebrick', alpha=1, linestyle='none')
    #ax2.set_ylabel("Normalized flux")
    #ax2.set_xlim(1766,1769)
    ##ax3
    #ax3.plot(time, flux, linewidth=0.1 ,color='teal', alpha=0.5)
    #ax3.errorbar(bin_centers, bin_means, marker='.', markersize=4, color='darkorange', alpha=1, linestyle='none')
    #ax3.set_ylabel("Normalized flux")
    #ax3.set_xlabel("Time (days)")
    #ax3.set_xlim(1766,1769)
    #plt.savefig('model.png', dpi=200)



def logprint(*text):
#    print(*text)
    original = sys.stdout
    with open( os.path.join('tls/'+'P = '+str(period)+' days, Rp = '+str(rplanet)+'.log'), 'a' ) as f:
        sys.stdout = f
        print(*text)
    sys.stdout = original

    
#::: iterate through grid of periods and rplanet
dir = "/home/pozuelos/martin/curves"
if not os.path.isdir(dir):
    os.mkdir(dir)
max_period = 10
min_period = 0.5
for period in np.arange(min_period, max_period, 0.5):
    for t0 in np.arange(time[60], time[60] + period - 0.1, period / 5):
        for rplanet in np.arange(4, 0.65, -0.1):
            rplanet = np.around(rplanet, decimals=2)*u.R_earth
            print('\n')
            print('P = '+str(period)+' days, Rp = '+str(rplanet) + ", T0 = " + str(t0))
            time_model, flux_model, flux_err_model = make_model(t0, period, rplanet)
            file_name = os.path.join(dir + '/P' + str(period) + '_R' + str(rplanet.value) + '_' + str(t0) + '.csv')
            lc_df = pd.DataFrame(columns=['#time', 'flux', 'flux_err'])
            lc_df['#time'] = time_model
            lc_df['flux'] = flux_model
            lc_df['flux_err'] = flux_err_model
            lc_df.to_csv(file_name, index=False)
