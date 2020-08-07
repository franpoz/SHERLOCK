#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

#::: modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import ellc
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask, cleaned_array
from transitleastsquares import catalog_info
import astropy.constants as ac
import astropy.units as u
import lightkurve as lk
from lightkurve import search_lightcurvefile
from scipy import stats
from wotan import t14


np.random.seed(42)

#::: load data and set the units correctly
TIC_ID = 354687625    # TIC_ID of our candidate
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
    
    flux = flux0+model-1.
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
     
    
    return time, flux



def logprint(*text):
#    print(*text)
    original = sys.stdout
    with open( os.path.join('tls/'+'P = '+str(period)+' days, Rp = '+str(rplanet)+'.log'), 'a' ) as f:
        sys.stdout = f
        print(*text)
    sys.stdout = original
        


def IsMultipleOf(a, b, tolerance=0.05):
   a = np.float(a)
   b = np.float(b) 
   result = a % b
   return (abs(result/b) <= tolerance) or (abs((b-result)/b) <= tolerance)

    

#::: tls search
def tls_search(time, flux, epoch, period, rplanet):
    SNR = 1e12
    SNR_threshold = 5.
    FOUND_SIGNAL = False
    
    #::: mask out the first detection at 6.2 days, with a duration of 2.082h, and T0=1712.54922
    #intransit = transit_mask(time, 6.26391, 2*2.082/24., 1712.54922)
    #time = time[~intransit]
    #flux = flux[~intransit]
    time, flux = cleaned_array(time, flux)
    
    #::: search for the rest
    while (SNR >= SNR_threshold) and (FOUND_SIGNAL==False):
        logprint('\n====================================================================')
    
        model = transitleastsquares(time, flux)
        R_starx=rstar/u.R_sun
        #print(rstar_min/u.R_sun)
        #print(rstar_max/u.R_sun)
        #print(mstar/u.M_sun) 
        #print(mstar_min/u.M_sun)
        #print(mstar_max/u.M_sun)
        results = model.power(u=ab,
                              R_star= radius,#rstar/u.R_sun, 
                              R_star_min=rstar_min, #rstar_min/u.R_sun,
                              R_star_max=rstar_max, #rstar_max/u.R_sun,
                              M_star= mass, #mstar/u.M_sun, 
                              M_star_min= mstar_min,#mstar_min/u.M_sun,
                              M_star_max=mstar_max, #mstar_max/u.M_sun,
                              period_min=0.5,
                              period_max=20,
                              n_transits_min=1,
                              show_progress_bar=False
                              )
        
        #mass and radius for the TLS
#rstar=radius
##mstar=mass
#mstar_min = mass-massmin
#mstar_max = mass+massmax
#rstar_min = radius-radiusmin
#rstar_max = radius+raduismax
        
        
        if results.snr >= SNR_threshold:
            
            logprint('\nPeriod:', format(results.period, '.5f'), 'd')
            logprint(len(results.transit_times), 'transit times in time series:', ['{0:0.5f}'.format(i) for i in results.transit_times])
            logprint('Transit depth:', format(1.-results.depth, '.5f'))
            logprint('Best duration (days):', format(results.duration, '.5f'))
            logprint('Signal detection efficiency (SDE):', results.SDE)
            logprint('Signal-to-noise ratio (SNR):', results.snr)
        
            intransit = transit_mask(time, results.period, 2*results.duration, results.T0)
            time = time[~intransit]
            flux = flux[~intransit]
            time, flux = cleaned_array(time, flux)
            
            
            #::: check if it found the right signal
            right_period  = IsMultipleOf(results.period, period/2.) #check if it is a multiple of half the period to within 5%
            
            right_epoch = False
            for tt in results.transit_times:
                for i in range(-5,5):
                    right_epoch = right_epoch or (np.abs(tt-epoch+i*period) < (1./24.)) #check if any epochs matches to within 1 hour
                     
#            right_depth   = (np.abs(np.sqrt(1.-results.depth)*rstar - rplanet)/rplanet < 0.05) #check if the depth matches
                        
            if right_period and right_epoch:
                logprint('*** SUCCESFULLY FOUND THE TRANSIT ***')
                with open( os.path.join('tls_table/tls_table.csv'), 'a' ) as f:
                    f.write(str(period)+','+str(rplanet/u.R_earth)+','+'1\n')
                    FOUND_SIGNAL = True
                
          
        
        else:
            logprint('No other signals detected with SNR >= ' + str(SNR_threshold))
            if FOUND_SIGNAL == False:                
                logprint('*** FAILED FINDING THE TRANSIT ***')
                with open( os.path.join('tls_table/tls_table.csv'), 'a' ) as f:
                    f.write(str(period)+','+str(rplanet)+','+'0\n')
    
        SNR = results.snr
    
    
    
    
#::: iterate through grid of periods and rplanet
for period in np.arange(1,20,0.5):
    for rplanet in np.arange(2.7,0.65,-0.05):
        epoch = time[0]+np.random.rand()*period
        #epoch = time[0]+5
        rplanet = np.around(rplanet, decimals=2)*u.R_earth
        print('\n')
        print('P = '+str(period)+' days, Rp = '+str(rplanet))
        if not os.path.exists('tls/'+'P = '+str(period)+' days, Rp = '+str(rplanet)+'.log'):
#            print('do it')
            time, flux = make_model(epoch, period, rplanet)
            tls_search(time, flux, epoch, period, rplanet)
        else:
            print('already exists')
