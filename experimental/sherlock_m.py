###
### The SHERLOCK PIPEline; Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-based seeKers 
### 
### Pipeline to download, detrend and search for transits in 2-min cadence TESS data
###
### Pozuelos, Thuillier & Garcia 
### V13 10/06/2020
###
### Change Log:
###               sherlock_m is devoted to find transit in lc which are given as modeled.csv(time, flux,flux_err).
###               This version chooses the signal with lartest SNR to keep searching. The SDE and FAP are printed but   
###               are not used to chose which is the best signal. This is motivated by a bug discovered by Max 
###               It has been reported to TLS developers, it will be fix soon. 
###
###

Version = 13

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wotan import flatten
from wotan import t14
import warnings
from astropy.stats import sigma_clip
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask
from transitleastsquares import catalog_info
from transitleastsquares import cleaned_array
import lightkurve as lk
from lightkurve import search_lightcurvefile
from scipy import stats
import csv
import pandas as pd


# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# User definitions
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

TIC_ID = 136916387    # TIC_ID of our candidate
det_met = 1          # detrended method: =1 for bi-weigth ; =2 for GP matern 3/2
N_detrends = 12.      # Num. of detrend models applied
time_units = 1       # units of time: =0 (julian day) ; =1 (barycenter tess julian day)
P_protec = 10        # We procted the transit for a hypothetical Earth-size planet of Period = P_protec days. By default set = 1
Pmin = 0.5           # Minimun period in units of day
Pmax = 20            # Maximum period in units of day
minutes = 10         # time in minutes to bin the lightcurve
SNR_min = 5          # Threshold limit to keep searching. Strong restriction =>10
SDE_min = 5          # Threshold limit to keep searching. Strong restriction =>15
FAP_max = 1e-1       # Threshold limit to keep searching. Strong restriction <=1e-4
mask = 0             # = 1 means you need a mask; =0 means you do not want to use a mask 


# No need to change anything else from this point to down.


# *************************************************************************************************************************************************************
# Function that prints texts in a .log file
# *************************************************************************************************************************************************************
def logprint(*text):
#    print(*text)
    original = sys.stdout
    with open( os.path.join('TIC'+str(TIC_ID)+'_report.log'), 'a' ) as f:
        sys.stdout = f
        print(*text)
    sys.stdout = original


# *************************************************************************************************************************************************************
# Function that trim the axis (remove 'N' points)
# *************************************************************************************************************************************************************
def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


# *************************************************************************************************************************************************************
# Function that compute the data
# *************************************************************************************************************************************************************
def analyse(det_met, time_i, flatten_i, ab, mass, mass_min, mass_max, radius, radius_min, radius_max, id_run) :
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Detrending of the data using WOTAN
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # By default is used a 'biweight' method.
    # It will detrend the lightcurve 'N_detrends' independent times
    from wotan import flatten

    if det_met == 2 :
        wl_min = 1
        wl_max = 12
    else :
        tdur = t14(R_s=radius, M_s=mass, P=P_protec, small_planet=True) # we define the typical duration of a small planet in this star
        wl_min = 3*tdur # minimum transit duration
        wl_max = 20*tdur # maximum transit duration

    wl_step = (wl_max-wl_min)/N_detrends
    wl = np.arange(wl_min,wl_max,wl_step) # we define all the posibles window_length that we apply
    global_flatten_lc = np.zeros((len(wl),len(flatten_i)))
    global_trend_lc = np.zeros((len(wl),len(flatten_i)))

    for i in range(0,len(wl)):
        if det_met == 2 :
            flatten_lc, trend_lc  = flatten(time_i, flatten_i, method='gp', kernel='matern', kernel_size=wl[i], return_trend=True, break_tolerance=0.5)
        else :
            flatten_lc, trend_lc  = flatten(time_i, flatten_i, window_length=wl[i], return_trend=True, method='biweight', break_tolerance=0.5 )
        global_flatten_lc[i] = flatten_lc
        global_trend_lc[i] = trend_lc

    ## save in the log file all the information concerning the detrendins applied
    warnings.filterwarnings("ignore")

    global_final = np.zeros((len(wl),len(flatten_i)))

    logprint('\n MODELS IN THE DETRENDING - Run '+str(id_run))
    logprint('========================================\n')

    if det_met == 2 :
        det_model = 'kernel_size:'
    else :
        det_model = 'window_size:'

    bins=len(time_i)*2/minutes
    bin_means, bin_edges, binnumber = stats.binned_statistic(time_i, flatten_i, statistic='mean', bins=bins)
    bin_stds, _, _ = stats.binned_statistic(time_i, flatten_i, statistic='std', bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    
    logprint('PDCSAP_FLUX_'+str(id_run),'\t ',' ------ ' , '\t', '------', '\t', '\t','RMS (ppm):', np.std(flatten_i)*1e6, '\t','RMS_10min (ppm):', np.std(bin_means[~np.isnan(bin_means)])*1e6 )
    for i in range(len(wl)):
        flatten = sigma_clip(global_flatten_lc[i], sigma_lower=20, sigma_upper=3)
        global_final[i]=flatten
        bins_i=len(time_i)*2/minutes
        bin_means_i, bin_edges_i, binnumber_i = stats.binned_statistic(time_i, flatten, statistic='mean', bins=bins_i)
        bin_stds_i, _, _ = stats.binned_statistic(time_i, flatten, statistic='std', bins=bins_i)
        bin_width_i = (bin_edges_i[1] - bin_edges_i[0])
        bin_centers_i = bin_edges_i[1:] - bin_width_i/2
        logprint('flatten_lc%s' %i ,'\t ','trend_lc%s' %i, '\t', det_model, format(wl[i],'0.4f'), '\t', 'RMS (ppm):', np.std(flatten)*1e6,'\t', 'RMS_10min (ppm):', np.std(bin_means_i[~np.isnan(bin_means_i)])*1e6 )

    ## save in a plot all the detrendings and all the data to inspect visually.
    cases = np.zeros((len(wl),1))
    for i in range(len(wl)):
        cases[i] = wl[i]

    figsize = (8, 8) #x,y
    cols = 3
    rows = len(cases) // cols

    shift = 2*(1.0-(np.min(flatten_i))) #shift in the between the raw and detrended data
    ylim_max = 1.0+3*(np.max(flatten_i)-1.0) #shift in the between the raw and detrended data
    ylim_min = 1.0-2.0*shift #shift in the between the raw and detrended data

    fig1, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axs = trim_axs(axs, len(cases))
    for ax, case in zip(axs, cases):
        if det_met == 2 :
            ax.set_title('ks=%s' % str(np.around(case, decimals=4)))
        else:
            ax.set_title('ws=%s' % str(np.around(case, decimals=4)))

        bins_i=len(time_i)*2/minutes
        bin_means_i, bin_edges_i, binnumber_i = stats.binned_statistic(time_i, global_final[np.nonzero(cases == case)[0][0]], statistic='mean', bins=bins_i)
        bin_stds_i, _, _ = stats.binned_statistic(time_i, global_final[np.nonzero(cases == case)[0][0]], statistic='std', bins=bins_i)
        bin_width_i = (bin_edges_i[1] - bin_edges_i[0])
        bin_centers_i = bin_edges_i[1:] - bin_width_i/2
        
        ax.plot(time_i, flatten_i, linewidth=0.05 ,color='black', alpha=0.75, rasterized=True)
        ax.plot(time_i, global_trend_lc[np.nonzero(cases == case)[0][0]], linewidth=1, color='orange', alpha=1.0)
        ax.plot(time_i, global_final[np.nonzero(cases == case)[0][0]]-shift, linewidth=0.05 ,color='teal', alpha=0.75, rasterized=True)
        ax.plot(bin_centers_i, bin_means_i-shift,  marker='.', markersize=2, color='firebrick', alpha=0.5, linestyle='none', rasterized=True)
        
        ax.set_ylim(ylim_min, ylim_max)
        plt.savefig('Detrends_'+'run_'+str(id_run)+'_TIC'+str(TIC_ID)+'.png', dpi=200)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Search of signals in the raw data
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    logprint('\n SEARCH OF SIGNALS - Run', id_run)
    logprint('=================================\n')

    logprint('PDCSAP_FLUX_'+str(id_run), '\t', 'Period' ,'\t', 'Per_err', '\t', 'N.Tran', '\t', 'Mean Depth(ppt)', '\t', 'T. dur(min)', '\t','T0', '\t', 'SNR', '\t', 'SDE','\t', 'FAP\n')
    model = transitleastsquares(time_i, flatten_i)
    results_pdcsap = model.power(u=ab, M_star=mass, M_star_min=mass_min, M_star_max=mass_max, R_star=radius, R_star_min=radius_min, R_star_max=radius_max, period_min=Pmin, period_max=Pmax, n_transits_min=n_tra, show_progress_bar=False)
    x=[]

    if results_pdcsap.T0 != 0:
        for j in range(0,len(results_pdcsap.transit_depths)):
            # print (results.transit_depths[i])
            x = np.append(x, results_pdcsap.transit_depths[j])
            x = x[~np.isnan(x)]
            depth = (1.-np.mean(x))*100/0.1 #we change to ppt units
    else:
        depth = results_pdcsap.transit_depths

    logprint('-----', '\t ',
        format(results_pdcsap.period, '.5f'),'\t ',
        format(results_pdcsap.period_uncertainty, '.6f'),'\t ',
        results_pdcsap.distinct_transit_count,'\t',
        format(depth,'.3f'), '\t',
        format(results_pdcsap.duration*24*60, '.1f'),'\t',
        results_pdcsap.T0,'\t ',
        format(results_pdcsap.snr,'.3f'),'\t ',
        format(results_pdcsap.SDE,'.3f'), '\t ',
        results_pdcsap.FAP)
    logprint('\n')

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Saving a plot of the best signal from raw data
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    fig = save_plot(time_i, global_final, results_pdcsap, i)

    fig.suptitle('PDCSAP_FLUX_'+str(id_run)+' ## SNR:'+
        str(format(results_pdcsap.snr,'.3f'))+' ## SDE:'+
        str(format(results_pdcsap.SDE,'.3f'))+' ## FAP:'+
        str(results_pdcsap.FAP))
    plt.savefig('Run_'+str(id_run)+'_PDCSAP-FLUX_'+'TIC'+str(TIC_ID)+'.png', bbox_inches='tight', dpi=200)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Search of signals in the detrended data
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    if det_met ==2 :
        logprint('Kernel_size','\t', 'Period' ,'\t', 'Per_err','\t', 'N.Tran','\t','Mean Depth(ppt)','\t','T. dur(min)','\t','T0', '\t', 'SNR', '\t', 'SDE','\t', 'FAP\n')
    else:
        logprint('Window_size','\t', 'Period' ,'\t', 'Per_err','\t', 'N.Tran','\t','Mean Depth(ppt)','\t','T. dur(min)','\t','T0', '\t', 'SNR', '\t', 'SDE','\t', 'FAP\n')
    SNR = np.zeros((len(wl), 1))
    SDE = np.zeros((len(wl), 1))
    FAP = np.zeros((len(wl), 1))
    periods = np.zeros((len(wl), 1))
    per_err = np.zeros((len(wl), 1))
    durations = np.zeros((len(wl), 1))
    tos = np.zeros((len(wl), 1))

    for i in range(len(wl)):
        print(i)
        model = transitleastsquares(time_i, global_final[i])
        results = model.power(u=ab, M_star=mass, M_star_min=mass_min, M_star_max=mass_max, R_star=radius, R_star_min=radius_min, R_star_max=radius_max, period_min=Pmin, period_max=Pmax, n_transits_min=n_tra, show_progress_bar=False)
        SNR[i] = results.snr
        SDE[i] = results.SDE
        FAP[i] = results.FAP
        periods[i] = results.period
        per_err[i] = results.period_uncertainty
        durations[i] = results.duration
        tos[i] = results.T0
        x = []
        if results.T0 != 0:
            for j in range(0, len(results.transit_depths)):
                # print (results.transit_depths[i])
                x=np.append(x, results.transit_depths[j])
                x = x[~np.isnan(x)]
                depth = (1.-np.mean(x))*100/0.1 #we change to ppt units
        else:
            depth = results.transit_depths
        logprint(format(wl[i], '.4f'), '\t ',
            format(results.period, '.5f'),'\t ',
            format(results.period_uncertainty, '.6f'),'\t ',
            results.distinct_transit_count,'\t',
            format(depth,'.3f'), '\t',
            format(results.duration*24*60, '.1f'),'\t',
            results.T0,'\t ',
            format(results.snr,'.3f'),'\t ',
            format(results.SDE,'.3f'), '\t ',
            results.FAP)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Saving a plot of the best signal from detrended data
        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        fig = save_plot(time_i, global_final, results, i)

        if det_met ==2 :
            fig.suptitle('Run '+str(id_run)+' ## kernel_size:'+str(format(wl[i], '.4f'))+' ## SNR:'+str(format(results.snr,'.3f'))+' ## SDE:'+str(format(results.SDE,'.3f'))+' ## FAP:'+str(results.FAP))
            #plt.savefig('TIC'+str(TIC_ID)+'_ks='+str(format(wl[i],'.4f'))+'_run_'+str(id_run)+'.png', bbox_inches='tight', dpi=200)
            plt.savefig('Run_'+str(id_run)+'_ks='+str(format(wl[i],'.4f'))+'_TIC'+str(TIC_ID)+'.png', bbox_inches='tight', dpi=200)
        else:
            fig.suptitle('Run '+str(id_run)+' ## window_size:'+str(format(wl[i], '.4f'))+' ## SNR:'+str(format(results.snr,'.3f'))+' ## SDE:'+str(format(results.SDE,'.3f'))+' ## FAP:'+str(results.FAP))
            #plt.savefig('TIC'+str(TIC_ID)+'_ws='+str(format(wl[i],'.4f'))+'_run_'+str(id_run)+'.png', bbox_inches='tight', dpi=200)
            plt.savefig('Run_'+str(id_run)+'_ws='+str(format(wl[i],'.4f'))+'_TIC'+str(TIC_ID)+'.png', bbox_inches='tight', dpi=200)
    SNR=np.nan_to_num(SNR)
    a = np.nanargmax(SNR)  #check the maximum SRN

    if results_pdcsap.snr > SNR[a]:
        logprint('\nBest Signal -->', '\t','PDCSAP_FLUX' ,'\t', 'SNR:',format(results_pdcsap.snr,'.3f'))

        if (results_pdcsap.snr > SNR_min): # and results_pdcsap.SDE > SDE_min and results_pdcsap.FAP < FAP_max):
            logprint('\nBest Signal is good enough to keep searching. Going to the next run.')
            key=1
        else:
            logprint('\nBest Signal does not look very promising. End')
            key=0
    else:
        if det_met ==2:
            logprint('\nBest Signal -->', '\t','flatten_lc'+str(a) ,'\t', 'kernel_size:', format(wl[a],'0.4f'),'\t' ,'SNR:',format(SNR[a][0],'.3f'))
        else:
            logprint('\nBest Signal -->', '\t','flatten_lc'+str(a),'\t' , 'window_size:', format(wl[a],'0.4f'),'\t' ,'SNR:',format(SNR[a][0],'.3f'))

        if (SNR[a] > SNR_min): #and SDE[a] > SDE_min and FAP[a] < FAP_max):
            logprint('\nBest Signal is good enough to keep searching. Going to the next run.')
            key=1
        else:
            logprint('\nBest Signal does not look very promising. End')
            key=0

    print("### SNR :", str(format(results.snr,'.3f')), "   SDE :", str(format(results.SDE,'.3f')), "   FAP :", str(results.FAP))

    return results_pdcsap, SNR, key, periods, durations, tos


# *************************************************************************************************************************************************************
# Function that save a plot of the data
# *************************************************************************************************************************************************************
def save_plot(time_i, global_final, results, i) :
    #start the plotting
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10,3), constrained_layout=True)
    #1-Plot all the transits
    in_transit = transit_mask(time_i, results.period, results.duration, results.T0)
    ax1.scatter(time_i[in_transit], global_final[i][in_transit], color='red', s=2, zorder=0)
    ax1.scatter(time_i[~in_transit], global_final[i][~in_transit], color='black', alpha=0.05, s=2, zorder=0)
    ax1.plot(results.model_lightcurve_time, results.model_lightcurve_model, alpha=1, color='red', zorder=1)
    #plt.scatter(time_n, flux_new_n, color='orange', alpha=0.3, s=20, zorder=3)
    plt.xlim(time.min(), time.max())
    #plt.xlim(1362.0,1364.0)
    ax1.set(xlabel='Time (days)', ylabel='Relative flux')

    #phase folded plus binning
    ax2.plot(results.model_folded_phase, results.model_folded_model, color='red')
    ax2.scatter(results.folded_phase, results.folded_y, color='black', s=10, alpha=0.05, zorder=2)
    if (results.SDE) != 0:
        bins = 200
        bin_means, bin_edges, binnumber = stats.binned_statistic(results.folded_phase, results.folded_y, statistic='mean', bins=bins)
        bin_stds, _, _ = stats.binned_statistic(results.folded_phase, results.folded_y, statistic='std', bins=bins)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        ax2.errorbar(bin_centers, bin_means, yerr=bin_stds/2, xerr=bin_width/2, marker='o', markersize=4, color='teal', alpha=1, linestyle='none')
        ax2.set_xlim(0.45, 0.55)
        ax2.set(xlabel='Phase', ylabel='Relative flux')
        plt.ticklabel_format(useOffset=False)
    else:
        ax2.set_xlim(0.45, 0.55)
        ax2.set(xlabel='Phase', ylabel='Relative flux')
        plt.ticklabel_format(useOffset=False)

    #SDE
    ax3 = plt.gca()
    ax3.axvline(results.period, alpha=0.4, lw=3)
    plt.xlim(np.min(results.periods), np.max(results.periods))
    for n in range(2, 10):
        ax3.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax3.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    ax3.set(xlabel='Period (days)', ylabel='SDE')
    ax3.plot(results.periods, results.power, color='black', lw=0.5)
    ax3.set_xlim(0., max(results.periods))

    return fig


# *************************************************************************************************************************************************************
# Function that add a mask on the raw data
# *************************************************************************************************************************************************************
def add_mask(time_i, flatten_i, results_pdcsap, periods, durations, tos, a):
    if results_pdcsap.snr > SNR[a]:
        intransit = transit_mask(time_i, results_pdcsap.period, 2*results_pdcsap.duration, results_pdcsap.T0)
    else:
        intransit = transit_mask(time_i, periods[a], 2*durations[a], tos[a])

    flatten_i[intransit]=np.nan
    time_j, flatten_j = cleaned_array(time_i, flatten_i)

    return time_j, flatten_j


# *************************************************************************************************************************************************************
# Main
# *************************************************************************************************************************************************************

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
#      Initialization
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# Search of data by lightkurve function concerning the TIC_ID provided
# It retrieves all the data (sectors) available. 
# save in log file the info
ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=TIC_ID)
#search and donwload the TESS data using lightkurve
lcf= lk.search_lightcurvefile('TIC '+str(TIC_ID), mission="tess").download_all()

#we set the minum number of transits based on the sectors available
if len(lcf) > 1: 
    n_tra = 2
else:
    n_tra = 1

# saving usere definitions to keep the track of what have be done already
logprint('\n SHERLOCK (Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-base seeKers)')
logprint('\n Version ', Version )
logprint('\n USER DEFINITIONS')
logprint('========================\n')
logprint('TIC_ID: ', TIC_ID)

if det_met == 2 :
    logprint('Detrend method: Gaussian Process Matern 2/3')
else:
    logprint('Detrend method: Bi-Weight')

logprint('No of detrend models applied: ', N_detrends)
logprint('No of sectors available: ', len(lcf))
logprint('Minimum number of transits: ', n_tra)
logprint('Period planet protected: ', P_protec)
logprint('Minimum Period (d): ', Pmin)
logprint('Maximum Period (d): ', Pmax)
logprint('Binning size (min): ', minutes)

if time_units ==1 :
    logprint('Units of time: Barycenter TESS Julian Day')
else:
    logprint('Units of time: Julian Day')
if mask ==1 :
    logprint('Mask: yes')
else:
    logprint('Mask: no')

logprint('Threshold limit for SNR: ', SNR_min)
logprint('Threshold limit for SDE: ', SDE_min)
logprint('Threshold limit for FAP: ', FAP_max)

# We save here in a log file all the parameters available of the star.

mass_min = mass-mass_min
mass_max = mass+mass_max
radius_min = radius-radius_min
radius_max = radius+radius_max

logprint('\n STELLAR PROPERTIES FOR THE SIGNAL SEARCH')
logprint('================================================\n')
logprint('limb-darkening estimates using quadratic LD (a,b)=', ab)
logprint('mass =', format(mass,'0.5f'))
logprint('mass_min =', format(mass_min,'0.5f'))
logprint('mass_max =', format(mass_max,'0.5f'))
logprint('radius =', format(radius,'0.5f'))
logprint('radius_min =', format(radius_min,'0.5f'))
logprint('radius_max =', format(radius_max,'0.5f'))



logprint('\n SECTORS INFO')
logprint('================================================\n')
logprint('Sectors :', lcf)

lc=lcf.PDCSAP_FLUX.stitch().remove_nans() # remove of the nans
flux=lc.flux
flux_err=lc.flux_err
if time_units == 0:
    time=lc.astropy_time.jd #transform from TESS julian date to jd
elif time_units == 1:
    time=lc.time  #keep the TESS julian date

#we insert here the lc as modeled.csv
df = pd.read_csv('modeled.csv',float_precision='round_trip', sep=',',
                 usecols=['#TBJD','flux', 'flux_err'])

lc_new=lk.LightCurve(time=df['#TBJD'], flux=df['flux'],flux_err=df['flux_err'])
#lc_new=lk.LightCurve(time=time, flux=flux,flux_err=flux_err)
flux_clean=lc_new.remove_outliers(sigma_lower=float('inf'), sigma_upper=3) #remove outliers over 3sigma

if mask==1:
    #first mask (example of how to mask a range)
    time_in=1449.8
    time_out=1451.50
    mask1 = (flux_clean.time < time_in) | (flux_clean.time > time_out) 
    flux_clean = flux_clean[mask1]
    time_b=1463.5
    mask2 = (flux_clean.time < time_b)
    flux_clean = flux_clean[mask2]
    logprint('** Masking the lightcurve **')
else:
    None
        

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
#      Start of the runs
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
print("________________________________ run 1 ________________________________")
id_run = 1
time_i = flux_clean.time
flatten_i = flux_clean.flux
results_pdcsap, SNR, key, periods, durations, tos = analyse(det_met, time_i, flatten_i, ab, mass, mass_min, mass_max, radius, radius_min, radius_max, id_run)

while key == 1:
    id_run += 1
    print("________________________________ run", id_run, "________________________________")
    SNR=np.nan_to_num(SNR)
    a = np.nanargmax(SNR)  # check the maximum SDE

    # applying a new mask
    time_i, flatten_i = add_mask(time_i, flatten_i, results_pdcsap, periods, durations, tos, a)

    # analyzing the data with the new mask
    results_pdcsap, SNR, key, periods, durations, tos = analyse(det_met, time_i, flatten_i, ab, mass, mass_min, mass_max, radius, radius_min, radius_max, id_run)


# end
