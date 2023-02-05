import os

import foldedleastsquares
import lcbuilder.constants
import numpy as np
import lightkurve as lk
from astropy.timeseries import LombScargle

# Define the transiting candidate period and duration in days
period = 1.5
duration = 0.1

# Load the target pixel files using the lightkurve package
tpfs = []
transits_mask = []
mission = lcbuilder.constants.MISSION_TESS
tpfs_dir = "/home/martin/git_repositories/SHERLOCK-data/TOI 175/TIC307210830_all/tpfs"
if os.path.exists(tpfs_dir):
    for tpf_file in sorted(os.listdir(tpfs_dir)):
        tpf = lk.TessTargetPixelFile(tpfs_dir + "/" + tpf_file) if mission == lcbuilder.constants.MISSION_TESS else \
            lk.KeplerTargetPixelFile(tpfs_dir + "/" + tpf_file)
        for transit_mask in transits_mask:
            mask = foldedleastsquares.transit_mask(tpf.time.value, transit_mask["P"], transit_mask["D"] / 60 / 24,
                                                   transit_mask["T0"])
            tpf = tpf[~mask]
        tpfs.append(tpf)

# Define the aperture mask
cadence = 120
# Select the rolling window size
window_size = int(duration * 24 * 3600 / cadence)
# Initialize the arrays to store the flux and time data
flux_data = []
time_data = []
# Iterate over the target pixel files
for tpf in tpfs:
    aperture = np.full((tpf.shape[1], tpf.shape[2]), False)
    aperture[6:8, 6:8] = True #  aperture of size 2x3
    flux_data.append(tpf.flux[:, aperture].sum(axis=1).value)
    time_data.append(tpf.time.value)
# Concatenate the flux and time data from all the target pixel files
flux_data = np.concatenate(flux_data)
time_data = np.concatenate(time_data)
# Compute the rolling band histogram
rolling_mean = np.convolve(flux_data, np.ones(window_size) / window_size, mode='same')
residuals = flux_data - rolling_mean

# Compute the Lomb-Scargle periodogram
ls = LombScargle(time_data, residuals, normalization='psd')
frequencies = ls.autofrequency()
power = ls.power(frequencies)
periods = 1 / frequencies
FAP = ls.false_alarm_probability(power, method="bootstrap")

# Find the peak of the periodogram
peak_index = np.argmax(power)
peak_period = periods[peak_index]
peak_power = power[peak_index]
# Compute the SNR
SNR = peak_power / (peak_power + FAP)
# Print the results
print("Rolling band effect period: {:.4f} days".format(peak_period))
print("False alarm probability: {:.4f}".format(FAP))
print("SNR: {:.4f}".format(SNR))