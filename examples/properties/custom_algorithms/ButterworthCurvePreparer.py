from scipy import signal
from scipy.signal import savgol_filter

from sherlockpipe.curve_preparer.CurvePreparer import CurvePreparer
import numpy as np


class ButterworthCurvePreparer(CurvePreparer):
    def __init__(self):
        super().__init__()

    def prepare(self, object_info, time, flux, flux_err):
        clean_flux = savgol_filter(flux, 11, 3)
        flux_filter = flux - np.ones(len(clean_flux))
        filter = signal.butter(2, 12, btype='lp', fs=720, analog=False, output='sos')
        flux_filter = signal.sosfiltfilt(filter, flux_filter)
        clean_flux = flux_filter + np.ones(len(clean_flux))
        return time, clean_flux, flux_err