from scipy import signal
from scipy.signal import savgol_filter

from lcbuilder.curve_preparer.CurvePreparer import CurvePreparer
import numpy as np
import pywt


class WaveletCurvePreparer(CurvePreparer):
    def __init__(self):
        super().__init__()

    def prepare(self, object_info, time, flux, flux_err):
        wavelet = "sym6"
        w = pywt.Wavelet(wavelet)
        maxlev = pywt.dwt_max_level(len(flux), w.dec_len)
        # maxlev = 2 # Override if desired
        print("maximum level is " + str(maxlev))
        threshold = 0.2  # Threshold for filtering
        clean_flux = flux
        # Decompose into wavelet components, to the level selected:
        coeffs = pywt.wavedec(clean_flux, wavelet, level=maxlev)
        # cA = pywt.threshold(cA, threshold*max(cA))
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]), mode="soft")
        clean_flux = pywt.waverec(coeffs, wavelet)
        # TODO ensure clean_flux has the same length and take action if not
        clean_flux = clean_flux if len(clean_flux) == len(time) else clean_flux[0:(len(clean_flux) - 1)]
        return time, clean_flux, flux_err
