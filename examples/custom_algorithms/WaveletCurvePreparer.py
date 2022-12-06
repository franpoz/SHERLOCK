from scipy import signal
from scipy.signal import savgol_filter

from lcbuilder.curve_preparer.CurvePreparer import CurvePreparer
import numpy as np
import pywt


class WaveletCurvePreparer(CurvePreparer):
    def __init__(self):
        super().__init__()

    def prepare(self, object_info, time, flux, flux_err):
        clean_flux = self.suggestion(time, flux)
        return time, clean_flux, flux_err

    def suggestion(self, time, flux):
        from scipy.signal import periodogram, welch, savgol_filter
        import pywt
        clean_flux = flux
        #clean_flux = savgol_filter(flux, 11, 3)
        wavelet = "sym6"
        w = pywt.Wavelet(wavelet)
        # When you are performing the stationary wavelet transform (SWT) on a signal using PyWavelets, you can use
        # the pywt.dwt_max_level() function to determine the maximum number of levels that can be used in the
        # decomposition. This function takes the length of the signal and the name of the wavelet as input, and
        # returns the maximum number of levels that can be used in the SWT without introducing border effects.
        # Border effects can occur when the length of the signal is not a multiple of the wavelet filter length.
        # In this case, the signal cannot be decomposed completely using the SWT, and the decomposition may not
        # accurately capture the features of the signal near the edges of the signal.
        # The pywt.dwt_max_level() function can help to avoid border effects by limiting the number of levels in
        # the SWT to the maximum number that can be used without introducing border effects. This can ensure that
        # the decomposition accurately captures the features of the signal, even if the signal length is not a
        # multiple of the wavelet filter length.
        #maxlev = pywt.dwt_max_level(len(clean_flux), w.dec_len)
        maxlev = 4
        # maxlev = 2 # Override if desired
        print("maximum level is " + str(maxlev))
        # Decompose into wavelet components, to the level selected.
        # The pywt.swt() function performs the stationary wavelet transform (SWT) on a signal. This is a type of wavelet
        # transform that decomposes the signal into wavelet coefficients at different scales, but does not provide
        # information about the location of the coefficients in the original signal. This makes the SWT well-suited
        # for analyzing signals that are translation-invariant, such as signals with periodic or quasi-periodic behavior.
        # In contrast, the pywt.wavedec() function performs the wavelet decomposition (WD) on a signal. This is a
        # type of wavelet transform that decomposes the signal into wavelet coefficients at different scales and locations in the signal. This makes the WD well-suited for analyzing signals that have structure at multiple scales and locations, such as signals with different types of features at different positions in the signal.
        # In general, the pywt.swt() function is faster and more memory-efficient than the pywt.wavedec() function,
        # but it provides less information about the signal than the WD. Depending on the type of analysis you want
        # to perform, you may want to use the SWT or the WD.
        
        # For periodic transiting exoplanet signals in a light curve, the stationary wavelet transform (SWT) would
        # likely be the better choice. This is because the SWT is well-suited for analyzing signals that are
        # translation-invariant, such as signals with periodic or quasi-periodic behavior. In the case of transiting
        # exoplanet signals, the periodic nature of the signals makes the SWT a good fit for analyzing the light
        # curve data.
        # In contrast, the wavelet decomposition (WD) is better suited for analyzing signals that have structure
        # at multiple scales and locations, such as signals with different types of features at different positions
        # in the signal. For transiting exoplanet signals, the WD may not provide as much useful information as the
        # SWT, since the periodic nature of the signals means that they are not as sensitive to the location of the
        # features in the light curve.
        # Therefore, using the pywt.swt() function in PyWavelets would likely be the best choice for analyzing
        # periodic transiting exoplanet signals in a light curve. This function would allow you to decompose the
        # light curve into wavelet coefficients at different scales, which could then be used to analyze the periodic
        # features of the signals in the light curve data.
        #coeffs = pywt.wavedec(clean_flux, wavelet, level=maxlev)
        coeffs = pywt.wavedec(clean_flux, wavelet, level=maxlev)

        # If you were using the wavelet filter-bank for denoising or compression, you would typically threshold the
        # wavelet coefficients by setting small coefficients to zero and retaining only the largest coefficients.
        # This can be done using the pywt.threshold() function in PyWavelets.
        # Whitening a signal involves transforming the signal so that it has a flat power spectral density (PSD).
        # This means that all frequencies in the signal have the same power, and there are no dominant frequencies
        # or frequency bands in the signal. Whitening is often used as a preprocessing step for other signal processing
        # tasks, such as denoising or compression, because it can simplify the structure of the signal and make it
        # easier to analyze.
        # In contrast, applying a threshold to the wavelet coefficients is a way of reducing the number of coefficients
        # that are retained in the decomposition. This can be useful for denoising or compressing the signal, because
        # it reduces the amount of information that is stored in the decomposition and can eliminate noise or other
        # irrelevant features from the signal. However, it does not necessarily produce a whitened signal, because it
        # does not necessarily flatten the PSD of the signal.
        # Therefore, applying a threshold to the wavelet coefficients and whitening the signal are two different
        # operations that serve different purposes. In the example I provided, the goal was to show how to whiten a
        # signal using a wavelet filter-bank, so the wavelet coefficients were not thresholded. However, if your goal
        # is to denoise or compress the signal using the wavelet filter-bank, you may want to apply a threshold to the
        # coefficients to reduce their number and eliminate noise or irrelevant features from the signal.
        # Whitening a signal is not the same as denoising a signal, although they are related operations.
        # Whitening a signal involves transforming the signal so that it has a flat power spectral density (PSD), which
        # means that all frequencies in the signal have the same power. This is typically done as a preprocessing step
        # for other signal processing tasks, such as denoising or compression, because it can simplify the structure
        # of the signal and make it easier to analyze. However, whitening does not necessarily remove noise or other
        # irrelevant features from the signal. Instead, it changes the way that the signal is represented in the
        # frequency domain, which can make it easier to detect and remove noise or other features from the signal in
        # subsequent processing steps.
        # In contrast, denoising a signal involves removing noise or other irrelevant features from the signal. This
        # can be done using a variety of techniques, such as filtering, thresholding, or machine learning methods.
        # The goal of denoising is to produce a clean version of the signal that is free of noise or other irrelevant
        # features, and is more useful for the intended application. Denoising can be performed after whitening the
        # signal to make it easier to detect and remove noise or other features from the signal, but it is not the
        # same as whitening.
        # Therefore, while whitening and denoising are related operations that can be used together in signal
        # processing, they are not the same thing. Whitening changes the representation of the signal in the frequency
        # domain, while denoising removes noise or other irrelevant features from the signal.
        # Whether to denoise a signal before or after whitening it depends on the specific application and the
        # characteristics of the signal. In general, whitening a signal is typically done as a preprocessing step
        # before denoising, because it can simplify the structure of the signal and make it easier to detect and
        # remove noise or other irrelevant features from the signal.
        # However, there may be cases where it is useful to denoise a signal before whitening it. For example, if
        # the signal contains a large amount of high-frequency noise, it may be difficult to accurately estimate
        # the PSD of the signal in order to whiten it. In this case, denoising the signal to remove the high-frequency
        # noise first may make it easier to whiten the signal accurately.
        # Similarly, if the signal contains a large number of isolated or sparse features, such as transients or
        # glitches, whitening the signal before denoising it may cause these features to be spread out over a wider
        # range of frequencies. This can make it more difficult to detect and remove the features from the signal,
        # because they may not be concentrated at a single frequency or frequency band. In this case, denoising the
        # signal to remove the isolated features first may make it easier to whiten the signal without spreading the
        # features out over a wide range of frequencies.
        # Therefore, whether to denoise a signal before or after whitening it depends on the characteristics of the
        # signal and the goals of the signal processing. In general, it is usually best to whiten the signal first,
        # but there may be cases where denoising the signal first can improve the performance of the signal processing.
        # For transiting exoplanet signals in a light curve, where the signals are typically periodic and the light
        # curve also contains stellar variability and colored noise, it is generally best to whiten the signal first
        # and then denoise it.
        # Whitening the signal first will flatten the power spectral density (PSD) of the signal, which will make it
        # easier to detect and remove the periodic transiting exoplanet signals from the light curve. This is because
        # the periodic signals will have a relatively constant power at a specific frequency or frequency band, which
        # will be more pronounced after whitening. In contrast, if the signal is not whitened first, the periodic
        # signals may be spread out over a wide range of frequencies and may be more difficult to detect and remove.
        # After whitening the signal, denoising can then be used to remove any remaining noise or other irrelevant
        # features from the light curve. This will help to improve the signal-to-noise ratio of the transiting
        # exoplanet signals and make them easier to analyze. Denoising can be performed using a variety of techniques,
        # such as filtering, thresholding, or machine learning methods, depending on the specific characteristics of
        # the light curve and the goals of the signal processing.
        # Therefore, in the case of transiting exoplanet signals in a light curve, it is generally best to whiten
        # the signal first and then denoise it, in order to detect and remove the periodic signals and improve the
        # signal-to-noise ratio of the light curve data. This approach can help to simplify the structure of the
        # light curve and make it
        # cA = pywt.threshold(cA, threshold*max(cA))
        # threshold = 0.2  # Threshold for filtering
        # for i in range(1, len(coeffs)):
        #     coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]), mode="soft")

        # Set the wavelet coefficients to their absolute values
        for coeff in coeffs:
            cA, *cD = coeff
            cD = [np.abs(cd) for cd in cD]
            coeff = (cA, *cD)

        clean_flux = pywt.waverec(coeffs, wavelet)
        #clean_flux = pywt.waverec(coeffs, wavelet)
        # TODO ensure clean_flux has the same length and take action if not
        clean_flux = clean_flux if len(clean_flux) == len(time) else clean_flux[0:(len(clean_flux) - 1)]

        # f_p, psd_p = periodogram(clean_flux)
        # f_w, psd_w = welch(clean_flux)
        # power_p = np.trapz(psd_p, f_p)
        # power_w = np.trapz(psd_w, f_w)
        # snr_p = signal_power / power_p
        # snr_w = signal_power / power_w
        return clean_flux
