import scipy.io as sio
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pywt

# 60bpm = 1Hz, 180bpm = 3Hz
# 12-18 breaths per minute = 0.2-0.3Hz
default_frequencies_by_params = {
    "hr": (1, 3),
    "rr": (0.15, 0.5),
}


def normalize(signal):
    # normalize
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    return signal


def standardize(signal):
    mean = signal.mean()
    std = signal.std()
    return (signal - mean) / std


# define moving average filter
def moving_average(ppg, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(ppg, window, "same")


# define butterworth filter
def butterworth(ppg, order, low_cut, high_cut, sampling_rate):
    a, b = signal.butter(order, [low_cut, high_cut], fs=sampling_rate, btype="band")
    return signal.filtfilt(a, b, ppg)


# define median filter
def median(ppg, window_size):
    return signal.medfilt(ppg, window_size)


# define finite impulse response filter
def fir(ppg, order, low_cut, high_cut, sampling_rate):
    taps = signal.firwin(
        numtaps=101,
        cutoff=[low_cut, high_cut],
        fs=sampling_rate,
        width=0.5,
        pass_zero="bandpass",
    )
    return signal.filtfilt(taps, 1.0, ppg)


# define chebyshev filter
def chebyshev(ppg, order, ripple, low_cut, high_cut, sampling_rate):
    a, b = signal.cheby1(
        order, ripple, [low_cut, high_cut], fs=sampling_rate, btype="bandpass"
    )
    return signal.filtfilt(a, b, ppg)


# define chebyshev type 2 filter
def chebyshev2(ppg, order, rs, low_cut, high_cut, sampling_rate):
    a, b = signal.cheby2(
        order, rs, [low_cut, high_cut], fs=sampling_rate, btype="bandpass"
    )
    return signal.filtfilt(a, b, ppg)


# define elliptic filter
def elliptic(ppg, order, ripple, rs, low_cut, high_cut, fs):
    a, b = signal.ellip(order, ripple, rs, [low_cut, high_cut], fs=fs, btype="bandpass")
    return signal.filtfilt(a, b, ppg)


# define wavelet denoising filter
def wavelet_denoising(ppg, wavelet, level, threshold=0.5):
    # Choose a wavelet
    coeffs = pywt.wavedec(ppg, wavelet, level=level)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]

    coeffs_thresholded[-(level // 2 + 1) :] = [
        np.zeros_like(v) for v in coeffs_thresholded[-(level // 2 + 1) :]
    ]

    filtered_signal = pywt.waverec(coeffs_thresholded, wavelet)

    return filtered_signal


# define a general filter function that takes parameters
def filter(signal, filter_type, parameters):
    if filter_type == "normalize":
        return normalize(signal)

    elif filter_type == "standardize":
        return standardize(signal)

    elif filter_type == "moving_average":
        return moving_average(signal, parameters["window_size"])

    elif filter_type == "butterworth":
        return butterworth(
            signal,
            parameters["order"],
            parameters["low_cut"],
            parameters["high_cut"],
            parameters["sampling_rate"],
        )

    elif filter_type == "median":
        return median(signal, parameters["window_size"])

    elif filter_type == "fir":
        return fir(
            signal,
            parameters["order"],
            parameters["low_cut"],
            parameters["high_cut"],
            parameters["sampling_rate"],
        )

    elif filter_type == "chebyshev":
        return chebyshev(
            signal,
            parameters["order"],
            parameters["ripple"],
            parameters["low_cut"],
            parameters["high_cut"],
            parameters["sampling_rate"],
        )

    elif filter_type == "chebyshev2":
        return chebyshev2(
            signal,
            parameters["order"],
            parameters["rs"],
            parameters["low_cut"],
            parameters["high_cut"],
            parameters["sampling_rate"],
        )

    elif filter_type == "elliptic":
        return elliptic(
            signal,
            parameters["order"],
            parameters["ripple"],
            parameters["rs"],
            parameters["low_cut"],
            parameters["high_cut"],
            parameters["sampling_rate"],
        )

    elif filter_type == "wavelet_denoising":
        return wavelet_denoising(
            signal, parameters["wavelet"], parameters["level"], parameters["threshold"]
        )
    else:
        raise Exception("Unknown filter type: {}".format(filter_type))


moving_average_parameters = {"window_size": 25}

butterworth_parameters = {
    "order": 2,
    "low_cut": -1,
    "high_cut": -1,
}

median_parameters = {"window_size": 25}

fir_parameters = {
    "order": 3,
    "low_cut": -1,
    "high_cut": -1,
}

chebyshev_parameters = {
    "order": 3,
    "ripple": 0.5,
    "low_cut": -1,
    "high_cut": -1,
}

chebyshev2_parameters = {
    "order": 3,
    "rs": 40,
    "low_cut": -1,
    "high_cut": -1,
}

elliptic_parameters = {
    "order": 3,
    "ripple": 0.5,
    "rs": 40,
    "low_cut": -1,
    "high_cut": -1,
}

wavelet_denoising_parameters = {"wavelet": "db4", "level": 4, "threshold": 0.4}

filters = {
    "normalize": {},
    "standardize": {},
    "moving_average": moving_average_parameters,
    "butterworth": butterworth_parameters,
    "median": median_parameters,
    "fir": fir_parameters,
    "chebyshev": chebyshev_parameters,
    "chebyshev2": chebyshev2_parameters,
    "elliptic": elliptic_parameters,
    "wavelet_denoising": wavelet_denoising_parameters,
}


def get_default_filter_for_metric(
    filter_name, metric, sampling_rate=125, frequencies=None, parameters=None
):
    if frequencies is None:
        frequencies = default_frequencies_by_params[metric]

    return get_filter(filter_name, sampling_rate, frequencies, parameters)


def get_filter(filter_name, sampling_rate=125, frequencies=None, parameters=None):
    if parameters is None:
        parameters = filters[filter_name]
        # Copy the parameters so we don't modify the original
        parameters = {k: v for k, v in parameters.items()}

    if "low_cut" in parameters and parameters["low_cut"] == -1:
        parameters["low_cut"] = frequencies[0]
    if "high_cut" in parameters and parameters["high_cut"] == -1:
        parameters["high_cut"] = frequencies[1]

    parameters["sampling_rate"] = sampling_rate

    return lambda signal: filter(signal, filter_name, parameters)
