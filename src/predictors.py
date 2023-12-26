from scipy.signal import find_peaks
import numpy as np


def peak_rate(signal, sampling_rate, max_frequency=3):
    peaks, _ = find_peaks(signal, distance=sampling_rate / max_frequency)
    return sampling_rate * 60 / np.diff(peaks).mean()


def zero_crossing_rate(signal, sampling_rate):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0][::2]
    return sampling_rate * 60 / np.diff(zero_crossings).mean()


def mean_crossing_rate(signal, sampling_rate):
    mean_value = np.mean(signal)
    crossings = np.where(np.diff((signal > mean_value).astype(int)))[0][::2]
    return sampling_rate * 60 / np.diff(crossings).mean()


def spectral_peak(signal, sampling_rate):
    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)

    # Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(len(signal), d=1 / sampling_rate)

    # Take the magnitude of the FFT result
    magnitude = np.abs(fft_result)

    freq_w_mag = np.abs(frequencies[magnitude.argmax()])
    rate_w_mag = freq_w_mag * 60
    return rate_w_mag


default_max_frequency_by_metric = {
    "hr": 3,
    "rr": 0.5,
}


def get_detector(detector_name, metric, signal_sampling_rate=125, max_frequency=None):
    if detector_name == "peak":
        if max_frequency is None:
            max_frequency = default_max_frequency_by_metric[metric]
        return lambda signal: peak_rate(
            signal,
            signal_sampling_rate,
            max_frequency=max_frequency,
        )
    elif detector_name == "zcr":
        return lambda signal: zero_crossing_rate(signal, signal_sampling_rate)
    elif detector_name == "mcr":
        return lambda signal: mean_crossing_rate(signal, signal_sampling_rate)
    elif detector_name == "spectral":
        return lambda signal: spectral_peak(signal, signal_sampling_rate)
    else:
        raise ValueError(f"Unknown detector name: {detector_name}")
