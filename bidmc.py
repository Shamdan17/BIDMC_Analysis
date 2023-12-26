import scipy.io as sio
from scipy import signal
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import wfdb
import pywt

root_path = "bidmc-ppg-and-respiration-dataset-1.0.0"
patient_id = "bidmc01"
patient_csv_id = "bidmc_01"

# Read the WFDB data file
ppg_record = wfdb.rdrecord(f'{root_path}/{patient_id}')
rate_record = wfdb.rdrecord(f'{root_path}/{patient_id}n')

# Access the signal and other information from the record
ppg = ppg_record.p_signal[:, 1]
hr = rate_record.p_signal[:, 0]
rr = rate_record.p_signal[:, 2]

sampling_rate = 125 

# 60bpm = 1Hz, 180bpm = 3Hz
# 12-18 breaths per minute = 0.2-0.3Hz

param = 'heartrate' 

frequencies_by_params = {
    "heartrate": (1, 3),
    "respirationrate": (0.1, 0.33),
}

distances_by_params = {
    "heartrate": sampling_rate / 3,
    "respirationrate": sampling_rate / 0.33,
}

minimum_frequency = frequencies_by_params[param][0]
maximum_frequency = frequencies_by_params[param][1]

def normalize_pipeline(signal):
    # normalize
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    return signal

def normalize_pipeline(signal):
    mean = signal.mean()
    std = signal.std()
    return (signal - mean) / std

# define moving average filter
def moving_average(ppg, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(ppg, window, 'same')

# define butterworth filter
def butterworth(ppg, order, low_cut, high_cut, sampling_rate):
    a, b = signal.butter(order, [low_cut, high_cut], fs=sampling_rate, btype='band')
    return signal.filtfilt(a, b, ppg)

# define median filter
def median(ppg, window_size):
    return signal.medfilt(ppg, window_size)

# define finite impulse response filter
def fir(ppg, order, low_cut, high_cut, sampling_rate):
    taps = signal.firwin(numtaps=101, 
                         cutoff=[low_cut, high_cut], 
                         fs=sampling_rate, 
                         width=0.5,
                         pass_zero='bandpass')
    return signal.filtfilt(taps, 1.0, ppg)

# define chebyshev filter
def chebyshev(ppg, order, ripple, low_cut, high_cut, sampling_rate):
    a, b = signal.cheby1(order, 
                         ripple,
                         [low_cut, high_cut], 
                         fs=sampling_rate,
                         btype='bandpass')
    return signal.filtfilt(a, b, ppg)

# define chebyshev type 2 filter
def chebyshev2(ppg, order, rs, low_cut, high_cut, sampling_rate):
    a, b = signal.cheby2(order, 
                         rs,
                         [low_cut, high_cut], 
                         fs=sampling_rate,
                         btype='bandpass')
    return signal.filtfilt(a, b, ppg)


# define elliptic filter
def elliptic(ppg, order, ripple, rs, low_cut, high_cut, fs):
    a, b = signal.ellip(order, 
                        ripple, 
                        rs, 
                        [low_cut, high_cut], 
                        fs=sampling_rate, 
                        btype='bandpass')
    return signal.filtfilt(a, b, ppg)

# define wavelet denoising filter
def wavelet_denoising(ppg, wavelet, level, threshold=0.5):
    # Choose a wavelet
    coeffs = pywt.wavedec(ppg, wavelet, level=level)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    coeffs_thresholded[-(level//2+1):] = [np.zeros_like(v) for v in coeffs_thresholded[-(level//2+1):]]

    filtered_signal = pywt.waverec(coeffs_thresholded, wavelet)

    return filtered_signal


# Define detectors
def peak_rate(signal):
    peaks, _ = find_peaks(signal, distance=distances_by_params[param]) 
    return sampling_rate * 60 / np.diff(peaks).mean()

def zero_crossing_rate(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0][::2]
    return sampling_rate * 60 / np.diff(zero_crossings).mean()

def mean_crossing_rate(signal):
    mean_value = np.mean(signal)
    crossings = np.where(np.diff((signal > mean_value).astype(int)))[0][::2]
    return sampling_rate * 60 / np.diff(crossings).mean()

def spectral_peak(signal):
    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)
    
    # Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    
    # Take the magnitude of the FFT result
    magnitude = np.abs(fft_result)
    
    freq_w_mag = np.abs(frequencies[magnitude.argmax()])
    rate_w_mag = freq_w_mag * 60
    return rate_w_mag



# define a general filter function that takes parameters
def filter(signal, filter_type, parameters):
    if filter_type == 'moving_average':
        return moving_average(signal, 
                              parameters['window_size'])
    
    elif filter_type == 'butterworth':
        return butterworth(signal, 
                           parameters['order'], 
                           parameters['low_cut'],
                            parameters['high_cut'],
                           parameters['sampling_rate'])
    
    elif filter_type == 'median':
        return median(signal, 
                      parameters['window_size'])
    
    elif filter_type == 'fir':
        return fir(
                   signal, 
                   parameters['order'],
                   parameters['low_cut'],
                   parameters['high_cut'],
                   parameters['sampling_rate']
                   )
    
    elif filter_type == 'chebyshev':
        return chebyshev(signal, 
                         parameters['order'], 
                         parameters['ripple'], 
                         parameters['low_cut'],
                         parameters['high_cut'],
                         parameters['sampling_rate']
                         )
    
    elif filter_type == 'chebyshev2':
        return chebyshev2(signal, 
                          parameters['order'], 
                          parameters['rs'], 
                          parameters['low_cut'],
                          parameters['high_cut'],
                          parameters['sampling_rate'])
    
    elif filter_type == 'elliptic':
        return elliptic(signal, 
                        parameters['order'], 
                        parameters['ripple'], 
                        parameters['rs'],
                        parameters['low_cut'],
                        parameters['high_cut'],
                        parameters['sampling_rate'])
    
    elif filter_type == 'wavelet_denoising':
        return wavelet_denoising(signal, 
                                 parameters['wavelet'], 
                                 parameters['level'],
                                 parameters['threshold'])
    else:
        raise Exception('Unknown filter type: {}'.format(filter_type))


moving_average_parameters = {
    'window_size': 25
}

butterworth_parameters = {
    'order': 2,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125
}

median_parameters = {
    'window_size': 25
}

fir_parameters = {
    'order': 3,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125,
}

chebyshev_parameters = {
    'order': 3,
    'ripple': 0.5,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125
}

chebyshev2_parameters = {
    'order': 3,
    'rs': 40,    
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125
}

elliptic_parameters = {
    'order': 3,
    'ripple': 0.5,
    'rs': 40,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125
}

wavelet_denoising_parameters = {
    'wavelet': 'db4',
    'level': 4,
    'threshold': 0.4
}

filters = {
    'moving_average': moving_average_parameters,
    'butterworth': butterworth_parameters,
    'median': median_parameters,
    'fir': fir_parameters,
    'chebyshev': chebyshev_parameters,
    'chebyshev2': chebyshev2_parameters,
    'elliptic': elliptic_parameters,
    'wavelet_denoising': wavelet_denoising_parameters
}

detectors = {
    'peak': peak_rate,
    'zcr': zero_crossing_rate,
    'mcr': mean_crossing_rate,
    'spectral': spectral_peak
}

def apply_filters(signal, filters, measurement_technique='heartrate'):

    filtered_signals = {}
    for cur_filter in filters.keys():
        filtered_signal = filter(signal, cur_filter, filters[cur_filter])
        print('filtered_signal shape: {}'.format(filtered_signal.shape))

        filtered_signals[cur_filter] = filtered_signal


    return filtered_signals


def apply_detectors(filtered_signals):
    for filter_name, filtered_signal in filtered_signals.items():
        for detector_name, cur_detector in detectors.items():
            prediction = cur_detector(filtered_signal)
            print(f'filter name: {filter_name}, detector name: {detector_name}, prediction: {prediction}')
            if np.isnan(prediction):
                plt.plot(filtered_signal)
                plt.show()

ppg = ppg[:3751]

ppg = normalize_pipeline(ppg)
filtered_signals = apply_filters(ppg, filters, 'heartrate') # apply_filters(ppg, filters, 'heartrate')
apply_detectors(filtered_signals)
