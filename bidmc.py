
import scipy.io as sio
from scipy import signal
import numpy as np

def normalize_pipeline(signal):
    # normalize
    signal = (signal - np.min(signal)) / (np.max(signal) -np.min(signal))
    return signal


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
    return signal.medfilt(ppg, window_size
                          )

# define finite impulse response filter
def fir(ppg, order, low_cut, high_cut, sampling_rate):
    taps = signal.firwin(numtaps=101
                         , 
                         cutoff=[low_cut, high_cut], 
                         fs=sampling_rate, 
                         pass_zero=False)
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
def chebyshev2(ppg, order, ripple, low_cut, high_cut, sampling_rate):
    a, b = signal.cheby2(order, 
                         ripple,
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
    coeffs = pywt.wavedec(ppg, wavelet, level)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    filtered_signal = pywt.waverec(coeffs_thresholded, wavelet)

    return filtered_signal


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
                          parameters['ripple'], 
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
    'window_size': 51
}

butterworth_parameters = {
    'order': 3,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125
}

median_parameters = {
    'window_size': 51
}

fir_parameters = {
    'order': 3,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125,
}

chebyshev_parameters = {
    'order': 5,
    'ripple': 0.5,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125
}

chebyshev2_parameters = {
    'order': 5,
    'ripple': 0.5,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125
}

elliptic_parameters = {
    'order': 5,
    'ripple': 0.5,
    'rs': 40,
    'low_cut': minimum_frequency,
    'high_cut': maximum_frequency,
    'sampling_rate': 125
}

wavelet_denoising_parameters = {
    'wavelet': 'db4',
    'level': 2,
    'threshold': 0.5
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

def apply_filters(signal, filters, measurement_technique='heartrate'):

    filtered_signals = {}
    for cur_filter in filters.keys():
        filtered_signal = filter(signal, cur_filter, filters[cur_filter])
        print('filtered_signal shape: {}'.format(filtered_signal.shape))

        filtered_signals[cur_filter] = filtered_signal


    return filtered_signals

sampling_rate = 125 

# 60bpm = 1Hz, 180bpm = 3Hz
# 12-18 breaths per minute = 0.2-0.3Hz

param = 'heartrate' 

frequencies_by_params = {
    "heartrate": (1, 3),
    "respirationrate": (0.1, 0.33),
}

minimum_frequency = frequencies_by_params[param][0]
maximum_frequency = frequencies_by_params[param][1]

ppg_signal = ppg_signal[:3751
                        ,0]     

ppg_signal = normalize_pipeline(ppg_signal)
apply_filters(ppg_signal, filters, 'heartrate')

