signal = s

sampling_rate = 125 

frequencies_by_metric = {
    "heartrate": (1, 3),
    "respirationrate": (0.1, 0.33),
}



minimum_frequency = 0.1
maximum_frequency = 3

signal = normalize_pipeline(signal)

filter1 = Filter()
filter2 = Filter()

def create_filter_butterworth(order, cutoff, sampling_rate):
    a, b = signal.butter(order, cutoff / (sampling_rate / 2), btype='low')
    return lambda x: signal.filtfilt(a, b, x)

measurement_technique = PeakDetection(params)
zerocrossing 
spectral_peak = 


heartrate: between 60 and 180bpm, which is 1-3Hz
respirationrate: between 6 and 20bpm, which is 0.1-0.33Hz

========================================================

filter1 = Filter(parameters)
signal = filter1(signal) # N x 1 -> N x 1

measurement_technique = PeakDetection(parameters)
metric = measurement_technique(signal) # N x 1 -> 1 



def apply_pipeline(signal, filters, measurement_technique):
    for filter in filters:
        signal = filter(signal)


    return measurement_technique(signal)