#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import wfdb

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


# In[2]:


with open(f"{root_path}/bidmc_csv/{patient_csv_id}_Fix.txt", "r") as f:
    lines = [line.rstrip() for line in f]
    age = int(lines[5].split()[-1])
    gender = lines[6].split()[-1]


# In[3]:


def normalize_signal(signal):
    maximum = signal.max()
    minimum = signal.min()
    return (signal - minimum) / (maximum - minimum)


# In[4]:


def standardize_signal(signal):
    mean = signal.mean()
    std = signal.std()
    return (signal - mean) / std


# In[5]:


ppg = standardize_signal(ppg[:int(len(ppg)*0.005)])


# In[6]:


# Find peaks in the normalized PPG signal
peaks, _ = find_peaks(ppg, height=0.5)  # You can adjust the height parameter based on your signal characteristics

# Plot the PPG signal with detected peaks
plt.plot(ppg)
plt.plot(peaks, ppg[peaks], "x", label="peaks")
plt.title('PPG Signal with Peaks')
plt.legend()
plt.show()


# In[7]:


def zero_crossing_rate(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings)


# In[8]:


def mean_crossing_rate(signal):
    mean_value = np.mean(signal)
    crossings = np.where(np.diff((signal > mean_value).astype(int)))[0]
    return len(crossings)


# In[9]:


def spectral_peak(signal, sampling_rate):
    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)
    
    # Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    
    # Take the magnitude of the FFT result
    magnitude = np.abs(fft_result)
    
    freq_w_mag = frequencies[magnitude.argmax()]
    rate_w_mag = freq_w_mag * 60
    return rate_w_mag


# In[10]:


zcr = zero_crossing_rate(ppg)


# In[11]:


mcr = mean_crossing_rate(ppg)


# In[12]:


spectral_peak = spectral_peak(ppg, sampling_rate)


# In[13]:


zcr, mcr, spectral_peak


# In[ ]:




