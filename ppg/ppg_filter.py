import numpy as np
from numpy.fft import fft, fftshift
from scipy.io import loadmat, savemat, whosmat
import matplotlib.pyplot as plt
from matplotlib import rc
from pprint import pprint  # Pretty Print
from scipy.signal import find_peaks, butter, lfilter, filtfilt

# In[]:
# Plot Settings
rc("font", **{"size": 18})
plt.close("all")
plt.ion()


# In[]:
f = open('the-zen-of-python.txt','r')

fs = 256
subject_data = f.astype(np.float)



red = subject_data[:, 5]  # channel A
ir = subject_data[:, 6]  # channel A
spo2 = subject_data[:, 9]

redppg = red[fs:-fs]
irppg = ir[fs:-fs]
t = np.linspace(0, len(redppg) / fs, len(redppg))
print(len(t))


# In[]:


# Plotando um trecho do sinal

Tppg = 100
fig, ax = plt.subplots()
plt.plot(redppg[25000 : Tppg * fs], linewidth=2, label="PPG RED ")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Time [samples]")
plt.legend()
plt.grid()


# In[]:


# Filtrando o sinal PPG


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band", analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)
    y = filtfilt(b, a, data)  # zero-phase
    return y


def butter_design(cutoff, fs, order, btype="low"):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a


def butter_filter(data, cutoff, fs, order, btype="low"):
    b, a = butter_design(cutoff, fs, btype=btype, order=order)
    # y = lfilter(b, a, data)
    y = filtfilt(b, a, data)  # zero-phase
    return y


# In[]:

# LOWPASS FILTER
cutoff_lp = 8  # desired cutoff frequency of the filter, Hz

red_lp = butter_filter(redppg, cutoff_lp, fs, order=4)
ir_lp = butter_filter(irppg, cutoff_lp, fs, order=4)

# HIGHPASS SIGNAL
cutoff_dc = 0.4
red_bp = butter_filter(red_lp, cutoff_dc, fs, order=4, btype="high")
ir_bp = butter_filter(ir_lp, cutoff_dc, fs, order=4, btype="high")

# BANDPASS FILTER 
# lowcut = 2
# highcut = 10
# red_bp = butter_bandpass_filter(red_lp, lowcut, highcut, fs, order=6)
# ir_bp = butter_bandpass_filter(ir_lp, lowcut, highcut, fs, order=6)


# plt.plot(t,redppg, 'b-', label='raw pgg')
# plt.plot(t,red_dc, 'b-', label='dc pgg')
fig, ax = plt.subplots()
start = 0
end = 5000
plt.plot(
    t[start * fs : end * fs], ir_bp[start * fs : end * fs], "g-", linewidth=2, label="filtered lp"
)
plt.plot(
    t[start * fs : end * fs],
    irppg[start * fs : end * fs] - np.mean(irppg[start * fs : end * fs]),
    "k-",
    linewidth=2,
    label="original",
)
# plt.ylim((-0.05, +0.05))
plt.xlabel("Time [sec]")
plt.grid()
plt.legend()
plt.show()