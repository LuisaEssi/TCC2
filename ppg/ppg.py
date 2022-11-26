# In[]:
# from turtle import distance
from cv2 import threshold
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.signal import find_peaks
from filterbanks import cheby2_filter, cheby2_bandpass_filter, butter_filter, butter_bandpass_filter
from threshold_pev import adaptative_th, fix_pev, loc_pev

ZERO_PHASE_FILTERS = True
PLOTS = True
# In[]:
# Plot Settings
rc("font", **{"size": 18})
plt.close("all")
plt.ion()


# In[]:
# Load Subject Data using helper function
from utils import SubjectPPGRecord

MAT_FILE = "Subject14"
sub = SubjectPPGRecord(MAT_FILE, db_path=".", mat=True)
rec = sub.record
fs = sub.fs
red_ppg = rec.red_a
ir_ppg = rec.ir_a
spo2 = rec.spo2

red = red_ppg[200000:]
ir = ir_ppg[200000:]

# Time array
t = np.linspace(0, ((len(red) / fs)), len(red))
print(f"PPG Signal contains {len(red)} samples and {len(red)/fs} sec. duration.")


# In[]:
### BUTTER

# LOWPASS FILTER
cutoff_lp = 8  # desired cutoff frequency of the filter, Hz
LP_ORDER = 8
red_lp = butter_filter(red, cutoff_lp, fs, order=LP_ORDER)
ir_lp = butter_filter(ir, cutoff_lp, fs, order=LP_ORDER)

# HIGHPASS SIGNAL
cutoff_dc = 0.4
HP_ORDER = 8
red_hp = butter_filter(red_lp, cutoff_dc, fs, order=HP_ORDER, btype="high")
ir_hp = butter_filter(ir_lp, cutoff_dc, fs, order=HP_ORDER, btype="high")

# BANDPASS SIGNAL
BP_ORDER = 8
red_bp = butter_bandpass_filter(red, cutoff_dc, cutoff_lp, fs, order=BP_ORDER)
ir_bp = butter_bandpass_filter(ir, cutoff_dc, cutoff_lp, fs, order=BP_ORDER)


###CHEBY2

# LOWPASS FILTER
cutoff_lp2 = 6 # frequencia de corte
rs = 26

red_lp_cheby2 = cheby2_filter(red, cutoff_lp, fs, order=LP_ORDER, rs = rs) # btype="low"
ir_lp_cheby2 = cheby2_filter(ir, cutoff_lp, fs, order=LP_ORDER, rs = rs)

# HIGHPASS SIGNAL    # freq de corte pra remover sinal DC
HP_ORDER_CH = 6
red_dc_cheby2 = cheby2_filter(red_lp_cheby2, cutoff_dc, fs, order=HP_ORDER_CH, rs = rs, btype="high")
ir_dc_cheby2 = cheby2_filter(ir_lp_cheby2, cutoff_dc, fs, order=HP_ORDER_CH, rs = rs, btype="high")

# BANDPASS SIGNAL

# red_bp_cheby2 = cheby2_bandpass_filter(red, cutoff_dc, cutoff_lp, fs, rs = 26, order=HP_ORDER)
# ir_bp_cheby2 = cheby2_bandpass_filter(ir, cutoff_dc, cutoff_lp, fs, rs = 26, order=HP_ORDER)


#%%
# - np.mean(ir[start * fs : end * fs]

# if PLOTS:
#     fig, ax = plt.subplots()
#     start = 0
#     end = 5000
#     plt.plot(
#         t[start * fs : end * fs], 
#         ir_dc_cheby2[start * fs : end * fs], 
#         linewidth=2, 
#         label="Sinal IR filtrado",
#     )
#     plt.show()
#     plt.plot(
#         t[start * fs : end * fs],
#         ir[start * fs : end * fs],
#         linewidth=1,
#         label="Sinal IR original",
#     )
    # # plt.ylim((-0.05, +0.05))
    # plt.xlabel("Tempo [sec]")
    # plt.ylabel("Amplitude [mV]")
    # plt.minorticks_on()
    # plt.grid(True, which="major", alpha=0.8)
    # plt.grid(True, which="minor", alpha=0.3)
    # plt.legend()
    # # plt.show()

#%%
def find_peak_valley_2(sample, time, threshold, order,gain):

    i = 1
    num_upsteps = 0
    possible_peak = False
    possible_valley = False
    value_possible_peak = 0
    time_possible_peak = 0
    value_possible_valley = 0
    time_possible_valley = 0
    value_peak = []
    value_valley = []
    time_valley = []
    time_peak = []
    media_peak = 0
    media_valley = 0
    peak_vector =[0]*order
    valley_vector = [0]*order
    vec_media_peak = []
    vec_media_peak_time = []
    vec_media_valley = []
    vec_media_valley_time = []
    soma = 0

    while (i < len(sample)):
        if i>= 2:
            if (sample[i]>sample[i-1]):
                num_upsteps = num_upsteps + 1 
                if (possible_valley == False):
                    possible_valley = True
                    value_possible_valley = sample[i-1]
                    time_possible_valley = time[i-1]
            else:
                if (num_upsteps >= threshold):
                    possible_peak = True
                    value_possible_peak = sample[i-1]
                    time_possible_peak = time[i-1]
                else:
                    if (possible_valley == True):
                        if (sample[i] <= value_possible_valley): 
                            value_possible_valley = sample[i]
                            time_possible_valley = time[i]
                                                        
                    if (possible_peak == True):
                        if (sample[i-1]<value_possible_peak):
                            # if(soma < 1): 
                            # # np.roll(peak_vector,1)
                            # # peak_vector[0]=sample[i-1]
                            # # media_peak =( (np.sum(peak_vector))/order)*gain
                            # # vec_media_peak.append(media_peak)
                            # # vec_media_peak_time.append(time[i-1])
                            # # if (media_peak-value_possible_peak<value_possible_peak):
                            #     time_peak.append(time[i-1])
                            #     value_peak.append(sample[i-1])
                            #     soma = soma+1
                            #     print("if inicial da soma peak",soma)

                        # else:
                    
                            peak_vector[0]=value_possible_peak
                            media_peak = ((np.sum(peak_vector))/order)
                            peak_vector=np.roll(peak_vector,1)
                            vec_media_peak.append(media_peak*0.5)
                            vec_media_peak_time.append(time_possible_peak)
                            if (abs(media_peak*0.5)<=abs(value_possible_peak)):
                                if (soma < 1):
                                    time_peak.append(time_possible_peak)
                                    value_peak.append(value_possible_peak)
                                    soma = soma+1
                                elif (possible_valley == True):
                            
                                    valley_vector[0]=value_possible_valley
                                    media_valley = ((np.sum(valley_vector))/order)
                                    valley_vector= np.roll(valley_vector,1) 
                                    vec_media_valley.append(media_valley*0.5)
                                    vec_media_valley_time.append(time_possible_valley)
                                    if ((media_valley*0.5)>=(value_possible_valley)):
                                        if (soma == 1):
                                            time_valley.append(time_possible_valley)
                                            value_valley.append(value_possible_valley)
                                            possible_valley = False 
                                            soma = 0                            
                        possible_peak = False
                num_upsteps = 0
               
                    
        i = i + 1
                
    threshold = 0.6 * num_upsteps
    return value_peak, value_valley, time_peak, time_valley,vec_media_peak,\
           vec_media_valley,vec_media_peak_time,vec_media_valley_time

# moving_average_peak =  
# moving_average_valley  
vec_media_peak_ir = []
vec_media_valley_ir  = []
vec_media_peak_red = []
vec_media_valley_red  = []
vec_media_peak_red_time = []
vec_media_valley_red_time  = []
vec_media_peak_ir_time = []
vec_media_valley_ir_time  = []

avg_order = 5
threshold_peak_valley = 25

peak_ir,valley_ir, time_p_ir,time_v_ir,vec_media_peak_ir,\
vec_media_valley_ir,vec_media_peak_ir_time,\
vec_media_valley_ir_time = find_peak_valley_2(ir_dc_cheby2, t, threshold_peak_valley,avg_order,2)

peak_red,valley_red, time_p_red,time_v_red,vec_media_peak_red,\
vec_media_valley_red,vec_media_peak_red_time,\
vec_media_valley_red_time = find_peak_valley_2(red_dc_cheby2, t, threshold_peak_valley, avg_order,2)

if PLOTS:

    plt.plot(t, ir_dc_cheby2, "k", label="PPG")
    plt.plot(time_p_ir, peak_ir, "r.", label="Threshold Peaks IR")
    plt.plot(time_v_ir, valley_ir, "g.", label="Threshold Valley IR")
    plt.plot(vec_media_peak_ir_time, vec_media_peak_ir,"y", label="Media peak IR")
    plt.plot(vec_media_valley_ir_time, vec_media_valley_ir,"m", label="Media valley IR")
    plt.legend()
    plt.show()
    plt.figure()
    
if PLOTS:
    plt.plot(t, red_dc_cheby2, "k", label="PPG")
    plt.plot(time_p_red, peak_red, "r.", label="Threshold Peaks RED")
    plt.plot(time_v_red, valley_red, "g.", label="Threshold Valley RED")
    plt.legend()
    plt.show()

# r_ir = (np.array(peak_ir) - np.array(valley_ir))/np.array(peak_ir)
r_red = (np.array(peak_red[1:]) - np.array(valley_red))/np.array(peak_red[1:])
r_ir = (np.array(peak_ir[1:]) - np.array(valley_ir))/np.array(peak_ir[1:])
# r_ir = (np.array[1:peak_ir] - np.array[1:valley_ir])/np.array(peak_ir)



r = r_red[69:]/r_ir

k = []
for i in range(len(r)):
    if ((r[i] <= 1.0) and r[i] >= 0):
        k.append(r[i])

k = np.array(k)
print(r)
print(len(r_red))
print(len(r_ir))
print(type(peak_red[0]))
# print(type(peak_ir[0]))

#%%
#SPO2 function

A = 110
B = -25
rfunction = lambda R: A + B * R 

spo2_test = rfunction(k)

if PLOTS:
    plt.stem(spo2_test, label = 'SpO2')  # type: ignore
    plt.legend()
    plt.show()


    

# value_r = r[k]

a_ = 94 #110
b_ = -25
rfunction_ = lambda R_: a_ + b_ * R_

# spo2_ = rfunction_(value_r)

print(np.mean(spo2_test))
# print(np.mean(spo2_))
print("FIM")

# In[]:
ppg = ir_dc_cheby2
l_ir, lnn_ir, lnn_off_ir, lnn_th_ir = adaptative_th(ppg_ir,fs)

# Plot Envelope & Threshold Marks

if PLOTS:
    fig, ax = plt.subplots()
    plt.plot(l_ir, "k", label="PPG Mask")
    plt.plot(lnn_ir, lnn_th_ir, "r", label="Threshold")
    plt.stem(lnn_ir, l_ir[lnn_ir], label="ON")  # type: ignore
    # plt.stem(lnn_off, l[lnn_off], label='OFF') # type: ignore
    plt.minorticks_on()
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.6)
    plt.legend()
    plt.show()

# In[]:
# Interpolate threshold for all points +th, None
th = np.interp(np.arange(0, len(ppg_ir)), lnn_ir, lnn_th_ir)
PKS_DIST = np.ceil(300e-3 * fs)
# Peaks above threshold
p_pks, _ = find_peaks(ppg_ir, height=(+th, None), distance=PKS_DIST)
# Valleys below threshold
n_pks, _ = find_peaks(-ppg_ir, height=(+th, None), distance=PKS_DIST)  # type:ignore
pks = np.sort(np.concatenate((p_pks, n_pks)))

# In[]:
# Plot Original Signal & Peak Threshold
if PLOTS:
    fig, ax = plt.subplots()
    plt.plot(t, ppg_ir, "k", label="PPG")
    plt.plot(t[lnn_ir], lnn_th_ir, "r", t[lnn_ir], -lnn_th_ir, "r", label="Threshold")
    plt.stem(t[pks], ppg_ir[pks], label="Peaks")  # type: ignore
    plt.minorticks_on()
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.6)
    plt.legend()

# In[]:
#RED Thresholding
ppg_red = red_dc_cheby2
l_red, lnn_red, lnn_off_red, lnn_th_red = adaptative_th(ppg_red,fs)

# Interpolate threshold for all points +th, None
thr = np.interp(np.arange(0, len(ppg_red)), lnn_red, lnn_th_red)
PKS_DISTR = np.ceil(300e-3 * fs)
# Peaks above threshold
rp_pks, _ = find_peaks(ppg_red, height=(+thr, None), distance=PKS_DISTR)
# Valleys below threshold
rn_pks, _ = find_peaks(-ppg_red, height=(+thr, None), distance=PKS_DISTR)  # type:ignore
rpks = np.sort(np.concatenate((rp_pks, rn_pks)))

#%%
if PLOTS:
    fig, ax = plt.subplots()
    plt.plot(t, ppg_red, "k", label="PPG")
    # plt.plot(t[lnn_red], lnn_th_red, "r", t[lnn_red], -lnn_th_red, "r", label="Threshold")
    plt.stem(t[rpks], ppg_red[rpks], label="Peaks")  # type: ignore
    plt.minorticks_on()
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.6)
    plt.legend()


#%%

lpev_red = fix_pev(ppg_red,rpks)
lpev_ir = fix_pev(ppg_ir,pks)
#%%

l_neg_ir, l_pos_ir, x_ir = loc_pev(lpev_ir)
l_neg_red, l_pos_red, x_red = loc_pev(lpev_red)

#%%
def orderedsearch(data, alist, item):
    
    for i, w in enumerate(alist): #range(len(alist)):
        if data[w] == item:
            return w
    return None    

y_red = []

for i in range(len(x_red)):
    pos = orderedsearch(ppg_red, rpks, x_red[i])
    if pos is not None:
        y_red.append(pos)
    else:
        None

y_ir = []

for i in range(len(x_ir)):
    posir = orderedsearch(ppg_ir, pks, x_ir[i])
    if posir is not None:
        y_ir.append(posir)
    else:
        None


#%%
#AC/DC 

acdc_ir = x_ir-l_neg_ir/y_ir #peaky location aaaaa 
acdc_red = x_red-l_neg_red/y_red #peaky location aaaaa
# 
acdc_ir = np.concatenate((acdc_ir, acdc_ir[len(acdc_ir)-(len(acdc_red)-len(acdc_ir)):len(acdc_ir)]))

R = acdc_red/acdc_ir

#%%

k = []
for i in range(len(R)):
    if R[i] <= 0.98:
        k.append(i)
    
k = np.array(k)

#%%

R = R[k]

A = 110
B = -25
rfunction = lambda r: A + B * r 

spo2_test = rfunction(R)

#%%
beats_t = t[y_ir]
beats_RR = np.diff(beats_t)
# RR -> HR (bpm)
HR = 60/beats_RR


# %%
# Z = red_lp_cheby2/red_dc_cheby2
# Y = ir_lp_cheby2/ir_dc_cheby2
# J = Z/Y
# A= 110
# B = -25
# rfunction = lambda r: A + B * r 

# spo2_test = rfunction(J)
# %%
spo2 = spo2[~np.isnan(spo2)]