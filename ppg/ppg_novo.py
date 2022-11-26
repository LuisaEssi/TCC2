# In[]:
# from turtle import distance
# from cv2 import threshold
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import csv
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
# from utils import SubjectPPGRecord

# MAT_FILE = "Subject14"
# sub = SubjectPPGRecord(MAT_FILE, db_path=".", mat=True)
# rec = sub.record
# fs = sub.fs
# red_ppg = rec.red_a
# ir_ppg = rec.ir_a
# spo2 = rec.spo2

# red = red_ppg[200000:]
# ir = ir_ppg[200000:]

lista = [] 
fs = 512
with open('dadosnosso.csv', newline='') as csvfile:
    # o nome 'spamreader' abaixo é só exemplo, poderia ser qq. coisa
    spamreader = csv.reader(csvfile, delimiter=',') # separe por vírgula

    # o módulo csv detectará novas linhas automaticamente
    for linha in spamreader:
        lista.append(linha)
    lista = np.array(lista,float)

# print(lista[0::,0], "RED") #
# print(lista[0::,1], "IR") #

red = lista[0::,0]
ir = lista[0::,1]

# red = red
# ir = float(ir)

# Time array

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

red_lp_cheby2 = cheby2_filter(red_hp, cutoff_lp, fs, order=LP_ORDER, rs = rs) # btype="low"
ir_lp_cheby2 = cheby2_filter(ir_hp, cutoff_lp, fs, order=LP_ORDER, rs = rs)

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
   
    vec_media_peak = []
    vec_media_peak_time = []
    vec_media_valley = []
    vec_media_valley_time = []
    media_peak = 0
    media_valley = 0
    peak_roll = [0]*order
    valley_roll = [0]*order
    amp = 1


    
    while (i < len(sample)):
        if i>= 2:      
            # media 
            # if(sample[i]>0):
            #     peak_roll[0] = sample[i]
            #     media_peak = ((np.sum(peak_roll))/order)
            #     peak_roll = np.roll(peak_roll, 1)
            #     vec_media_peak.append(media_peak*amp)
            #     vec_media_peak_time.append(time[i])
                
            # else:
            #     valley_roll[0] = sample[i]
            #     media_valley = ((np.sum(valley_roll))/order)
            #     valley_roll = np.roll(valley_roll, 1)
            #     vec_media_valley.append(media_valley*amp)
            #     vec_media_valley_time.append(time[i])

            if (sample[i]>sample[i-1]):
                num_upsteps = num_upsteps + 1 
                threshold = 10 + 0.6 * num_upsteps                 
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
                        if ((sample[i] <= value_possible_valley)): 
                            value_possible_valley = sample[i]
                            time_possible_valley = time[i]
                                                        
                    if (possible_peak == True):

                        if ((sample[i-1]>value_possible_peak)):
                            time_peak.append(time[i-1])
                            value_peak.append(sample[i-1])
                        else:
                            time_peak.append(time_possible_peak)
                            value_peak.append(value_possible_peak)

                        if ((possible_valley == True)):
                            time_valley.append(time_possible_valley)
                            value_valley.append(value_possible_valley)
                            possible_valley = False   
                          
                        possible_peak = False

                num_upsteps = 0
        i = i + 1
              
    print(len(vec_media_peak))
    print(len(vec_media_valley))    

    print(len(value_peak))
    print(len(value_valley))
    
    return value_peak, value_valley, time_peak, time_valley,vec_media_peak,\
           vec_media_valley,vec_media_peak_time,vec_media_valley_time





def elimina_p_v (value_peak, value_valley, time_peak, time_valley):
    value_valley_c = []
    value_peak_c = []
    time_valley_c = []
    time_peak_c = []
    i = 0
    j = 1
    temp_i = 0
    min_len = 0
    p_or_v = 1
    time_diff = 0.2
    order = 10
    vec_media_peak = []
    vec_media_peak_time = []
    vec_media_valley = []
    vec_media_valley_time = []
    media_peak = 0
    media_valley = 0
    peak_roll = [0]*order
    valley_roll = [0]*order
    amp = 0.5
    
    while (i < len(value_peak)):
        
        if((value_peak[i]>=0) and (p_or_v ==1)):
            peak_roll[0] = value_peak[i]
            media_peak = ((np.sum(peak_roll))/order)
            peak_roll = np.roll(peak_roll, 1)
            vec_media_peak.append(media_peak*amp)
            vec_media_peak_time.append(time_peak[i])

            if(value_peak[i]>=media_peak*amp):# and (p_or_v == 1)):
                value_peak_c.append(value_peak[i])
                time_peak_c.append(time_peak[i])
                p_or_v = 0

        if((value_valley[i]<=0) and (p_or_v == 0)):
            valley_roll[0] = value_valley[i]
            media_valley = ((np.sum(valley_roll))/order)
            valley_roll = np.roll(valley_roll, 1)
            vec_media_valley.append(media_valley*amp)
            vec_media_valley_time.append(time_valley[i])

            if(value_valley[i]<=media_valley*amp):# and (p_or_v == 0)):
                value_valley_c.append(value_valley[i])
                time_valley_c.append(time_valley[i])
                p_or_v = 1
            

    # if(len(value_peak) < len(value_valley)):
    #     min_len = len(value_peak)
    # else:
    #     min_len = len(value_valley)

    # if(time_peak[0] < time_valley[0]):
    #     p_or_v = 0

    # else:
    #     p_or_v = 1

    # # value_peak_c.append(value_peak[0])
    # # time_peak_c.append(time_peak[0])
    # # value_valley_c.append(value_valley[0])
    # # time_valley_c.append(time_valley[0])

    # while(i<min_len):
    #     if i>=1:    
    #         if((abs(time_peak[i] - time_valley[i])< time_diff)and(abs(time_peak[i-1] - time_valley[i-1])< time_diff)): #metade da freq pois está de pico a vale
    #             if(p_or_v ==0): #comeca com peak  
    #                 value_peak_c.append(value_peak[i-1])
    #                 time_peak_c.append(time_peak[i-1])
    #                 value_valley_c.append(value_valley[i])
    #                 time_valley_c.append(time_valley[i])
    #             else: 
    #                 value_peak_c.append(value_peak[i])
    #                 time_peak_c.append(time_peak[i])
    #                 value_valley_c.append(value_valley[i-1])
    #                 time_valley_c.append(time_valley[i-1])
    #             i = i+1
    #         else:
    #             value_peak_c.append(value_peak[i-1])
    #             time_peak_c.append(time_peak[i-1])
    #             value_valley_c.append(value_valley[i-1])
    #             time_valley_c.append(time_valley[i-1])

            # else: #comeca com valley
            #     if((abs(time_peak[i] - time_valley[i])< time_diff)and(abs(time_peak[i-1] - time_valley[i-1])< time_diff)): #metade da freq pois está de pico a vale
            #         value_peak_c.append(value_peak[i-1])
            #         time_peak_c.append(time_peak[i-1])
            #         value_valley_c.append(value_valley[i])
            #         time_valley_c.append(time_valley[i])
            #     else:
            #         value_peak_c.append(value_peak[i-1])
            #         time_peak_c.append(time_peak[i-1])
            #         value_valley_c.append(value_valley[i-1])
            #         time_valley_c.append(time_valley[i-1])
        #     print("amplitude: ",(abs(value_valley[i] - value_peak[i])))
        #     # (abs(value_valley[i]) + abs(value_peak[j-1]))
        #     if((abs(value_peak[i]-value_valley[i]) > time_diff)):
        #         value_peak_c.append(value_peak[i])
        #         time_peak_c.append(time_peak[i])
        #         value_valley_c.append(value_valley[i])
        #         time_valley_c.append(time_valley[i])
        #         # print(time_valley_c[i-1])
        #         # # print(time_peak_c[i-1])
        #         # j=j+temp_i
        #         # temp_i=0
        #         i = i+1
        #         j= j+1
        #     else:
        #         temp_i = temp_i+1
 
        i = i + 1

    # print((np.array(time_peak_c) - np.array(time_valley_c)))
    print("picos", len(value_peak_c))   
    print("vales", len(value_valley_c))

     # print(len(value_valley))


    return value_peak_c,value_valley_c, time_peak_c, time_valley_c, vec_media_peak,\
           vec_media_valley,vec_media_peak_time,vec_media_valley_time

    #_c => valores corrigidos


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
threshold_peak_valley = 6

peak_ir,valley_ir, time_p_ir,time_v_ir,vec_media_peak_ir,\
vec_media_valley_ir,vec_media_peak_ir_time,\
vec_media_valley_ir_time = find_peak_valley_2(ir_dc_cheby2, t, threshold_peak_valley,avg_order,2)

peak_red,valley_red, time_p_red,time_v_red,vec_media_peak_red,\
vec_media_valley_red,vec_media_peak_red_time,\
vec_media_valley_red_time = find_peak_valley_2(red_dc_cheby2, t, threshold_peak_valley, avg_order,2)


# ARRAY CORRIGIDOS
peak_ir_c = []
valley_ir_c = []
peak_red_c = []
valley_red_c =[]
time_p_ir_c = []
time_v_ir_c = []
time_p_red_c = []
time_v_red_c =[]
vec_media_peak_ir_c = []
vec_media_valley_ir_c = []
vec_media_peak_time_ir_c = []
vec_media_valley_time_ir_c = []

vec_media_peak_red_c = []
vec_media_valley_red_c = []
vec_media_peak_time_red_c = []
vec_media_valley_time_red_c = []

peak_ir_c,valley_ir_c, time_p_ir_c,time_v_ir_c, vec_media_peak_ir_c, vec_media_valley_ir_c, vec_media_peak_time_ir_c, vec_media_valley_time_ir_c = elimina_p_v (peak_ir,valley_ir, time_p_ir,time_v_ir)

peak_red_c,valley_red_c, time_p_red_c,time_v_red_c, vec_media_peak_red_c, vec_media_valley_red_c, vec_media_peak_time_red_c, vec_media_valley_time_red_c = elimina_p_v (peak_red,valley_red, time_p_red,time_v_red)


# plots sem correção
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
    plt.figure()

# # plots com correção

if PLOTS:

    plt.plot(t, ir_dc_cheby2, "k", label="PPG")
    plt.plot(time_p_ir_c, peak_ir_c, "r.", label="Threshold Peaks IR C")
    plt.plot(time_v_ir_c, valley_ir_c, "g.", label="Threshold Valley IR C")
    plt.plot(vec_media_peak_time_ir_c, vec_media_peak_ir_c,"y", label="Media peak IR")
    plt.plot(vec_media_valley_time_ir_c, vec_media_valley_ir_c,"m", label="Media valley IR")
    plt.legend()
    plt.show()
    plt.figure()
    
if PLOTS:
    plt.plot(t, red_dc_cheby2, "k", label="PPG")
    plt.plot(time_p_red_c, peak_red_c, "r.", label="Threshold Peaks RED")
    plt.plot(time_v_red_c, valley_red_c, "g.", label="Threshold Valley RED")
    plt.plot(vec_media_peak_time_red_c, vec_media_peak_red_c,"y", label="Media peak RED")
    plt.plot(vec_media_valley_time_red_c, vec_media_valley_red_c,"m", label="Media valley RED")
    plt.legend()
    plt.show()


# normalização 


# r_red = (np.array(peak_red[1:]) - np.array(valley_red))/np.array(peak_red[1:])
# r_ir = (np.array(peak_ir[1:]) - np.array(valley_ir))/np.array(peak_ir[1:])

r_red = (np.array(peak_red) - np.array(valley_red))/np.array(peak_red)
r_ir = (np.array(peak_ir) - np.array(valley_ir))/np.array(peak_ir)

# r_ir = (np.array(peak_ir) - np.array(valley_ir))/np.array(peak_ir)
# r_ir = (np.array[1:peak_ir] - np.array[1:valley_ir])/np.array(peak_ir)



# r = r_red[69:]/r_ir

r = r_red[3:]/r_ir

k = []
for i in range(len(r)):
    if ((r[i] <= 1.0) and r[i] >= 0):
        k.append(r[i])

k = np.array(k)
print(k)
print(len(r_red))
print(len(r_ir))
print(type(peak_red[0]))
# print(type(peak_ir[0]))

#%%
#SPO2 function

A = 94
B = -25
rfunction = lambda R: A + B * R 

spo2_test = rfunction(k)

if PLOTS:
    plt.stem(spo2_test, label = 'SpO2')  # type: ignore
    plt.legend()
    plt.show()


    

# value_r = r[k]

# a_ = 94 #110
# b_ = -25
# rfunction_ = lambda R_: a_ + b_ * R_

# spo2_ = rfunction_(value_r)

print(np.mean(spo2_test),"%")
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