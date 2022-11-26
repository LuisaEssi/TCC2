#%%
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
red = rec.red_a
ir = rec.ir_a
spo2 = rec.spo2

#%%
# Split Signal by Stages
stages_duration = np.array([5*60, 12*60, 6*60, (1.5*60 + 1*60)*3, 6*60], dtype=np.int64) * fs
stages_duration_sum = stages_duration.cumsum()

def split_by_stage(sig):
    sig_stage = []
    prev_duration = 0
    for duration in stages_duration_sum:
        sig_stage.append(sig[prev_duration:duration])
        prev_duration = duration
    return sig_stage

ir_st = split_by_stage(ir)
red_st = split_by_stage(red)

# Time array
t = np.linspace(0, ((len(red) / fs)), len(red))
# t = np.linspace(0, stages_duration_sum[-1] / fs, stages_duration_sum[-1])
print(f"PPG Sinal contains {len(red)} samples and {len(red)/fs} sec. duration.")

# Crop Original Signal



#%%
# # Time array
# t = np.linspace(0, ((len(red) / fs)), len(red))
# print(f"PPG Sinal contains {len(red)} samples and {len(red)/fs} sec. duration.")

#%%
###CHEBY2

# LOWPASS FILTER
cutoff_lp = 8  # frequencia de corte
LP_ORDER = 8

red_lp_cheby2 = cheby2_filter(red, cutoff_lp, fs, order=LP_ORDER, rs = 26) # btype="low"
ir_lp_cheby2 = cheby2_filter(ir, cutoff_lp, fs, order=LP_ORDER, rs = 26)

# HIGHPASS SIGNAL    # freq de corte pra remover sinal DC
HP_ORDER_CH = 8
cutoff_dc = 0.4

red_dc_cheby2 = cheby2_filter(red_lp_cheby2, cutoff_dc, fs, order=HP_ORDER_CH, rs = 26, btype="high")
ir_dc_cheby2 = cheby2_filter(ir_lp_cheby2, cutoff_dc, fs, order=HP_ORDER_CH, rs = 26, btype="high")

# warmup = [:300*fs]
# aerob = [300*fs:1020*fs]
# rec_at = [1020*fs:1380*fs]
# anaer1 = [1380*fs:1470*fs]
# rec_at1 = [1470*fs:1530*fs]
# anaer2 = [1530*fs:1620*fs]
# rec_at2 = [1620*fs:1680*fs]
# anaer3 = [1680*fs:1770*fs]
# rec_at3 = [1770*fs:1830*fs]

start = 1770
end = 1830
red = red_dc_cheby2[start*fs:end*fs]
ir = ir_dc_cheby2[start*fs:end*fs]
t = t[start*fs:end*fs]
# plt.plot(t, red_dc_cheby2, "-k")

# %%

ppg_ir = ir
l_ir, lnn_ir, lnn_off_ir, lnn_th_ir = adaptative_th(ppg_ir,fs)

#%%
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

#%%
# Interpolate threshold for all points +th, None
th = np.interp(np.arange(0, len(ppg_ir)), lnn_ir, lnn_th_ir)
PKS_DIST = np.ceil(300e-3 * fs)
# Peaks above threshold
p_pks, _ = find_peaks(ppg_ir, height=(+th, None), distance=PKS_DIST)
# Valleys below threshold
n_pks, _ = find_peaks(-ppg_ir, height=(+th, None), distance=PKS_DIST)  # type:ignore
pks = np.sort(np.concatenate((p_pks, n_pks)))

#%%
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

# %%
# if len(n_pks) > len(p_pks):
#     x = np.concatenate((p_pks, p_pks[len(p_pks)-(len(n_pks)-len(p_pks)):len(p_pks)]))
# %%

#RED Thresholding
ppg_red = red
l_red, lnn_red, lnn_off_red, lnn_th_red = adaptative_th(ppg_red,fs)

# Interpolate threshold for all points +th, None
thr = np.interp(np.arange(0, len(ppg_red)), lnn_red, lnn_th_red)
PKS_DISTR = np.ceil(250e-3 * fs)
# Peaks above threshold
rp_pks, _ = find_peaks(ppg_red, height=(+thr, None), distance=PKS_DISTR)
# Valleys below threshold
rn_pks, _ = find_peaks(-ppg_red, height=(+thr, None), distance=PKS_DISTR)  # type:ignore
rpks = np.sort(np.concatenate((rp_pks, rn_pks)))
 

# %%
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

if len(acdc_ir) > len(acdc_red):
    acdc_red = np.concatenate((acdc_red, acdc_red[len(acdc_red)-(len(acdc_ir)-len(acdc_red)):len(acdc_red)]))
else:
    acdc_ir = np.concatenate((acdc_ir, acdc_ir[len(acdc_ir)-(len(acdc_red)-len(acdc_ir)):len(acdc_ir)]))

R = acdc_red/acdc_ir
# %%
k = []
for i in range(len(R)):
    if R[i] <= 0.97:
        k.append(i)
    
k = np.array(k)

#%%
R = R[k]

A = 110
B = -25
rfunction = lambda r: A + B * r 

spo2_test = rfunction(R)

print(np.mean(spo2_test))
print(np.std(spo2_test))
# %%
## 4 e 5:1.1, 6:0.95 7:1, 
### AQUECIMENTO -> 0 a 5 min == 0 a 300 sec 
### AERÓBICO -> 5 a 17 min == 300 a 1020 sec 
### RECUP. ATIVA -> 17 a 23 min == 1020 a 1380 sec 
### ANAERÓBICO -> 23 a 24 min == 1380 a 1440 sec 
### RECUP. ATIVA -> 24 a 25 min == 1440 a sec 
### ANAERÓBICO -> 25 a 26 min == 1380 a 1440 sec
### RECUP. ATIVA -> 26 a 27 min == 1440 a sec 
### ANAERÓBICO -> 27 a 28 min == 1380 a 1440 sec
### RECUP. ATIVA -> 29 a 36 min == 1440 a sec

### RELAXAMENTO -> 48 min até o final da gravação 

# winlen = fs
# stepsize = winlen // 2
# start = 300 
# end = 1020
# red = red_bp[start*fs:end*fs]
# ir = ir_bp[start*fs:end*fs]
# t = t[start*fs:end*fs]

# warmup = [:300*fs]
# aerob = [300*fs:1020*fs]
# rec_at = [1020*fs:1380*fs]
# anaer1 = [1380*fs:1470*fs]
# rec_at1 = [1470*fs:1530*fs]
# anaer2 = [1530*fs:1620*fs]
# rec_at2 = [1620*fs:1680*fs]
# anaer3 = [1680*fs:1770*fs]
# rec_at3 = [1770*fs:1830*fs]


# idxlist = range(0, len(ir_bp), stepsize)
# C = np.zeros((len(idxlist), winlen))

# for o, i in enumerate(range(0, len(ir_bp), stepsize)):
#     C[o, :] = ir_bp[i:i+winlen]
# %%
