#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy.fft import fft, fftshift
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import rc
from pprint import pprint  # Pretty Print
from scipy.signal import find_peaks, butter, lfilter, filtfilt

ZERO_PHASE_FILTERS = True
PLOTS = True

# In[3]:


# Plot Settings
rc('font', **{'size': 18})
plt.close('all')
plt.ion()


# In[4]:


# Carregando o sinal do arquivo .mat

mat = loadmat('Subject14.mat')
fs = 256
raw_ppg = mat['Subject14'].astype(np.float)
#raw_ppg = np.asarray(mat, dtype = float)
red = raw_ppg[:,7] #channel B
ir = raw_ppg[:,8] #channel B
spo2 = raw_ppg[:,9]

redppg = red[fs:-fs]
irppg = ir[fs:-fs]
t = np.linspace(0, len(red)/fs, len(red))
print(len(t))


# In[5]:


# Plotando um trecho do sinal

Tppg = 100
fig, ax = plt.subplots()
plt.plot(redppg[25000:Tppg * fs], linewidth=2, label='PPG RED ')
plt.ylabel('Amplitude [mV]')
plt.xlabel('Time [samples]')
plt.legend()
plt.grid()


# In[10]:


# Filtrando o sinal PPG


def butter_bandpass(lowcut, highcut, fs, max_order=15, filter_tol=1e-5):
    b, a = [], []
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    order = max_order
    not_finished = True
    while not_finished:
        b, a = butter(order, [low, high], btype="band", analog=False)
        # Check if filter is stable, if not keep iterating
        not_finished = any(np.abs(np.roots(a)) > 1 - filter_tol)
        # Decrease order
        order = order - 1
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, max_order=order)
    if ZERO_PHASE_FILTERS:
        y = filtfilt(b, a, data)  # zero-phase
    else:
        y = lfilter(b, a, data)
    return y


def butter_design(cutoff, fs, max_order=12, btype="low", filter_tol=1e-5):
    b, a = [], []
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    order = max_order
    not_finished = True
    while not_finished:
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        # Check if filter is stable, if not keep iterating
        not_finished = any(np.abs(np.roots(a)) > 1 - filter_tol)
        # Decrease order
        order = order - 1
    return b, a


def butter_filter(data, cutoff, fs, order, btype="low"):
    b, a = butter_design(cutoff, fs, btype=btype, max_order=order)
    if ZERO_PHASE_FILTERS:
        y = filtfilt(b, a, data)  # zero-phase
    else:
        y = lfilter(b, a, data)
    return y


# In[]:
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

fig, ax = plt.subplots()
start = 0
end = 5000
plt.plot(
    t[start * fs : end * fs], ir_bp[start * fs : end * fs], "g-", linewidth=2, label="ir bp"
)
plt.plot(
    t[start * fs : end * fs],
    ir[start * fs : end * fs] - np.mean(ir[start * fs : end * fs]),
    "k-",
    linewidth=2,
    label="original (zero-mean)",
)
# plt.ylim((-0.05, +0.05))
plt.xlabel("Time [sec]")
plt.minorticks_on()
plt.grid(True, which="major", alpha=0.8)
plt.grid(True, which="minor", alpha=0.3)
plt.legend()
plt.show()

#%%

duration=int(444*fs)

n_segms = int(np.ceil(2*(len(ir_hp)/duration)-1))

for i in range(n_segms):
    tempBuf = ir_hp[i:i+duration]

plt.plot(tempBuf)
plt.show()
plt.plot(ir_hp)


# In[8]:

# Segmentar os sinais de PPG em intervalos X

# Definir o deltaT
deltaT = round(0.8 * fs) #duration=int(windowLen*fs) // duration for the window segm
n_win = np.ceil(len(ir_bp)/deltaT).astype('int') #n_segms = int(np.ceil(len(signal)/duration))


# # Recorte do sinal em deltaT

pad_length = int(n_win * deltaT - len(ir_bp))
ppg_ir = np.append(ir_bp, np.zeros([pad_length]))

# Recorte do sinal em deltaT usando ordem F (Matlab/Fortran)

crop_ppg = ppg_ir.reshape((deltaT, -1), order='F') #new x axys

for i in range(n_win):
    col = crop_ppg[i:i+deltaT]
    #redppg_s = redppg[p:p+w-1]
    #col = crop_ppg[:, i]
    #irppg_s = irppg[i:i+fs-1]

plt.plot(col[0])
# for i in range(0, duration):
#     temp = [data[i:i+duration]]

# freqs = np.linspace(0, fs/2, num=len(irppg_s))
# # for i in range(n_win):
# #     col = crop_ppg[:, i]
# #     t_f = fftshift(fft(col, deltaT))
# #     tf = (np.abs(t_f))
# #     # mean_psd += tf
# #     # psd_ppg[:, i] = tf   

# # tf_ir = tf[:int(np.floor(len(tf)/2))]
# # f = np.linspace(0, fs/2, num = len(tf_ir))

# # freqs = np.linspace(0, fs / 2, num=len(tf_ir))

# fig, ax = plt.subplots()
# plt.figure(100)
# plt.plot(freqs,irppg_s, "b-", linewidth=1, label="FTT IR original")
# plt.ylabel("Amplitude [mV]")
# plt.xlabel("Frequencia [Hz]")
# plt.grid()

#%%

# Criando Figura e Eixos
fig, ax = plt.subplots()
plt.plot(freqs, tf_ir, 'k--',
         label='Mean PSD, win={0:.2f}s'.format(deltaT / fs), linewidth=3)
locs, _ = plt.yticks()
ax.fill_between(freqs, tf_ir, locs[0], color='r', alpha=0.15)
plt.xlim([-1, 1 + fs / 2])
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.legend()
plt.minorticks_on()
plt.grid(True, which='major')
plt.grid(True, which='minor', alpha=0.3)
plt.show()

# %%


# raise SystemExit(0) #HACK: DEBUG

# shiftLen = tempo de deslocamento em segundos
# dataOverlap = calcula o número de amostras com base na taxa de amostragem

#janela deslizante => tem que descartar as amostras 
# overlap de metade da janela = percentual de overlap de 50%....99%???
# não ter efeito de borda
# fazer 512 e faz a transformada, a próxima janela de tamanho 512 
# joga fora os primeiros 256, desloca a  origem pro 257 e enche a nova janela da metade pra frente. 
# overlap, deslizar a janela entre transformadas
# amostras de 0 511 estima a saturação a próxima joga fora do 0 a 255 e pega a 256 a 511+767 

def buffer(data,duration,dataOverlap):
    numberOfSegments = int(np.ceil((len(data)-dataOverlap)/(duration-dataOverlap)/3))
    print(data.shape)
    tempBuf = [data[i:i+duration] for i in range(0,len(data),(duration-int(dataOverlap)))]
    tempBuf[numberOfSegments-1] = np.pad(tempBuf[numberOfSegments-1],(0,duration-tempBuf[numberOfSegments-1].shape[0]),'constant')
    tempBuf2 = np.vstack(tempBuf[0:numberOfSegments])
    print(numberOfSegments)
    return tempBuf2
	

sampleRate=254
windowLen=600
shiftLen=45

duration=int(windowLen*sampleRate)
dataOverlap = (windowLen-shiftLen)*sampleRate
# Example Sin Waveform for 4s long and 10 Hz
t = np.linspace(0, len(redppg)/fs, len(redppg))
data=red_bp
#print("python main function")

#%%
winlen = fs*2 #512
stepsize = winlen // 2

idxlist = range(0, len(ir_bp)-winlen, stepsize)
C = np.zeros((len(idxlist), winlen))

for o, i in enumerate(idxlist):
    C[o, :] = fftshift(fft(ir_bp[i:i+winlen]))

# In[ ]:

#Segmentar os sinais de PPG em intervalos X
p = 0
w = 512     #janela

max_p = 2*(len(redppg)/w) - 1

for p in range(max_p-1):
    redppg_s = redppg[p:p+w-1]
    irppg_s = irppg[p:p+w-1]

duration=int(windowLen*sampleRate)
n_segms = int(np.ceil(len(signal)/duration))

for i in range(0, duration):
    temp = data[i:i+duration]
# In[ ]:

#Dividir os sinais de PPG em intervalos de ΔT

# Definir o deltaT
deltaT = round(0.75 * fs) #192

# Recorte do sinal em deltaT
n_win = np.ceil(len(red_bp) / deltaT).astype('int')
pad_length = int(n_win * deltaT - len(red_bp))
red_bp = np.append(red_bp, np.zeros([pad_length]))

# Recorte do sinal em deltaT
# usando ordem F (Matlab/Fortran)
crop_red_ppg = red_bp.reshape((deltaT, -1), order='F')


# In[ ]:


## Matriz para armazenar PSD de cada janela
psd_ecg = np.zeros(crop_red_ppg.shape)

# PSD
mean_psd = np.zeros([deltaT])

for i in range(n_win):
    col = crop_red_ppg[:, i]
    t_f = fftshift(fft(col, deltaT))
    tf = 10 * np.log10(np.abs(t_f)**2)
    mean_psd += tf
    psd_ecg[:, i] = tf

# Calculamos o valor médio (cortamos as freqs negativas)
mean_psd = fftshift(mean_psd / n_win)
mean_psd = mean_psd[:int(np.floor(len(mean_psd) / 2))]
freqs = np.linspace(0, fs / 2, num=len(mean_psd))
#%%
# pode ser que esse de certo

sampleRate=256
windowLen=2
tempo_desloc=1
duration=int(windowLen*sampleRate)
dataOverlap = (windowLen-tempo_desloc)*sampleRate

print(dataOverlap)

data=ir_bp

numberOfSegments = int(np.ceil((len(data)-dataOverlap)/(duration-dataOverlap)/3))
print(data.shape)
print(numberOfSegments)

tempBuf = [data[i:i+duration] for i in range(0,len(data),(duration-int(dataOverlap)))]

tempBuf[numberOfSegments-1] = np.pad(tempBuf[numberOfSegments-1],
    (0,duration-tempBuf[numberOfSegments-1].shape[0]),'constant')
tempBuf2 = np.vstack(tempBuf[0:numberOfSegments])


#raise SystemExit(0) #HACK: DEBUG
# %%
#%%

ppg = ir_dc_cheby2
l_n = []
for i in range(len(ppg)):
    if ppg[i] < 0:
        a = ppg[i]
        l_n.append(a)

l_neg = np.array(l_n)
# Initial Parameters
WIN_DURATION = 900e-3
OFF_WIN_DURATION = 250e-3
MASK_WIN_DURATION = 120e-3

# Threshold Functions Valley
updateThresholdn = (
    lambda th_prev, idx: (th_prev + np.min(l_neg[idx]) + (np.max(l_neg[idx]) - np.min(l_neg[idx])) / 2.7) / 2
)
offThresholdn = lambda th: 0.25 * th

win = np.arange(fs)  # first search window
win_step = int(WIN_DURATION * fs)  # window step
force_step = False

# Threshold Lists
lthn = [np.min(l_neg[win]) + (np.min(l_neg[win]) + np.max(l_neg[win])) / 1.5]  # threshold list
lnnn = [0]
lnnn_off = [0]
lnnn_th = [lthn[-1]]

while True:
    # Update Window for next search
    win_start = lnnn[-1] + win_step + int(force_step) * win_step
    force_step = False
    win = np.arange(win_start, win_start + win_step)

    # Adjust Window
    if win[-1] > len(l_neg):
        win = win[win < len(l_neg)]
    # Break (no more samples) :)
    if not win.size:
        break

    # Update Threshold
    lthn.append(updateThresholdn(lthn[-1], win))

    # Find next Burst
    lnn = l_neg[win]
    lnn[lnn <= lthn[-1]] = 0
    lnn = np.where(lnn)[0]
    # Short Time Rejection
    if lnnn[-1] != 0:  # only mask if this is not the first iteration...
        lnn = lnn[lnn > int(MASK_WIN_DURATION * fs)]
    # Nothing above the threshold
    if not lnn.size:  # force a step and continue ...
        force_step = True
        continue
    # Append to list of Coarse Burst Limits
    lnnn.append(lnn[0] + win[0] - 1)
    # Append to Threshold Values
    lnnn_th.append(lthn[-1])

    # Find burst end
    if lnnn[-2] != 0:  # excluded first iteration
        #  lnn[-2] bc lnn was updated 6 lines above
        win_off = win - win_step  # take a step back
        win_off = np.arange(win_off[0], win_off[0] + int(OFF_WIN_DURATION * fs))  # crop window
        win_off = win_off[win_off < len(l_neg)]  # crop on borders
        # Locate Burst End
        lnn_off = l_neg[win_off]
        lnn_off[lnn_off <= offThresholdn(lthn[-1])] = 0
        lnn_off = np.where(lnn_off)[0]
        # Append to List
        try:
            lnnn_off.append(win_off[0] + ln_off[-1] - 1)
        except IndexError:
            # print('invalid ln_off!')
            pass

# Remove first element in threshold indexes
lnnn = np.array(lnnn[1:])
lnnn_off = np.array(lnnn_off[1:])
lnnn_th = np.array(lnnn_th[1:])




#%%
if PLOTS:
    fig, ax = plt.subplots()
    plt.plot(l_neg, "k", label="PPG Mask")
    plt.plot(lnnn, lnnn_th, "r", label="Threshold")
    plt.stem(lnnn, l_neg[lnnn], label="ON")  # type: ignore
    # plt.stem(lnn_off, l[lnn_off], label='OFF') # type: ignore
    # plt.show()
    # plt.plot(l, "b", label="PPG Mask")
    # plt.plot(lnn, lnn_th, "r", label="Threshold")
    plt.minorticks_on()
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.6)
    plt.legend()
    # s(idx).PI = (s(idx).pks - s(idx).vls)./s(idx).dc(s(idx).plocs);

#%%

# th = np.interp(np.arange(0, len(ppg)), lnn, lnn_th)
thn = np.interp(np.arange(0, len(ppg)), lnnn, lnnn_th)
PKS_DIST = np.ceil(300e-3 * fs)
# Peaks above threshold
p_pks, _ = find_peaks(ppg, height=(+th, None), distance=PKS_DIST)
# Valleys below threshold
n_pks, _ = find_peaks(-ppg, height=(thn, None), distance=PKS_DIST)  # type:ignore
pks = np.sort(np.concatenate((p_pks, n_pks)))

# ac_dc = ppg[p_pks]-ppg[n_pks]/

# zeros = np.zeros()


#%%


#%%
plt.plot(t, ppg, "k", label="PPG")
plt.stem(t[pks], l, label="Peaks")  # type: ignore


list_pks = [-0.17780045, -0.46764584, -0.2,  0.59981918, 0.60001265, 0.7, -0.20320578, 0.23450197]

l = []

l.append(list_pks[0])

for pos in range(len(list_pks)-1): #and not found and not stop
      
    if list_pks[pos+1] < 0:
        if list_pks[pos] < 0:
            if list_pks[pos+1] < list_pks[pos]: #anterior maior que zero
                a = list_pks[pos+1] 
                a = 0.0
                l.append(a) #do something to remove the index of the atual position
            else: 
                l.append(list_pks[pos+1])        
        else: 
            l.append(list_pks[pos+1])
    else: # list_pks[pos+1] > 0: 
        if list_pks[pos] > 0:
            if list_pks[pos+1] < list_pks[pos]:
                a = list_pks[pos+1] 
                a = 0.0 #do something to remove the index of the atual position             
                l.append(a)     
            else:            
                a = list_pks[pos+1] 
                a = 0.0 #do something to remove the index of the atual position             
                l.append(a) 
        else:
            l.append(list_pks[pos+1])
            
#%%

def busca_sequencial( seq, x):
    
    for i in range(len(seq)):
        if seq[i] == x:
            return True
    return False

seq = [4, 10, 80, 90, 91, 99, 100, 101] 
testes = [80, 50, 90, 15, 99] #l_neg e l_pos
idx = [2, 3, 5, 6, 7] #pks

seq = np.array(seq)
idx = np.array(idx)
testes = np.array(testes)

x = seq[idx] #ppg[pks] 

#pks = [105 175 600]
#ppg[pks] = [-0.176, 0.769, -0.12]
#l_neg e l_pos = -0,176, 0.542, -0.3


for i in range(len(testes)):
    pos = busca_sequencial(x, testes[i])
    if pos is False:
        print("Nao achei ", i)
    else:
        print("Achei ", i)

# for i in range(len(x)):
#     pos = busca_sequencial(seq, x[i])
#     if pos is False:
#         print("Nao achei ", i)
#     else:
#         print("Achei ", i)


# for j in range(len(x)):

b = []
for i in range(len(x)):
    if seq[idx][i] == testes[i:i+len(testes)]:
        a = seq[idx][i]
        b.append(a)
    else:
        i = i + 1
        
# %%
tests = [80, 50, 90, 15, 99]
if x in tests:
    index = tests.index(x, [0,[len(tests)]])
    print(index)

#%%
for k,w in enumerate(idx):
    if seq[idx][i] == testes[i:i+len(testes)]:
        a = seq[idx][i]
        b.append(a)
    else:
        i = i + 1
#%%

seq = [4, 10, 80, 90, 91, 99, 100, 101] 
testes = [80, 50, 90, 99] #l_neg e l_pos
idx = [2, 3, 5, 6, 7] #pks

seq = np.array(seq)
idx = np.array(idx)
testes = np.array(testes)

x = seq[idx] #ppg[pks] 

#pks = [105 175 600]
#ppg[pks] = [-0.176, 0.769, -0.12]
#l_neg e l_pos = -0,176, 0.542, -0.3

# b = []
# for k,w in enumerate(idx):
#     r = seq[w]
#     b.append(r)
#     if b[k] == seq[w]:
#         print(w)
# crop_ecg = np.zeros([2 * crop_win, len(R_w)])
idx = [2, 3, 5, 6, 7] #pks
b = []
for k,w in enumerate(idx):
    if  x[k] == testes[k]:
        print(w)
    else: 
        if x[k] == testes[k+1]:
            print(w)
        elif x[k] == testes[k-1]:
            print(w)
        else:
            x[k] = x[k-1]
            
        
# %%

seq = [-0.17768, -0.17780045, -0.46764584,  0.59981918, 0.8743] 
tests = [-0.17768, -0.46764584, 0.59981918] #l_neg e l_pos
idx = [0, 1, 2, 3] #pks 6,7,73,105

seq = np.array(seq)
idx = np.array(idx)

# [-0.17768, -0.17780045, -0.46764584,  0.59981918]
x = seq[idx] #ppg[pks]


for i in range(len(x)):
    index = tests.index(x[i])
    print(index)
# %%
def orderedsearch(data, alist, item):
    
    for i, w in enumerate(alist): #range(len(alist)):
        if data[w] == item:
            return w
    # while pos < len(alist) and not found and not stop:
    #     if alist[pos] == item:
    #         found = True
    #     else:
    #         if alist[pos] > item:
    #             stop = True
    #         else:
    #             pos = pos+1
    return None    
seq = [4, 10, 80, 90, 91, 99, 100, 101] 
testes = [80, 50, 90, 99] #l_neg e l_pos
idx = [2, 3, 5, 6, 7] #pks
# x = [80, 90, 99, 100, 101]

seq = np.array(seq)
idx = np.array(idx)
testes = np.array(testes)

# x = seq[idx] #ppg[pks] 
y = []

for i in range(len(testes)):
    pos = orderedsearch(seq, idx, testes[i])
    if pos is not None:
        y.append(pos)
    else:
        None
# %%
x = ir_dc_cheby2

thresh = 0.12 * (abs(min(x)) + abs(max(x)))
distance = 0.25*fs

# Find indices of peaks
peaks_idx, _ = find_peaks(x, height=(-thresh,thresh), distance=distance)

# Find indices of valleys (from inverting the signal)
valley_idx, _ = find_peaks(-x, height=(-thresh,thresh), distance=distance)#(-(abs(max(x))),-thresh)

#FIX LEN PEAKS AND VALLEYS ?????????????? 

# Plot signal
plt.plot(x)
plt.plot(peaks_idx, x[peaks_idx], "x")
plt.plot(valley_idx, x[valley_idx], 'b.')
plt.grid()

print(len(peaks_idx))
print(thresh)
print(len(valley_idx))
print(len(peaks_idx)-len(valley_idx))

#%%

b = []
for k,w in enumerate(pks):
    if  ppg[w] == x[k]:
        print(w)
    else: 
        if x[k] == testes[k+1]:
            print(w)
        elif x[k] == testes[k-1]:
            print(w)
        else:
            x[k] = x[k-1]


list_pks = ppg[pks]

l = []

l.append(list_pks[0])

for pos in range(len(list_pks)-1):
      
    if list_pks[pos+1] < 0:
        if list_pks[pos] < 0:
            if list_pks[pos+1] < list_pks[pos]: #previous greater than zero
                a = list_pks[pos+1] 
                a = 0.0
                l.append(a) #do something to remove the index of the current position
            else: 
                l.append(list_pks[pos+1])        
        else: 
            l.append(list_pks[pos+1])
    else: # list_pks[pos+1] > 0: 
        if list_pks[pos] > 0:
            if list_pks[pos+1] < list_pks[pos]:
                a = list_pks[pos+1] 
                a = 0.0 #do something to remove the index of the current position             
                l.append(a)     
            else:            
                a = list_pks[pos+1] 
                a = 0.0 #do something to remove the index of the current position             
                l.append(a) 
        else:
            l.append(list_pks[pos+1])
    
lpev = np.array(l) 


#  s(idx).PI = (s(idx).pks - s(idx).vls)
#  /s(idx).dc(s(idx).plocs)
#%%
def orderedsearch(data, alist, item):
    
    for i, w in enumerate(alist): #range(len(alist)):
        if data[w] == item:
            return w
    # while pos < len(alist) and not found and not stop:
    #     if alist[pos] == item:
    #         found = True
    #     else:
    #         if alist[pos] > item:
    #             stop = True
    #         else:
    #             pos = pos+1
    return None    
alis = pks
y = []

for i in range(len(x)):
    pos = orderedsearch(ppg, pks, x[i])
    if pos is not None:
        y.append(pos)
    else:
        None
# lis = [1, 2] 
# n = np.array(lis) 
# p = [0.1, 0.2, 0.3]
# q = [0.3, 0.2, 0.4]

# o = np.array(p)
# print(o[n])
# r = []


# for i in range(len(p)):
#     if q == o[n][i]:
#         a = n
#         r.append(a)
#%%
y = np.array(y)
acdc = x-l_neg/(y) ##localização dos picos aaaaa 


# %%

#%%

winlen = fs
stepsize = winlen // 2

idxlist = range(0, len(ir_bp)-winlen, stepsize)
C = np.zeros((len(idxlist), winlen))

for o, i in enumerate(idxlist):
    C[o, :] = fftshift(fft(ir_bp[i:i+winlen]))



if PLOTS:
    fig, ax = plt.subplots()
    start = 0
    end = 5000
    plt.plot(
        t[start * fs : end * fs], ir_bp[start * fs : end * fs], linewidth=2, label="ir bp"
    )
    plt.plot(
        t[start * fs : end * fs], C[0],
        "k-",
        linewidth=2,
        label="FFT",
    )
    # plt.ylim((-0.05, +0.05))
    plt.xlabel("Time [sec]")
    plt.minorticks_on()
    plt.grid(True, which="major", alpha=0.8)
    plt.grid(True, which="minor", alpha=0.3)
    plt.legend()
    plt.show()
