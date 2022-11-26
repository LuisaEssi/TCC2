#%%


import numpy as np
from scipy.signal import butter, cheby2, lfilter, filtfilt
import matplotlib.pyplot as plt 
# 

ZERO_PHASE_FILTERS = True

#%%
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


def butter_design(cutoff, fs, max_order=15, btype="low", filter_tol=1e-5):
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

###-------------------- Cheby type 2 -------------------###

def cheby2_bandpass(lowcut, highcut, fs, rs=40, max_order=12, filter_tol=1e-5):
    b, a = [], []
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    order = max_order
    not_finished = True
    while not_finished:
        b, a = cheby2(order, rs, [low, high], btype='bandpass', analog=False)
        # Check if filter is stable, if not keep iterating
        not_finished = any(np.abs(np.roots(a)) > 1 - filter_tol)
        # Decrease order
        order = order - 1
    return b, a


def cheby2_bandpass_filter(data, lowcut, highcut, fs, rs, order):
    b, a = cheby2_bandpass(lowcut, highcut, fs, rs, max_order=order)
    if ZERO_PHASE_FILTERS:
        y = filtfilt(b, a, data)  # zero-phase
    else:
        y = lfilter(b, a, data)
    return y


def cheby2_design(cutoff, fs, max_order=12, rs=40, btype="low", filter_tol=1e-5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    order = max_order
    not_finished = True
    while not_finished:
        b, a = cheby2(order, rs, normal_cutoff, btype=btype, analog=False, output='ba')
         # Check if filter is stable, if not keep iterating
        not_finished = any(np.abs(np.roots(a)) > 1 - filter_tol)
        # Decrease order
        order = order - 1
    return b, a


def cheby2_filter(data, cutoff, fs, order, rs, btype="low"):
    b, a = cheby2_design(cutoff, fs, btype=btype, max_order=order, rs = rs)
    y = filtfilt(b, a, data)  # zero-phase
    return y


# from utils import SubjectPPGRecord

# MAT_FILE = "Subject14"
# sub = SubjectPPGRecord(MAT_FILE, db_path=".", mat=True)
# rec = sub.record
# fs = sub.fs
# red = rec.red_a
# ir = rec.ir_a
# spo2 = rec.spo2

# t = np.linspace(0, len(red) / fs, len(red))
# print(f"PPG Sinal contains {len(red)} samples and {len(red)/fs} sec. duration.")


# def butter_design(cutoff, fs, order, btype="low"):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype=btype, analog=False)
#     return b, a


# def butter_filter(data, cutoff, fs, order, btype="low"):
#     b, a = butter_design(cutoff, fs, btype=btype, order=order)
#     y = filtfilt(b, a, data)  # zero-phase
#     return y

# # FILTRO PASSA-BAIXA
# cutoff_lp = 8  # frequencia de corte
# LP_ORDER = 8

# red_lp = butter_filter(red, cutoff_lp, fs, order=LP_ORDER) # btype="low"
# ir_lp = butter_filter(ir, cutoff_lp, fs, order=LP_ORDER)

# # FILTRO PASSA-ALTA
# cutoff_dc = 0.2    # freq de corte pra remover sinal DC
# HP_ORDER = 8
# red_dc = butter_filter(red_lp, cutoff_dc, fs, order=HP_ORDER, btype="high")
# ir_dc = butter_filter(ir_lp, cutoff_dc, fs, order=HP_ORDER, btype="high")

# #PLOT REMOCAO DC
# fig, ax = plt.subplots(2)
# start = 0
# end = 50000
# ax[1].plot(
#     t[start * fs : end * fs], ir_dc[start * fs : end * fs], "b-", linewidth=1, label="filtered ir"
# )

# ax[0].plot(
#     t[start * fs : end * fs], red_dc[start * fs : end * fs], "r-", linewidth=1, label="filtered red"
# )
# plt.xlabel("Tempo [sec]")
# plt.ylabel("Amplitude [mV]")
# ax[1].grid()
# ax[1].legend()
# ax[0].grid()
# ax[0].legend()

# #PLOT SEM REMOCAO DC

# #%%

# fig, ax = plt.subplots()
# plt.plot(t[start * fs : end * fs], ir_lp[start * fs : end * fs], "b-", linewidth=2, label="Sinal IR filtrado")
# # plt.plot(t[start * fs : end * fs], red_lp[start * fs : end * fs], "r-", linewidth=1, label="filtered red")
# plt.plot(t[start * fs : end * fs], ir[start * fs : end * fs], "r-", linewidth=1, label="Sinal IR original")
# plt.xlabel("Tempo [sec]")
# plt.ylabel("Amplitude [mV]")
# plt.legend()
# plt.grid()

# #%%
# # ---------------------------------- CHEBY2 ------------------------------------------------------

# def cheby2_design(cutoff, fs, order, rs, btype="low"):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = cheby2(order, rs, normal_cutoff, btype=btype, analog=False, output='ba')
#     return b, a
  

# def cheby2_filter(data, cutoff, fs, order, rs, btype="low"):
#     b, a = cheby2_design(cutoff, fs, btype=btype, order=order, rs = rs)
#     y = filtfilt(b, a, data)  # zero-phase
#     return y

# # FILTRO PASSA-ALTA

# LP_ORDER = 4
# cutoff_lp = 8    # freq de corte pra remover sinal DC
#   # freq de corte pra remover sinal DC
# red_lp_cheby2 = cheby2_filter(red, cutoff_lp, fs, order=LP_ORDER, rs = 20) # btype="low"
# ir_lp_cheby2 = cheby2_filter(ir, cutoff_lp, fs, order=LP_ORDER, rs = 20)


# HP_ORDER = 6
# cutoff_dc = 0.2  
# red_dc_cheby2 = cheby2_filter(red_lp_cheby2, cutoff_dc, fs, order=HP_ORDER, rs=20, btype="high")
# ir_dc_cheby2 = cheby2_filter(ir_lp_cheby2, cutoff_dc, fs, order=HP_ORDER, rs=20, btype="high")

# fig, ax = plt.subplots()
# start = 0
# end = 50000
# plt.plot(
#     t[start * fs : end * fs], ir_lp_cheby2[start * fs : end * fs], "b-", linewidth=2, label="Sinal IR filtrado"
# )

# plt.plot(
#     t[start * fs : end * fs], ir[start * fs : end * fs], color="orange", linewidth=1, label="Sinal IR original"
# )
# plt.xlabel("Tempo [sec]")
# plt.ylabel("Amplitude [mV]")
# plt.legend()
# plt.grid()
# plt.legend()


#cutoff_lp = 8  # frequencia de corte
#cutoff_dc = 0.4
#
#BP_ORDER = 8
#
#red_bp_cheby2 = cheby2_bandpass_filter(red, cutoff_dc, cutoff_lp, fs, rs = 26, order=BP_ORDER)
#ir_bp_cheby2 = cheby2_bandpass_filter(ir, cutoff_dc, cutoff_lp, fs, rs = 26, order=BP_ORDER)
#
#red_lp_cheby2 = cheby2_filter(red, cutoff_lp, fs, order=BP_ORDER, rs = 26) # btype="low"
#ir_lp_cheby2 = cheby2_filter(ir, cutoff_lp, fs, order=BP_ORDER, rs = 26)
#
#red_dc_cheby2 = cheby2_filter(red_lp_cheby2, cutoff_dc, fs, order=BP_ORDER, rs=26, btype="high")
#ir_dc_cheby2 = cheby2_filter(ir_lp_cheby2, cutoff_dc, fs, order=BP_ORDER, rs=26, btype="high")
#
#red_lp = butter_filter(red, cutoff_lp, fs, order=BP_ORDER) # btype="low"
#ir_lp = butter_filter(ir, cutoff_lp, fs, order=BP_ORDER)


# # FILTRO PASSA-ALTA
# cutoff_dc = 0.2    # freq de corte pra remover sinal DC

# red_dc_cheby2 = cheby2_filter(red_lp_cheby2, cutoff_dc, fs, order=4, rs=20, btype="high")
# ir_dc_cheby2 = cheby2_filter(ir_lp_cheby2, cutoff_dc, fs, order=4, rs=20, btype="high")

# red_dc = butter_filter(red_lp, cutoff_dc, fs, order=4, btype="high")
# ir_dc = butter_filter(ir_lp, cutoff_dc, fs, order=4, btype="high")




# def cheby2_design(lowcut, highcut, fs, rs=40, max_order=12, filter_tol=1e-5):
#     b, a = [], []
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq

#     order = max_order

#     bands = [low,high]
#     mode = "band"

#     if low == 0:
#         bands = bands[1]
#         mode = "low"
#     if high == 0.5:
#         bands = bands[0]
#         mode = "high"
    
#     not_finished = True
#     while not_finished:
#         b, a = cheby2(order, rs, bands, btype=mode, analog=False)
#         # Check if filter is stable, if not keep iterating
#         not_finished = any(np.abs(np.roots(a)) > 1 - filter_tol)
#         # Decrease order
#         order = order - 1
#     return b, a

# def cheby2_filter(data, lowcut, highcut, fs, rs, order):
#     b, a = cheby2_design(lowcut, highcut, fs, rs, max_order=order)
#     if ZERO_PHASE_FILTERS:
#         y = filtfilt(b, a, data)  # zero-phase
#     else:
#         y = lfilter(b, a, data)
#     return y

# %%
