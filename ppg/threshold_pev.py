import numpy as np

def adaptative_th(data, fs):
    # PPG Envelope for Adaptive Thresholding
    l = np.sqrt(data ** 2)  # type: ignore
    # Initial Parameters
    WIN_DURATION = 900e-3
    OFF_WIN_DURATION = 250e-3
    MASK_WIN_DURATION = 120e-3 #900, 250, 120

    # Threshold Functions
    updateThreshold = (
        lambda th_prev, idx: (th_prev + np.min(l[idx]) + (np.max(l[idx]) - np.min(l[idx])) / 2.8) / 2 ##2.4) / 1.94
    )
    offThreshold = lambda th: 0.25 * th
    # Window
    win = np.arange(fs)  # first search window
    win_step = int(WIN_DURATION * fs)  # window step
    force_step = False
    # Threshold Lists
    lth = [np.min(l[win]) + (np.min(l[win]) + np.max(l[win])) / 4]  # threshold list 4, 3.4
    lnn = [0]
    lnn_off = [0]
    lnn_th = [lth[-1]]

    while True:
        # Update Window for next search
        win_start = lnn[-1] + win_step + int(force_step) * win_step
        force_step = False
        win = np.arange(win_start, win_start + win_step)

        # Adjust Window
        if win[-1] > len(l):
            win = win[win < len(l)]
        # Break (no more samples) 
        if not win.size:
            break

        # Update Threshold
        lth.append(updateThreshold(lth[-1], win))

        # Find next Burst
        ln = l[win]
        ln[ln <= lth[-1]] = 0
        ln = np.where(ln)[0]
        # Short Time Rejection
        if lnn[-1] != 0:  # only mask if this is not the first iteration...
            ln = ln[ln > int(MASK_WIN_DURATION * fs)]
        # Nothing above the threshold
        if not ln.size:  # force a step and continue ...
            force_step = True
            continue
        # Append to list of Coarse Burst Limits
        lnn.append(ln[0] + win[0] - 1)
        # Append to Threshold Values
        lnn_th.append(lth[-1])

        # Find burst end
        if lnn[-2] != 0:  # excluded first iteration
            #  lnn[-2] bc lnn was updated 6 lines above
            win_off = win - win_step  # take a step back
            win_off = np.arange(win_off[0], win_off[0] + int(OFF_WIN_DURATION * fs))  # crop window
            win_off = win_off[win_off < len(l)]  # crop on borders
            # Locate Burst End
            ln_off = l[win_off]
            ln_off[ln_off <= offThreshold(lth[-1])] = 0
            ln_off = np.where(ln_off)[0]
            # Append to List
            try:
                lnn_off.append(win_off[0] + ln_off[-1] - 1)
            except IndexError:
                # print('invalid ln_off!')
                pass

    # Remove first element in threshold indexes
    lnn = np.array(lnn[1:])
    lnn_off = np.array(lnn_off[1:])
    lnn_th = np.array(lnn_th[1:])
    
    return l, lnn, lnn_off, lnn_th

def find_peak_valley(data,peakOrValley):
    index=[]
    extreme=[]
    npts = 100
    i = 0
    if (peakOrValley == 1):
        while (i < npts):
            if(max(data[0:i+npts]) == data[i]):
                index.append(i)
                extreme.append(data[i])
                i = i+npts
            else:
                i = i+1
        while (i < len(data)-npts):
            if(max(data[i-npts:i+npts]) == data[i]):
                index.append(i)
                extreme.append(data[i])
                i = i+npts
            else:
                i = i+1
        while (i < len(data)):
            if(max(data[i-npts:len(data)]) == data[i]):
                index.append(i)
                extreme.append(data[i])
                i = i+npts
            else:
                i = i+1
        return index,extreme
    elif (peakOrValley == -1):
        while (i < npts):
            if(min(data[0:i+npts]) == data[i]):
                index.append(i)
                extreme.append(data[i])
                i = i+npts
            else:
                i = i+1 
        while (i < len(data)-npts):
            if(min(data[i-npts:i+npts]) == data[i]):
                index.append(i)
                extreme.append(data[i])
                i = i+npts
            else:
                i = i+1
        while (i < len(data)):
            if(min(data[i-npts:len(data)]) == data[i]):
                index.append(i)
                extreme.append(data[i])
                i = i+npts
            else:
                i = i+1
        return index,extreme
    else:
        print("error!")
        return index, extreme


def fix_pev(data,pev):
    
    list_pks = data[pev]

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
    return lpev 

def loc_pev(lpev):
    l_neg = []
    l_pos = []

    for i in range(len(lpev)):
        if lpev[i] < 0:
            b = lpev[i]
            l_neg.append(b)
        elif lpev[i] > 0:
            b = lpev[i]
            l_pos.append(b)
        else: 
            None

    l_neg = np.array(l_neg)
    l_pos = np.array(l_pos)
    x = np.concatenate((l_pos, l_pos[len(l_pos)-(len(l_neg)-len(l_pos)):len(l_pos)]))
    return l_neg, l_pos, x

