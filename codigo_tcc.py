def find_peak_valley_2(sample, time, threshold):

    i = 1
    num_upsteps = 0
    possible_peak = False
    possible_valley = False
    possible_min = True
    value_possible_peak = 0
    time_possible_peak = 0
    value_possible_valley = 0
    time_possible_valley = 0
    value_peak = 0
    value_valley = 0
    time_valley = 0
    time_peak = 0

    while (i< len(sample)):
        if i>= 2:
            if (sample[i]>sample[i-1]):
                num_upsteps = num_upsteps +1 
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
                    if (possible_min == True):
                        if (sample[i] <= value_possible_valley): 
                            value_possible_valley = sample[i]
                            time_possible_valley = time[i]
                    if (possible_peak == True):
                        if (sample[i-1]>value_possible_peak):
                            value_peak = sample[i-1]
                            time_peak = time[i-1]
                        else:
                            value_peak = value_possible_peak
                            time_peak = time_possible_peak
                        if (possible_valley == True):
                            value_valley = value_possible_valley
                            time_valley = time_possible_valley
                            possible_valley = False
                        threshold = 0.6 * num_upsteps
                num_upsteps = 0
        i = i + 1

    return value_peak, value_valley

