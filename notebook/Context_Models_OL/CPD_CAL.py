import numpy as np

## Calculation of Change points (DS)
def cpd_cal(frequency, maxes, trial_nos, cut_off):
    
    cpd_temp=trial_nos - maxes[0:len(maxes)-1]
    records_array = cpd_temp[cpd_temp>cut_off]
    idx_sort = np.argsort(records_array)
    sorted_records_array = records_array[idx_sort]
    cps, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    cps = cps[count > frequency]
    
    return cps,count