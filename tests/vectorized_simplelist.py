import numpy as np

""""
imagine two inputs with second based and minute based data both in same format
high_res=dict(time=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
low_res=dict(time=[3,5,8,11,14,16], values=[3,5,8,11,14,16]) . 

Assume i want to expand low_res to high_res, aligned based on time of high resolution.
no forward fill. Use previous value if no value available.
If no previous value use 0. Show the code.

"""

def expand_low_res_vectorized(high_res, low_res):
    indices = np.searchsorted(high_res["time"], low_res["time"], side='right') - 1
    
    print(indices)
    expanded_values = np.zeros(len(high_res["time"]))

    for i, idx in enumerate(indices):
        expanded_values[idx] = low_res['values'][i]

    # Forward fill
    valid_idx = 0
    for i in range(len(expanded_values)):
        if expanded_values[i] != 0:
            valid_idx = i
            break

    for i in range(valid_idx, len(expanded_values)):
        if expanded_values[i] == 0:
            expanded_values[i] = expanded_values[i - 1]

    return expanded_values

high_res=dict(time=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
            values=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
##result           [ 0.  0.  3.  3.  5.  5.  5.  8.  8.  8. 11. 11. 11. 14. 14. 16.]

low_res=dict(time=[3,5,8,11,14,16], values=[3,5,8,11,14,16])
# Expand low_res data using the corrected vectorized function
expand_low_res = expand_low_res_vectorized(high_res, low_res)
print(expand_low_res)