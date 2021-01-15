# Compute standard deviation
import numpy as np

def compute_std_obj(obj_list, history_size):
    std_obj = np.inf
    if len(obj_list) >= history_size:
        std_obj = np.std(obj_list[(len(obj_list)-history_size):])
        
        std_obj = std_obj/np.abs(obj_list[-1])
    return std_obj