# Project onto the shifted Lp ball
import numpy as np

def project_shifted_Lp_ball(x, shift_vec, p):
    shift_x = x - shift_vec
    normp_shift = np.linalg.norm(shift_x, ord=p)
    n = len(x)
    
    if normp_shift**p != n/2**p:
        xp = shift_x / (normp_shift/ (n**(1/p)/2) ) + shift_vec
    else:
        xp = x
    return xp