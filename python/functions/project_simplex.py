# Projection of a D-dimension vector into the probability simplex
import numpy as np

def project_simplex(Y):
    n = Y.size
    X = np.array(sorted(Y, reverse = True))
    Xtemp = (np.cumsum(X) - 1) * (1/np.arange(1, n+1))
    e = Xtemp[np.sum(X > Xtemp) - 1]
    f = Y - e
    X = np.maximum(f, 0)
    return X
        
        
        