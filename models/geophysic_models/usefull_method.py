import numpy as np
from scipy.linalg import expm

def exact_discretization(A, B, time) :
    lA,cA = np.shape(A)
    lB,cB = np.shape(B)
    M = np.vstack((np.hstack((A,B)), np.zeros(cA + cB)))
    MM = expm(M * time)
    Ad = MM[:lA,:cA]
    Bd = MM[:lB, cA:]
    return Ad, np.squeeze(Bd)