import numpy as np
import scipy.special as ssp

def gamma(x, a=4, scale=1):
    return np.exp(-x/scale)/(ssp.factorial(a-1)*scale**a) * x**(a-1)

def dergamma(x, a=4, scale=1):
    return ((a-1.)/x - 1./scale)*gamma(x,a,scale)


