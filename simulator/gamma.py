import numpy as np
import scipy.special as ssp

def mygamma(x, a=4, scale=1, loc=0.):
    res = np.zeros_like(x)
    y = x-loc
    res [y>0] = np.exp(-y[y>0]/scale)/(ssp.factorial(a-1)*scale**a) * y[y>0]**(a-1)
    return res

def dergamma(x, a=4, scale=1, loc=0.):
    y = x-loc
    res = np.zeros_like(x)
    res [y>0] = ((a-1.)/y[y>0] - 1./scale)*mygamma(y[y>0],a,scale)
    return res




