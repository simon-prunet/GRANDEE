from torch.utils.data import Dataset
import numpy as np
from simulator.gamma import dergamma, mygamma
import scipy.special as ssp

class Simulator(Dataset):

    def __init__(self, in_nc=3, n_samples=8192, a=4, 
                 sigma_range=[0.1,1.], amplitude_range=[0.1,10.],
                 scale_range=[1.,30.],loc_range=[100,8000],
                 length=100000):


        self.a = a # Gamma function derivative parameter
        self.in_nc = in_nc # Number of channels
        self.n_samples = n_samples # Number of samples per channel
        self.sigma_range = sigma_range
        self.amplitude_range = amplitude_range
        self.scale_range = scale_range
        self.loc_range = loc_range
        self.length = length # Because datasets must have a length

    def __len__(self):
        return self.length

    def __getitem__(self, idx, sigma_in=None):

        loc = np.random.uniform(low=self.loc_range[0],high=self.loc_range[1])
        scale = np.random.uniform(low=self.scale_range[0],high=self.scale_range[1])
        amplitudes = np.random.uniform(low=self.amplitude_range[0],high=self.amplitude_range[1],size=self.in_nc)
        # Typical peal
        # position of maximum of derivative dergamma (with loc=0)
        # This is to make sure that SNR=1 at peak if sigma=amplitude=1
        x1 = scale*(self.a-1 - np.sqrt(self.a-1))
        peak_scale = mygamma(np.array([x1,]), a=self.a, scale=scale)/scale / (np.sqrt(self.a-1)-1)

        if sigma_in is not None:
            sigma = sigma_in * peak_scale
        else:
            sigma = np.random.uniform (low=self.sigma_range[0],high=self.sigma_range[1]) * peak_scale


        traces = np.zeros((self.in_nc, self.n_samples))
        noisy  = np.zeros((self.in_nc, self.n_samples))
        for i in range(self.in_nc):
            traces[i,:] = dergamma(np.arange(1,self.n_samples+1,dtype=float), a=self.a, scale=scale,loc=loc)*amplitudes[i]
            noisy[i,:]= traces[i,:] + sigma*np.random.standard_normal(self.n_samples)

        return (sigma, noisy,traces)
    