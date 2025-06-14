import numpy as np
import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import scipy as sp
from typing import Dict, Any, List, Tuple
import warnings
import time
warnings.filterwarnings('ignore')

from apply_rfchain import open_event_root, percieved_theta_phi, get_leff, efield_2_voltage
from input_script import *
import models.basicblock as B
from PRUNetwrap import *

def create_simple_bandpass_operator():
    """Create a perfect bandpass filter forward operator"""
    
    class SimpleBandpassOp:
        def __init__(self, fs=2e9, lowcut=10e6, highcut=200e6):
            self.fs = fs
            self.lowcut = lowcut
            self.highcut = highcut
            print(f"Created perfect bandpass filter: {lowcut/1e6:.0f}-{highcut/1e6:.0f} MHz")
        
        def forward(self, signal):
            """Apply perfect bandpass filter in frequency domain"""
            filtered = np.zeros_like(signal)
            for ant in range(signal.shape[0]):
                for pol in range(signal.shape[1]):
                    signal_fft = sp.fft.rfft(signal[ant, pol, :])
                    freqs = sp.fft.rfftfreq(signal.shape[2], 1/self.fs)
                    mask = (freqs >= self.lowcut) & (freqs <= self.highcut)
                    signal_fft[~mask] = 0
                    filtered[ant, pol, :] = sp.fft.irfft(signal_fft, n=signal.shape[2])
            return filtered
        
        def adjoint(self, signal):
            return self.forward(signal)
    
    return SimpleBandpassOp()


def bandlimit_signal(signal, low_freq=10e6, high_freq=200e6, fs=2e9):
    """Apply ideal bandpass filter"""
    signal_fft = sp.fft.rfft(signal, axis=-1)
    freqs = sp.fft.rfftfreq(signal.shape[-1], 1/fs)
        
    bandpass_mask = (freqs >= low_freq) & (freqs <= high_freq)
    signal_fft[..., ~bandpass_mask] = 0
        
    return sp.fft.irfft(signal_fft, n=signal.shape[-1], axis=-1)



def wiener_filter_reconstruction(y, A, signal_power_est=None, noise_power_est=None):
    x_wiener = np.zeros_like(y)
    
    for ant in range(y.shape[0]):
        for pol in range(y.shape[1]):
            n_samples = y.shape[2]
            
            
            freqs = sp.fft.rfftfreq(n_samples, 1/2e9)  
            H = np.zeros_like(freqs, dtype=complex)
            mask = (freqs >= 10e6) & (freqs <= 200e6)  
            H[mask] = 1.0
            
            Y = sp.fft.rfft(y[ant, pol, :])
            
            if signal_power_est is None:
                if np.any(mask):
                    signal_power = np.mean(np.abs(Y[mask])**2)
                else:
                    signal_power = np.mean(np.abs(Y)**2)
            else:
                signal_power = signal_power_est
                
            if noise_power_est is None:
                noise_mask = ~mask
                if np.any(noise_mask):
                    noise_power = np.mean(np.abs(Y[noise_mask])**2)
                else:
                    noise_power = 0.1 * signal_power
            else:
                noise_power = noise_power_est
            
            noise_power = max(noise_power, 1e-12)
            
            H_conj = np.conj(H)
            H_mag_sq = np.abs(H)**2
            
            wiener_filter = (H_conj * signal_power) / (H_mag_sq * signal_power + noise_power)
            
            X_wiener = wiener_filter * Y
            x_wiener[ant, pol, :] = sp.fft.irfft(X_wiener, n=n_samples)
    
    return x_wiener

def pnp_hqs_with_prunet(y, A, denoiser, max_iter=8, verbose=True):
   
    x = A.adjoint(y)
    z = x.copy()
    
    if verbose:
        print(f"PnP HQS with PRUNet ({max_iter} iterations)")
        print(f"Initial amplitude: {np.max(np.abs(x)):.4f}")
    
    for k in range(max_iter):

        sigma_k = 1.0 * (0.01 / 1.0) ** (k / (max_iter - 1))
        mu_k = 1.0 / (sigma_k ** 2)
        
        prunet_sigma = np.clip(sigma_k, 0.01, 1)
        
        for cg_iter in range(10):
            residual_data = A.forward(x) - y
            gradient = A.adjoint(residual_data) + mu_k * (x - z)
            step_size = 0.01 / (1 + mu_k)
            x = x - step_size * gradient
        

        z = denoiser.denoise(x, prunet_sigma)
    
    
    return x

def test_with_wiener_comparison(model_path):
    """Test PnP-HQS against Wiener filter benchmark with a simple bandpass operator"""
    root_dir = "/sps/grand/DC2Training/ZHAireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000"
    
    all_antenna_pos, meta_data, efield_data = open_event_root(root_dir, single_event=0)
    clean_efield = efield_data['traces'].to_numpy().astype(np.float64)
    print(f"Loaded real E-field data: {clean_efield.shape}")
    
        
    original_std = np.std(clean_efield, axis=-1, keepdims=True) + 1e-8
    original_mean = np.mean(clean_efield, axis=-1, keepdims=True)
    clean_efield_normalized = (clean_efield - original_mean) / original_std
    
    clean_efield_bandlimited = bandlimit_signal(clean_efield_normalized)
    
    A = create_simple_bandpass_operator()
    clean_voltage = A.forward(clean_efield_normalized)
    

    noise_level = 0.9 * np.std(clean_voltage)
    np.random.seed(42)
    noisy_voltage = clean_voltage + noise_level * np.random.randn(*clean_voltage.shape)
    
    print(f" SNR: {20*np.log10(np.std(clean_voltage)/noise_level):.1f} dB")
    

    denoiser = PRUNetW(model_path)
    

    start_time = time.time()
    adjoint_recon = A.adjoint(noisy_voltage)
    adjoint_time = time.time() - start_time
    
    start_time = time.time()
    wiener_recon = wiener_filter_reconstruction(noisy_voltage, A)
    wiener_time = time.time() - start_time
    

    start_time = time.time()
    pnp_recon = pnp_hqs_with_prunet(noisy_voltage, A, denoiser, max_iter=8, verbose=True)
    pnp_time = time.time() - start_time


    plt.plot(pnp_recon[0,0,:])
    plt.plot(wiener_recon[0,0,:])
    plt.plot(clean_efield_bandlimited[0,0,:])
    plt.show()
    def compute_metrics(clean, recon):
        mse = np.mean((clean - recon) ** 2)
        max_val = np.max(np.abs(clean))
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
        return psnr, mse
    
    adjoint_psnr, adjoint_mse = compute_metrics(clean_efield_bandlimited, adjoint_recon)
    wiener_psnr, wiener_mse = compute_metrics(clean_efield_bandlimited, wiener_recon)
    pnp_psnr, pnp_mse = compute_metrics(clean_efield_bandlimited, pnp_recon)
    
   
    print(f"Adjoint:          PSNR={adjoint_psnr:.2f} dB, MSE={adjoint_mse:.2e}, Time={adjoint_time:.3f}s")
    print(f"Wiener:           PSNR={wiener_psnr:.2f} dB, MSE={wiener_mse:.2e}, Time={wiener_time:.3f}s")
    print(f"PnP+PRUNet:       PSNR={pnp_psnr:.2f} dB, MSE={pnp_mse:.2e}, Time={pnp_time:.3f}s")
    
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    ant_idx = 0
    methods = [
        ("Original", clean_efield_bandlimited, None),
        ("Wiener", wiener_recon, wiener_psnr),
        ("PnP+PRUNet", pnp_recon, pnp_psnr)
    ]
    
    pol_names = ['North', 'East', 'Vertical']
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for pol in range(3):
        for method_idx, (name, data, psnr) in enumerate(methods):
            if method_idx == 0:
                axes[0, pol].plot(data[ant_idx, pol, :], color=colors[method_idx], 
                                linewidth=2, label=name)
            else:
                linestyle = '-' if method_idx < 4 else ':'
                alpha = 0.8 if method_idx < 4 else 1.0
                linewidth = 1 if method_idx < 4 else 2
                axes[0, pol].plot(data[ant_idx, pol, :], color=colors[method_idx], 
                                linestyle=linestyle, alpha=alpha, linewidth=linewidth, label=name)
        
        title = f'{pol_names[pol]} - Time Domain'
        axes[0, pol].set_title(title)
        axes[0, pol].legend()
        axes[0, pol].grid(True, alpha=0.3)
        
        # Frequency domain
        freqs = np.fft.rfftfreq(clean_efield.shape[-1], d=1/2e9) / 1e6
        
        for method_idx, (name, data, psnr) in enumerate(methods):
            fft_data = np.abs(np.fft.rfft(data[ant_idx, pol, :]))
            linestyle = '-' if method_idx < 4 else ':'
            alpha = 0.8 if method_idx < 4 else 1.0
            linewidth = 1 if method_idx < 4 else 2
            axes[1, pol].semilogy(freqs, fft_data, color=colors[method_idx], 
                                linestyle=linestyle, alpha=alpha, linewidth=linewidth, label=name)
        
        axes[1, pol].axvline(10, color='black', linestyle=':', alpha=0.7, label='Filter band')
        axes[1, pol].axvline(200, color='black', linestyle=':', alpha=0.7)
        axes[1, pol].set_title(f'{pol_names[pol]} - Frequency Domain')
        axes[1, pol].set_xlabel('Frequency [MHz]')
        axes[1, pol].legend()
        axes[1, pol].grid(True, alpha=0.3)
        axes[1, pol].set_xlim(0, 500)
    
    plt.suptitle(f'Method Comparison: PnP-HQS vs Wiener Filters (PnP improvement: {pnp_psnr - adjoint_psnr:+.2f} dB)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return {
        'adjoint': {'psnr': adjoint_psnr, 'mse': adjoint_mse, 'time': adjoint_time},
        'wiener': {'psnr': wiener_psnr, 'mse': wiener_mse, 'time': wiener_time},
        'pnp': {'psnr': pnp_psnr, 'mse': pnp_mse, 'time': pnp_time}
    }
if __name__ == "__main__":
    model_path = "/sps/grand/selbouch/output/final_model_2M.pth"
    
    results = test_with_wiener_comparison(model_path)