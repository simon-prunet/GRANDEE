import torch
import math

class PnP_HQS:
    def __init__(self, 
                 denoiser,           
                 physics,            
                 lambda_param=0.23,  
                 K=50,                
                 sigma_1=0.05,       
                 sigma_K=0.01):      
        
        self.denoiser = denoiser
        self.physics = physics
        self.lambda_param = lambda_param
        self.K = K
        
        
        self.sigma_k = torch.logspace(math.log10(sigma_1), math.log10(sigma_K), K)
        
        
        self.sigma = getattr(physics.noise_model, 'sigma', 0.01)
        
        self.alpha_k = lambda_param * (self.sigma**2) / (self.sigma_k**2)
    
    def restore(self, y, init_z=None):
    
        z_k = self.physics.A_adjoint(y) if init_z is None else init_z
        
        z_history = [z_k.detach().clone()]
        
        
        for k in range(self.K):

            x_k = self._solve_data_subproblem(y, z_k, self.alpha_k[k])
            

            with torch.no_grad():
                z_k = self.denoiser(x_k, sigma=self.sigma_k[k])
            
            # Store results
            z_history.append(z_k.detach().clone())
            
            
        
        return z_k, z_history
    
    def _solve_data_subproblem(self, y, z_k, alpha_k):
      
        y_fft = torch.fft.fft(y)
        z_fft = torch.fft.fft(z_k)
        freqs = torch.fft.fftfreq(y.shape[-1])
        bandpass = torch.exp(-((freqs - self.physics.center_freq) / self.physics.bandwidth)**2)
        
        
        numerator = torch.conj(bandpass) * y_fft + alpha_k * z_fft
        denominator = torch.conj(bandpass) * bandpass + alpha_k
        x_fft = numerator / (denominator + 1e-8)  
        
        return torch.fft.ifft(x_fft).real