import torch
import math
import deepinv as dinv 
import optuna
from simulator.simulator import Simulator
from denoiser.PRUNet import PRUNet, Trainer

class BandpassFilterConvolution(dinv.physics.LinearPhysics):
    def __init__(self, center_freq, bandwidth, noise_level):
        super().__init__()
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.noise_model = dinv.physics.GaussianNoise(sigma=noise_level)
        
    def A(self, x):
        x = x.float()
        fft = torch.fft.fft(x)
        freqs = torch.fft.fftfreq(x.shape[-1])
        bandpass = torch.exp(-((freqs - self.center_freq) / self.bandwidth)**2)
        filtered = fft * bandpass
        convolved = torch.fft.ifft(filtered).real
        return self.noise_model(convolved)
    
    def A_adjoint(self, y):
        y = y.float()
        fft = torch.fft.fft(y)
        freqs = torch.fft.fftfreq(y.shape[-1])
        bandpass = torch.exp(-((freqs - self.center_freq) / self.bandwidth)**2)
        filtered = fft * bandpass
        return torch.fft.ifft(filtered).real

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
    
from torch.utils.data import DataLoader
if __name__=="__main__":
    
    simu = Simulator(
            in_nc=3,
            n_samples=256,
            a=4,
            sigma_range=[0.1, 2],
            amplitude_range=[0.1, 1.],
            scale_range=[1., 2.],
            loc_range=[100,150],
            length=100
        )
    train_loader = DataLoader(simu, batch_size=10, drop_last=True)
        
    
    model = PRUNet(in_nc=3,nc=[64, 128, 256, 512, 1024], nb=2)
    print(model)
    trainer = Trainer(model, train_loader, lr=0.001)
        
    trainer.train(num_epochs=10)

    sigma, noisy, traces = next(iter(train_loader))
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    traces = traces.to(device)
    noisy = noisy.to(device)
    sigma = sigma.to(device)

    def objective(trial):

        sigma_1 = trial.suggest_float("sigma_1", 0.01, 0.011) 
        sigma_K = trial.suggest_float("sigma_K", 0.001, 0.0011)  
        lambda_param = trial.suggest_float("lambda_param", 0.01, 1.0, log=True)  
        
        
        
        center_freq = 0.1
        bandwidth = 0.01
        noise_level = 0 
        
        physics = BandpassFilterConvolution(center_freq, bandwidth, noise_level)

        y = physics(traces)
        
        pnp_hqs_solver = PnP_HQS(
            denoiser=model.float(),  
            physics=physics,
            lambda_param=lambda_param,
            K=50,
            sigma_1=sigma_1,
            sigma_K=sigma_K
        )
        

        try:
            deconvolved, _ = pnp_hqs_solver.restore(y)
            
            
            mse = dinv.metric.MSE()(traces, deconvolved).mean().item()
            
            
            if torch.isnan(deconvolved).any() or torch.isinf(deconvolved).any() or mse < 0:
                return -float('inf')
                
            return mse
        except Exception as e:
            print(f"Error during trial: {e}")
            return -float('inf')


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print(f"Best parameters: {best_params}")
print(f"Best MSE: {study.best_value}")


