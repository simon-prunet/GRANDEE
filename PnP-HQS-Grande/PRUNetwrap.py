import torch
import numpy as np
from denoiser.PRUNet import *

class PRUNetW:
    """Wrapper for trained PRUNet model"""
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_prunet_model(model_path)
        self.model.eval()
        
    
    def load_prunet_model(self, model_path):
        try:
            model = PRUNet(in_nc=3, nb=2, nc=[64, 128, 256, 512, 1024])
            checkpoint = torch.load(model_path, map_location=self.device)
            
         
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            return model
            
        except Exception as e:
            print(f" Error loading PRUNet model: {e}")
            raise e
    
    def denoise(self, signal, sigma):
        with torch.no_grad():
            denoised = np.zeros_like(signal)
            
            for ant in range(signal.shape[0]):
                x = torch.from_numpy(signal[ant]).float().unsqueeze(0).to(self.device)  
                sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=self.device)
                x_denoised = self.model(x, sigma_tensor)
                denoised[ant] = x_denoised.squeeze(0).cpu().numpy()
        
        return denoised