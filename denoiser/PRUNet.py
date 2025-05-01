import torch 
import torch.nn as nn
import torch.optim as optim
from models.network_unet import UNetRes
import matplotlib.pyplot as plt

class PRUNet(UNetRes):

    def __init__(self, in_nc=3, n_downs=3, nc=[64, 128, 256, 512], nb=2):

        # Init res block unet, with an additional input channel for the noise level vector
        super(PRUNet,self).__init__(in_nc=in_nc+1, out_nc=in_nc, n_downs=n_downs, nc=nc, nb=nb)

    
    def forward(self, x, sigma=None):
        # If sigma is not provided, use a default value
        if sigma is None:
            sigma = torch.ones(1, device=x.device) * 0.1
            
        # Make sure sigma is a tensor
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=x.device, dtype=torch.float)
            
        # Handle scalar sigma (size 1) by expanding to match batch size
        if sigma.numel() == 1:  # If sigma is a scalar
            sigma = sigma.expand(x.shape[0])
        
        # Reshape sigma to match what the model expects
        sigma = sigma.view(x.shape[0], 1).unsqueeze(-1).expand(-1, -1, x.shape[-1]).float()
        xx = torch.cat((sigma,x),dim=1)    
        # Call the model with properly formatted inputs
        return super(PRUNet,self).forward(xx)
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, model, dataloader, lr=1e-4):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def training_step(self, batch):
        print(f"Batch type: {type(batch)}, length: {len(batch)}")
        if isinstance(batch, tuple) or isinstance(batch, list):
            for i, item in enumerate(batch):
                print(f"Item {i} shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
        sigma, noisy, clean = batch
        
        sigma, noisy, clean = sigma.to(device).float(),noisy.to(device).float(), clean.to(device).float()
        batch_size = noisy.shape[0]
        #sigma_map =  sigma.view(batch_size, 1).unsqueeze(-1).expand(-1, -1, 256)
        #sigma_map = sigma_map.float()
        x_noisy = noisy.float()
        x_hat = self.model(x_noisy, sigma) 

        
        loss = self.criterion(x_hat, clean)
        return loss

    def train(self, num_epochs=20):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for batch in self.dataloader:
                self.optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss / len(self.dataloader):.6f}")

def visualize_denoising(model, dataset, num_samples=5):
    model.eval()
    device = next(model.parameters()).device

    noisy_samples, clean_samples = [], []
    denoised_samples = []
    for i in range(num_samples):
        sigma, noisy, clean = dataset[i] 
        noisy_tensor = torch.tensor(noisy).unsqueeze(0).to(device)  
        clean_tensor = torch.tensor(clean).unsqueeze(0).to(device)
        batch_size = noisy_tensor.shape[0]
        sigma        = torch.tensor(sigma).unsqueeze(0).to(device)

        #reshape sigma
        
        sigma =  sigma.view(batch_size, 1).unsqueeze(-1).expand(-1, -1, 256).float()
        x_noisy = noisy_tensor.float()
        

        with torch.no_grad():
            denoised = model(x_noisy, sigma)

        noisy_samples.append(x_noisy.cpu().squeeze().numpy())
        clean_samples.append(clean_tensor.cpu().squeeze().numpy())
        denoised_samples.append(denoised.cpu().squeeze().numpy())

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))

    for i in range(num_samples):
        axes[i, 0].plot(noisy_samples[i][0], label="Noisy Signal")
        axes[i, 0].set_title("Noisy")
        axes[i, 1].plot(clean_samples[i][0], label="Clean Signal")
        axes[i, 1].set_title("Clean")
        axes[i, 2].plot(denoised_samples[i][0], label="Denoised Signal")
        axes[i, 2].set_title("Denoised")

    plt.tight_layout()
    plt.show()