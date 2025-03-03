import torch
from models.network_unet import UNetRes
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from simulator.simulator import Simulator
from argparse import ArgumentParser
import matplotlib.pyplot as plt 

class PRUNet(UNetRes):

    def __init__(self, in_nc=3, n_downs=3, nc=[64, 128, 256, 512], nb=2):

        # Init res block unet, with an additional input channel for the noise level vector
        super(PRUNet,self).__init__(in_nc=in_nc+1, out_nc=in_nc, n_downs=n_downs, nc=nc, nb=nb)

    def forward(self, x0, noise):
        xx = torch.cat((noise,x0),dim=1)
        return super(PRUNet,self).forward(xx)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, model, dataloader, lr=1e-4):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def training_step(self, batch):
        noisy, clean = batch
        noisy, clean = noisy.to(device).float(), clean.to(device).float()

        sigma_map = noisy[:, :1, :].float()  
        x_noisy = noisy[:, 1:, :].float()    

        x_hat = self.model(x_noisy, sigma_map) 

        
        loss = self.criterion(x_hat, clean)
        return loss

    def train(self, num_epochs=20):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for batch in self.dataloader:
                self.optimizer.zero_grad()
                loss = self.training_step(batch).float()
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
        noisy, clean = dataset[i] 
        noisy_tensor = torch.tensor(noisy).unsqueeze(0).to(device)  
        clean_tensor = torch.tensor(clean).unsqueeze(0).to(device)

        sigma = noisy_tensor[:, :1, :].float()
        x_noisy = noisy_tensor[:, 1:, :].float()

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

def parse_args():
    parser = ArgumentParser(description="Train PRUNet")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save/load the model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    dataset = Simulator(
        in_nc=3,
        n_samples=256,
        a=4,
        sigma_range=[0.1, 0.5],
        amplitude_range=[0.1, 1.],
        scale_range=[1., 2.],
        loc_range=[100,150],
        length=256
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size)
    
    model = PRUNet(in_nc=3)
    trainer = Trainer(model, train_loader, lr=args.lr, device=args.device)
    

    trainer.train(num_epochs=args.epochs)
    
 
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")
