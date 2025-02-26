import torch
from models.network_unet import UNetRes

class PRUNet(UNetRes):

    def __init__(self, in_nc=3, n_downs=3, nc=[64, 128, 256, 512], nb=2):

        # Init res block unet, with an additional input channel for the noise level vector
        super(PRUNet,self).__init__(in_nc=in_nc+1, out_nc=in_nc, n_downs=n_downs, nc=nc, nb=nb)

    def forward(self, x0, sigma):

        # Beware, sigma is a vector of size n_batch here...
        n_batch, _ , n_sample = x0.size()
        noise = sigma[:,None,None]*torch.ones(n_batch, 1, n_sample, dtype=torch.float64) # Noise level 
        xx = torch.cat((noise,x0),dim=1)
        return super(PRUNet,self).forward(xx)
    
def train(model, train_loader, optimizer, criterion, device, min_sigma=0.1, max_sigma=10):
    model.train()
    rss = 0
    for sigma, noisy, data in train_loader:
        noisy = noisy.to(device).float()
        data = data.to(device).float()

        optimizer.zero_grad()
        output = model(noisy_data, sigma)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        rss += loss.item()

    return rss / len(train_loader)

