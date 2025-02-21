import torch
from models.network_unet import UNetRes

class PRUNet(UNetRes):

    def __init__(self, in_nc=3, sigma=1.0, n_downs=3, nc=[64, 128, 256, 512], nb=2):

        # Init res block unet, with an additional input channel for the noise level vector
        super(PRUNet,self).__init__(in_nc=in_nc+1, out_nc=in_nc, n_downs=n_downs, nc=nc, nb=nb)
        self.sigma = sigma

    def forward(self, x0):

        n_batch, _ , n_sample = x0.size()
        noise = self.sigma*torch.randn(n_batch, 1, n_sample)
        xx = torch.cat((noise,x0),dim=1)
        return super(PRUNet,self).forward(xx)
    



