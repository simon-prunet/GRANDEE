import torch
import torch.nn as nn
import models.basicblock as B
import numpy as np

'''
# ====================
# unet
# ====================
'''


class UNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, n_downs=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNet, self).__init__()

        self.n_downs = n_downs # Needed by forward method
        self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        # Loop on depth levels
        for nd in range(n_downs):

            settattr(self,'m_down%d'%(nd+1), B.sequential(*[B.conv(nc[nd], nc[nd], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[nd], nc[nd+1], mode='2'+act_mode))
        # self.m_down1 = ...B.conv(nc[0],nc[0])...downsample_block(nc[0],nc[1])
        # self.m_down2 = ...B.conv(nc[1],nc[1])...downsample_block(nc[1],nc[2])
        # self.m_down3 = ...B.conv(nc[2],nc[2])...downsample_block(nc[2],nc[3])

        self.m_body  = B.sequential(*[B.conv(nc[self.n_downs], nc[self.n_downs], mode='C'+act_mode) for _ in range(nb+1)])
        # self.m_body = ...B.conv(nc[3],nc[3])...

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        for nu in range(n_downs,0,-1):

            settattr(self, 'm_up%d'%nu, B.sequential(upsample_block(nc[nu], nc[nu-1], mode='2'+act_mode), *[B.conv(nc[nu-1], nc[nu-1], mode='C'+act_mode) for _ in range(nb)])
        #self.m_up3 = ...upsample_block(nc[3],nc[2])...B.conv(nc[2],nc[2])...
        #self.m_up2 = ...upsample_block(nc[2],nc[1])...B.conv(nc[1],nc[1])...
        #self.m_up1 = ...upsample_block(nc[1],nc[0])...B.conv(nc[0],nc[0])...

        self.m_tail = B.conv(nc[0], out_nc, bias=True, mode='C')

    def forward(self, x0):

        xs = []
        # First block
        xs.append(self.m_head(x0))
        # x1 = self.m_head(x0)

        # Going down in size
        for nd in range(self.n_downs):
            xs.append(getattr(self,'m_down%d'%(nd+1),xs[-1]))
        #x2 = self.m_down1(x1)
        #x3 = self.m_down2(x2)
        #x4 = self.m_down3(x3)

        # Lower level
        x = self.m_body(xs[-1])
        # x = self.m_body(x4)

        # Going up, with skip connections
        for i, nu in enumerate(range(self.n_downs,0,-1))
            x = getattr(self, 'm_up%d'%nu, x+xs[-1-i])
        #x = self.m_up3(x+x4)
        #x = self.m_up2(x+x3)
        #x = self.m_up1(x+x2)

        # Last block
        x = self.m_tail(x+xs[0]) + x0
        #x = self.m_tail(x+x1) + x0

        
        return x


class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, n_downs=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.n_downs = n_downs # Needed by forward method
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        for nd in range(self.n_downs):
            setattr(self, 'm_down%d'%(nd+1), B.sequential(*[B.ResBlock(nc[nd], nc[nd], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[nd], nc[nd+1], bias=False, mode='2')))
        # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[self.n_downs], nc[self.n_downs], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        # self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        for nu in range(self.n_downs,0,-1):
            setattr(self, 'm_up%d'%(nu), B.sequential(upsample_block(nc[nu], nc[nu-1], bias=False, mode='2'), *[B.ResBlock(nc[nu-1], nc[nu-1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]))
        # self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        # self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        # self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):

        xs = []
        # First block
        xs.append(self.m_head(x0))
        # x1 = self.m_head(x0)

        # Going down in size
        for nd in range(self.n_downs):
            xs.append(getattr(self,'m_down%d'%(nd+1),xs[-1]))
        # x2 = self.m_down1(x1)
        # x3 = self.m_down2(x2)
        # x4 = self.m_down3(x3)

        # Lower level
        x = self.m_body(xs[-1])
        # x = self.m_body(x4)

        # Going up, with skip connections
        for i, nu in enumerate(range(self.n_downs,0,-1))
            x = getattr(self, 'm_up%d'%nu, x+xs[-1-i])
        # x = self.m_up3(x+x4)
        # x = self.m_up2(x+x3)
        # x = self.m_up1(x+x2)

        # Last block
        x = self.m_tail(x+xs[0])
        # x = self.m_tail(x+x1)

        return x


class ResUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, n_downs=3, nc=[64, 128, 256, 512], nb=4, act_mode='L', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.n_downs = n_downs # Needed by forward method
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        for nd in range(self.n_downs):
            setattr(self, 'm_down%d'%(nd+1), B.sequential(*[B.IMDBlock(nc[nd], nc[nd], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[nd], nc[nd+1], bias=False, mode='2')))
        # self.m_down1 = B.sequential(*[B.IMDBlock(nc[0], nc[0], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        # self.m_down2 = B.sequential(*[B.IMDBlock(nc[1], nc[1], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        # self.m_down3 = B.sequential(*[B.IMDBlock(nc[2], nc[2], bias=False, mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.IMDBlock(nc[n_downs], nc[n_downs], bias=False, mode='C'+act_mode) for _ in range(nb)])
        # self.m_body  = B.sequential(*[B.IMDBlock(nc[3], nc[3], bias=False, mode='C'+act_mode) for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        for nu in range(self.n_downs,0,-1):

            setattr(self, 'm_up%d'%(nu), B.sequential(upsample_block(nc[nu], nc[nu-1], bias=False, mode='2'), *[B.IMDBlock(nc[nu-1], nc[nu-1], bias=False, mode='C'+act_mode) for _ in range(nb)]))
        #self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.IMDBlock(nc[2], nc[2], bias=False, mode='C'+act_mode) for _ in range(nb)])
        #self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.IMDBlock(nc[1], nc[1], bias=False, mode='C'+act_mode) for _ in range(nb)])
        #self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.IMDBlock(nc[0], nc[0], bias=False, mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h = x.size()[-1]
        paddingBottom = int(np.ceil(h/8)*8-h)
        x = nn.ReplicationPad1d((0, paddingBottom))(x)

        # First block
        xs.append(self.m_head(x))
        # x1 = self.m_head(x)
        for nd n range(self.n_downs):
            xs.append(getattr(self, 'm_down%d'%(nd+1), xs[-1]))
        # x2 = self.m_down1(x1)
        # x3 = self.m_down2(x2)
        # x4 = self.m_down3(x3)

        # Lower level
        x = self.m_body(xs[-1])
        # x = self.m_body(x4)

        for i,nu in enumerate(range(self.n_downs,0,-1)):
            x = getattr(self, 'm_up%d'%nu, x+xs[-1-i])
        # x = self.m_up3(x+x4)
        # x = self.m_up2(x+x3)
        # x = self.m_up1(x+x2)

        # Last block
        x = self.m_tail(x+xs[0])
        # x = self.m_tail(x+x1)
        
        # Crop
        x = x[..., :h]

        return x


class UNetResSubP(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, n_downs = 3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):

        self.n_downs = n_downs # Needed by forward method
        super(UNetResSubP, self).__init__()
        sf = 2
        self.m_ps_down = B.PixelUnShuffle(sf)
        self.m_ps_up = nn.PixelShuffle(sf)
        self.m_head = B.conv(in_nc*sf*sf, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        for nd in range(self.n_downs):
            setattr(self, 'm_down%d'%(nd+1), B.sequential(*[B.ResBlock(nc[nd], nc[nd], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[nd], nc[nd+1], mode='2'+act_mode)))
        # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = B.sequential(*[B.ResBlock(nc[self.n_downs], nc[self.n_downs], mode='C'+act_mode+'C') for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        for i,u in enumerate(range(self.n_downs,0,-1)):
            setattr(self, 'm_up%d'%nu, B.sequential(upsample_block(nc[nu], nc[nu-1], mode='2'+act_mode), *[B.ResBlock(nc[nu-1], nc[nu-1], mode='C'+act_mode+'C') for _ in range(nb)]))
        # self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.ResBlock(nc[2], nc[2], mode='C'+act_mode+'C') for _ in range(nb)])
        # self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.ResBlock(nc[1], nc[1], mode='C'+act_mode+'C') for _ in range(nb)])
        # self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.ResBlock(nc[0], nc[0], mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc*sf*sf, bias=False, mode='C')

    def forward(self, x0):

        xs = []
        # Init block
        x0_d = self.m_ps_down(x0)

        # Head block
        xs.append(self.m_head(x0_d))
        # x1 = self.m_head(x0_d)

        # Going down in size
        for nd in range(self.n_downs):
            xs.append(getattr(self, 'm_down%d'%(nd+1), xs[-1]))
        # x2 = self.m_down1(x1)
        # x3 = self.m_down2(x2)
        # x4 = self.m_down3(x3)

        # Lower level
        x = self.m_body(xs[-1])
        # x = self.m_body(x4)

        # Going up, with skip connections
        for i,u in enumerate(range(self.n_downs,0,-1)):
            x = getattr(self, 'm_up%d'%(nd), x+xs[-1-i])
        # x = self.m_up3(x+x4)
        # x = self.m_up2(x+x3)
        # x = self.m_up1(x+x2)

        # Tail block
        x = self.m_tail(x+xs[0])
        # x = self.m_tail(x+x1)

        # Closing block
        x = self.m_ps_up(x) + x0

        return x


class UNetPlus(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetPlus, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode[1]))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode[1]))
        self.m_down3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode[1]))

        self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[2], nc[2], mode='C'+act_mode[1]))
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[1], nc[1], mode='C'+act_mode[1]))
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb-1)], B.conv(nc[0], nc[0], mode='C'+act_mode[1]))

        self.m_tail = B.conv(nc[0], out_nc, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0
        return x

'''
# ====================
# nonlocalunet
# ====================
'''

class NonLocalUNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64,128,256,512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(NonLocalUNet, self).__init__()

        down_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='strideconv')
        up_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='strideconv')

        self.m_head = B.conv(in_nc, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))


        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.m_down3 = B.sequential(down_nonlocal, *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.m_body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))


        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], up_nonlocal)
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1) + x0
        return x


if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
#    net = UNet(act_mode='BR')
    net = NonLocalUNet()
    net.eval()
    with torch.no_grad():
        y = net(x)
    y.size()

