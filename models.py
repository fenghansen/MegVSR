import math
from modules import *

class SlowFusion_RRDB(nn.Module):
    def __init__(self, in_nc=9, out_nc=3, nf=64, nb=5, gc=32, cv2_INTER=True):
        super().__init__()
        self.nb = nb
        self.cf = in_nc // 6
        self.nframes = in_nc//3

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.RRDB = [RRDB(nf=nf, gc=gc)] * nb
        self.FusionUnet = [FusionUnet(nf*2, nf, nf=nf)] * (self.nframes-1)
        # self.RRDB1 = RRDB(nf=nf, gc=gc)
        # self.FusionUnet1 = FusionUnet(nf*2, nf, nf=nf)
        # self.RRDB2 = RRDB(nf=nf, gc=gc)
        # self.FusionUnet2 = FusionUnet(nf*2, nf, nf=nf)
        # self.RRDB3 = RRDB(nf=nf, gc=gc)
        # self.FusionUnet3 = FusionUnet(nf*2, nf, nf=nf)
        # self.RRDB4 = RRDB(nf=nf, gc=gc)
        # self.FusionUnet4 = FusionUnet(nf*2, nf, nf=nf)
        # self.RRDB5 = RRDB(nf=nf, gc=gc)
        
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upsample = make_layer(UpsampleBLock(nf, 2), 2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.concat = Concat()
        self.cv2_INTER = cv2_INTER
        if self.cv2_INTER is False:
            if use_mge:
                self.bicubic = Upsample(scale_factor=4, mode='BILINEAR')
            else:
                self.bicubic = nn.Upsample(scale_factor=4, mode='bicubic')

    def forward(self, x):
        frames = []
        for i in range(self.nframes):
            frames.append(x[:,i*3:i*3+3,:,:])
        # fisrt conv
        fusion = []
        for i in range(self.nframes):
            fusion.append(self.conv_first(frames[i]))
        center_fea = fusion[self.cf]
        
        # Slow Fusion
        for step in range(self.nb-1):
            fea = []
            for i in range(self.nframes-step):
                fea.append(self.RRDB[step](fusion[i]))
            fusion = []
            for i in range(self.nframes-step-1):
                cat = self.concat((fea[i], fea[i+1]))
                fusion.append(self.FusionUnet[step](cat))
        # fusion = [Tensor(b,nf,w,h)]
        RRDB_last = self.RRDB[self.nframes-1](fusion[0])
        for i in range(self.nb-self.nframes):
            RRDB_last = self.RRDB[self.nb-i-1](RRDB_last)

        trunk = self.trunk_conv(RRDB_last)
        fea = center_fea + trunk

        fea = self.upsample(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        if self.cv2_INTER is False:
            imgs_bc = self.bicubic(x[:,self.cf,:,:,:])
            out = out + imgs_bc

        return out


class VSR_RRDB(nn.Module):
    def __init__(self, in_nc=9, out_nc=3, nf=64, nb=5, gc=32, cv2_INTER=True):
        super().__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.cf = in_nc // 6
        self.nframes = in_nc//3
        # PCC alignment module with Pyramid, Cascading and Convolution
        self.Unet = Unet(in_nc, out_channels=nf, nf=nf)
        # self.PCC = PCCUnet(3, out_channels=nf, nf=nf, nframes=self.nframes)
        # self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        # self.TSA = TSAFusion(nf=nf, nframes=self.nframes)
        self.RRDB_trunk = make_layer(RRDB_block_f(nf=nf, gc=gc), nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upsample = make_layer(UpsampleBLock(nf, 2), 2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.cv2_INTER = cv2_INTER
        if self.cv2_INTER is False:
            if use_mge:
                self.bicubic = Upsample(scale_factor=4, mode='BILINEAR')
            else:
                self.bicubic = nn.Upsample(scale_factor=4, mode='bicubic')

    def forward(self, x):
        fea = self.Unet(x)
        # b,t,c,h,w = x.size()
        # fea = self.PCC(x)
        # fea = self.conv_first(x.view(-1,c,h,w)).view(b,t,-1,h,w)
        # fea = self.TSA(fea)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.upsample(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        if self.cv2_INTER is False:
            imgs_bc = self.bicubic(x[:,self.cf,:,:,:])
            out = out + imgs_bc

        return out

class SRResnet(nn.Module):
    def __init__(self, scale_factor=4, nb=16, nf=64, res=True):
        super().__init__()
        self.res = res
        self.block_pre = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.residual_blocks = make_layer(ResidualBlock(nf), nb-1)
        self.block_post = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf, momentum=0.8)
        )
        self.upsample = make_layer(UpsampleBLock(nf, 2), 2)
        self.out = nn.Conv2d(nf, 3, kernel_size=9, padding=4)
        if self.res:
            if use_mge:
                self.bicubic = Upsample(scale_factor=4, mode='BILINEAR')
            else:
                self.bicubic = nn.Upsample(scale_factor=4, mode='bicubic')

    def forward(self, x):
        block_pre = self.block_pre(x)
        residual_blocks = self.residual_blocks(block_pre)
        block_post = self.block_post(residual_blocks)
        small_features = block_post + block_pre
        upsample = self.upsample(small_features)
        out = self.out(upsample)
        if self.res:
            imgs_bc = self.bicubic(x)
            return out + imgs_bc
        else:
            return (torch.tanh(out) + 1) / 2


class SR_RRDB(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=5, gc=32, cv2_INTER=True):
        super().__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f(nf=nf, gc=gc), nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upsample = make_layer(UpsampleBLock(nf, 2), 2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.cv2_INTER = cv2_INTER
        if self.cv2_INTER is False:
            if use_mge:
                self.bicubic = Upsample(scale_factor=4, mode='BILINEAR')
            else:
                self.bicubic = nn.Upsample(scale_factor=4, mode='bicubic')

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.upsample(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        if self.cv2_INTER is False:
            imgs_bc = self.bicubic(x)
            out = out + imgs_bc

        return out
        

class PCCUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=32, nframes=3):
        super().__init__()
        self.nframes = nframes
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.encoder = nn.Sequential(
            self.conv1_1, self.conv1_2, self.pool1,
            self.conv2_1, self.conv2_2, self.pool2,
            self.conv3_1, self.conv3_2, self.pool3
        )

        # 此处承接三路下采样
        self.concat = Concat(dim=1)
        self.cbam4_1 = CBAM(nf*8)
        self.cbam4_2 = CBAM(nf*4)
        self.conv4_1 = nn.Conv2d(nf*4*self.nframes, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1)

        self.upv5 = nn.ConvTranspose2d(nf*4, nf*4, 4, stride=2, padding=(0,1))
        self.conv5_1 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(nf*2, nf*2, 4, stride=2, padding=(0,1))
        self.conv6_1 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=(0,1))
        self.conv7_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.out = nn.Conv2d(nf, out_channels, kernel_size=1, stride=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x, mode='test'):
        frames = []
        for i in range(self.nframes):
            frames.append(x[:,i*3:i*3+3,:,:])
        cf = self.nframes // 2

        # center frame encode
        conv1 = self.lrelu(self.conv1_1(frames[cf]))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)
        
        # other frames encoder feature
        features = [None] * self.nframes
        for i in range(self.nframes):
            features[i] = pool3 if i == cf else self.encoder(frames[i])

        merge = self.concat(features)
        conv4 = self.lrelu(self.conv4_1(merge))
        conv4 = self.cbam4_1(conv4)
        conv4 = self.lrelu(self.conv4_2(conv4))
        conv4 = self.cbam4_2(conv4)
        
        up5 = self.upv5(conv4)
        up5 = self.concat([conv3, up5[:,:,:conv3.shape[2],:conv3.shape[3]]], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = self.concat([conv2, up6[:,:,:conv2.shape[2],:conv2.shape[3]]], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = self.concat([conv1, up7[:,:,:conv1.shape[2],:conv1.shape[3]]], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        out= self.out(conv7)

        return out

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=32, nframes=3):
        super().__init__()
        self.nframes = nframes
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(nf)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(nf*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1)
        self.cbam3 = CBAM(nf*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.concat = Concat(dim=1)
        self.conv4_1 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1)
        self.cbam4 = CBAM(nf*8)

        self.upv5 = nn.ConvTranspose2d(nf*8, nf*4, 4, stride=2, padding=(0,1))
        self.conv5_1 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        self.cbam5 = CBAM(nf*2)
        
        self.upv6 = nn.ConvTranspose2d(nf*2, nf*2, 4, stride=2, padding=(0,1))
        self.conv6_1 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.cbam6 = CBAM(nf*1)

        self.upv7 = nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=(0,1))
        self.conv7_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.out = nn.Conv2d(nf, out_channels, kernel_size=1, stride=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x, mode='test'):
        # center frame encode
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        conv1 = self.cbam1(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        conv2 = self.cbam2(conv2)
        pool2 = self.pool2(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        conv3 = self.cbam3(conv3)
        pool3 = self.pool3(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        conv4 = self.cbam4(conv4)
        
        up5 = self.upv5(conv4)
        up5 = self.concat([conv3, up5[:,:,:conv3.shape[2],:conv3.shape[3]]], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        conv5 = self.cbam5(conv5)
        
        up6 = self.upv6(conv5)
        up6 = self.concat([conv2, up6[:,:,:conv2.shape[2],:conv2.shape[3]]], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        conv6 = self.cbam6(conv6)
        
        up7 = self.upv7(conv6)
        up7 = self.concat([conv1, up7[:,:,:conv1.shape[2],:conv1.shape[3]]], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        out= self.out(conv7)

        return out

class FusionUnet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, nf=64, **kwargs):
        super().__init__()
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1 = ResConvBlock_CBAM(in_channels,  nf=nf)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = ResConvBlock_CBAM(nf, nf=nf*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = ResConvBlock_CBAM(nf*2, nf=nf*4)
        
        self.upv4 = nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=(0,1))
        self.conv4 = ResConvBlock_CBAM(nf*4, nf=nf*2)

        self.upv5 = nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=(0,1))
        self.conv5 = ResConvBlock_CBAM(nf*2, nf=nf*1)

        self.out = nn.Conv2d(nf, out_channels, kernel_size=1, stride=1)
        self.concat = Concat()
    
    def forward(self, x, mode='test'):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        
        up4 = self.upv4(conv3)
        up4 = self.concat([conv2, up4[:,:,:conv2.shape[2],:conv2.shape[3]]], 1)
        conv4 = self.conv4(up4)
        
        up5 = self.upv5(conv4)
        up5 = self.concat([conv1, up5[:,:,:conv1.shape[2],:conv1.shape[3]]], 1)
        conv5 = self.conv5(up5)

        out = self.out(conv5)

        return out

if __name__ == '__main__':
    SRGAN = SRResnet()
    unet = PCCUnet()
    print('success')