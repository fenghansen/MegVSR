import math
from modules import *

class VSR_RRDB(nn.Module):
    def __init__(self, in_nc=9, out_nc=3, nf=64, nb=5, gc=32, cv2_INTER=True):
        super().__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.cf = in_nc // 6
        # PCC alignment module wtih Pyramid, Cascading and Convolution
        self.PCC = PCCUnet(3, out_channels=nf, nf=nf, nframes=in_nc//3)
        # self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
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
        fea = self.PCC(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.upsample(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        if self.cv2_INTER is False:
            imgs_bc = self.bicubic(x[:,self.cf*3:self.cf*3+3,:,:])
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

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f(nf=nf, gc=gc), nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='BILINEAR')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='BILINEAR')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

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
        self.conv4_1 = nn.Conv2d(nf*4*self.nframes, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1)

        self.upv5 = nn.ConvTranspose2d(nf*4, nf*4, 2, stride=2)
        self.conv5_1 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(nf*2, nf*2, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(nf, nf, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.out = nn.Conv2d(nf, out_channels, kernel_size=1, stride=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x, mode='test'):
        frames = x.split(3, dim=1)
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
        conv4 = self.lrelu(self.conv4_2(conv4))
        
        up5 = self.upv5(conv4)
        up5 = self.concat([conv3, up5], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = self.concat([conv2, up6], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = self.concat([conv1, up7], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        out= self.out(conv7)

        return out

if __name__ == '__main__':
    SR_RRDBnetwork = RRDBNet()
    SRGAN = SRResnet()
    unet = PCCUnet()
    print('success')