import math
from modules import *

class SRResnet(nn.Module):
    def __init__(self, scale_factor, nb=16, filters=64):
        upsample_block_num = int(math.log(scale_factor, 2))

        super().__init__()
        self.block_pre = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.residual_blocks = make_layer(ResidualBlock(filters), nb-1)
        self.block_post = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters, momentum=0.8)
        )
        self.upsample = make_layer(UpsampleBLock(filters, 2), upsample_block_num)
        self.out = nn.Conv2d(filters, 3, kernel_size=9, padding=4)

    def forward(self, x):
        block_pre = self.block_pre(x)
        residual_blocks = self.residual_blocks(block_pre)
        block_post = self.block_post(residual_blocks)
        small_features = block_post + block_pre
        upsample = self.upsample(small_features)
        out = self.out(upsample)

        return (torch.tanh(out) + 1) / 2


class SR_RRDB(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=5, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upsample = make_layer(UpsampleBLock(nf, 2), 2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.upsample(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
        

class DeepUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, filters=32):
        super().__init__()
        
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(filters, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(filters, filters*2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(filters*2, filters*2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(filters*2, filters*4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(filters*4, filters*4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(filters*4, filters*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(filters*8, filters*8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(filters*8, filters*16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(filters*16, filters*16, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(filters*16, filters*8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(filters*16, filters*8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(filters*8, filters*8, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(filters*8, filters*4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(filters*4, filters*4, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(filters*4, filters*2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(filters*2, filters*2, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv9_1 = nn.Conv2d(filters*2, filters, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        
        self.conv10_1 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1)

        # Deep Supervision
        self.out8 = nn.Conv2d(filters*8, out_channels, kernel_size=1)
        self.out4 = nn.Conv2d(filters*4, out_channels, kernel_size=1)
        self.out2 = nn.Conv2d(filters*2, out_channels, kernel_size=1)
    
    def forward(self, x, mode='test'):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        
        out= self.conv10_1(conv9)

        if mode == 'train':
            # Deep Supervision
            out8 = self.out8(conv6)
            out4 = self.out4(conv7)
            out2 = self.out2(conv8)
            return out, out2, out4, out8
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt