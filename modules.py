
import functools
try:
    import megengine as torch
    import megengine.module as nn
    import megengine.functional as F
    use_mge = True
    print('You are Using Megengine as Network backend...')
except:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    use_mge = False
    print('You are Using Pytorch as Network backend...')

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block)
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, channels=64, momentum=0.8):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=momentum)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=momentum)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.concat = Concat()
        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(self.concat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(self.concat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(self.concat((x, x1, x2, x3), 1)))
        x5 = self.conv5(self.concat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        if use_mge:
            self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
            self.pixel_shuffle = PixelShuffle(up_scale)
            # self.pixel_shuffle = Upsample(scale_factor=up_scale)
        else:
            self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class PixelShuffle(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        N, C, iH, iW = x.shape
        oH = iH * self.scale
        oW = iW * self.scale
        oC = C // self.scale // self.scale
        after_view = x.reshape(N, oC, self.scale, self.scale, iH, iW)
        after_transpose = F.dimshuffle(after_view, (0,1,4,2,5,3))
        output = after_transpose.reshape(N, oC, oH, oW)

        return output


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='BILINEAR'):
        super().__init__()
        self.scale = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = 1
        self.backend = 'megengine' if use_mge else 'pytorch'
        self.concat = self.func_cat()

    def func_cat(self):
        func_cat = None
        if self.backend == 'megengine':
            func_cat = F.concat
        elif self.backend == 'pytorch':
            func_cat = torch.cat
        return func_cat

    def forward(self, x, dim=None):
        return self.concat(x, dim if dim is not None else self.dim)


if __name__ == '__main__':
    RRDBnet = RRDB()
    up = UpsampleBLock(64, 4)
