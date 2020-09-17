
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
    # print('You are Using Pytorch as Network backend...')

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block)
    return nn.Sequential(*layers)

class ResConvBlock_CBAM(nn.Module):
    def __init__(self, in_nc, nf=64, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cbam = CBAM(nf)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        out = self.res_scale * self.cbam(self.lrelu(self.conv2(x))) + x
        return x + out * self.res_scale

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        nf (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
    """

    def __init__(self, nf=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlock(nn.Module):
    """Residual block with BN.

    It has a style of:
        ---Conv-BN_PReLU-Conv-BN-+-
         |_______________________|

    Args:
        nf (int): Channel number of intermediate features.
            Default: 64.
        momentum (float): Momentum of BN. Default: 0.8
    """
    def __init__(self, nf=64, momentum=0.8):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(nf, momentum=momentum)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(nf, momentum=momentum)

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


class TSAFusion(nn.Module):
    # Copy from EDVR (不Copy PCD是因为不会改成MegEngine版)
    """Temporal Spatial Attention (TSA) fusion module.
    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)
    Args:
        nf (int): Channel number of middle features. Default: 64.
        nframes (int): Number of frames. Default: 5.
        cf (int): The index of center frame. Default: 2.
    """

    def __init__(self, nf=32, nframes=5):
        super(TSAFusion, self).__init__()
        self.nframes = nframes
        self.cf = nframes // 2
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(nframes * nf, nf, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(nframes * nf, nf, 1)
        self.spatial_attn2 = nn.Conv2d(nf * 2, nf, 1)
        self.spatial_attn3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(nf, nf, 1)
        self.spatial_attn5 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(nf, nf, 1)
        self.spatial_attn_l2 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(nf, nf, 1)
        self.spatial_attn_add2 = nn.Conv2d(nf, nf, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        if use_mge:
            self.upsample = Upsample(scale_factor=2, mode='BILINEAR')
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        aligned_cf = aligned_feat[:, self.cf, :, :, :].clone()
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding_ref = self.temporal_attn1(aligned_cf)
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d() if use_mge else nn.AdaptiveAvgPool2d(1)
        self.max_pool = AdaptiveMaxPool2d() if use_mge else nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.concat = Concat()
        self.mean = F.mean if use_mge else torch.mean
        self.max = F.max if use_mge else torch.max

    def forward(self, x):
        avgout = self.mean(x, 1, True)
        maxout = self.max(x, 1, True)
        x = self.concat([avgout, maxout], 1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return out

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
    
    def padding(self, tensors):
        if len(tensors) > 2: 
            return tensors
        x , y = tensors
        print(x.shape)
        if self.backend == 'megengine':
            y = torch.tensor(0).reshape(x.shape).set_subtensor(y)
        elif self.backend == 'pytorch':
            xb, xc, xh, xw = x.size()
            yb, yc, yh, yw = y.size()
            diffY = xh - yh
            diffX = xw - yw
            y = F.pad(y, (diffX // 2, diffX - diffX//2, 
                        diffY // 2, diffY - diffY//2))
        return (x, y)

    def forward(self, x, dim=None):
        # x = self.padding(x)
        return self.concat(x, dim if dim is not None else self.dim)

class AdaptiveAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.mean(F.mean(x, axis=-2, keepdims=True), axis=-1, keepdims=True)

class AdaptiveMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.max(F.max(x, axis=-2, keepdims=True), axis=-1, keepdims=True)

if __name__ == '__main__':
    RRDBnet = RRDB()
    up = UpsampleBLock(64, 4)
