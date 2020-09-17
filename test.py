import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim

class AdaptiveAvgPool2d(M.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.mean(F.mean(x, axis=-2, keepdims=True), axis=-1, keepdims=True)

class AdaptiveMaxPool2d(M.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.max(F.max(x, axis=-2, keepdims=True), axis=-1, keepdims=True)

class ChannelAttention(M.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d()
        self.max_pool = AdaptiveMaxPool2d()
        self.sharedMLP = M.Sequential(
            M.Conv2d(in_planes, in_planes // ratio, 1, bias=False), M.ReLU(),
            M.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = M.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(M.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = M.Conv2d(2,1,kernel_size, padding=1, bias=False)
        self.sigmoid = M.Sigmoid()
        self.concat = F.concat
        self.mean = F.mean
        self.max = F.max

    def forward(self, x):
        avgout = self.mean(x, 1, True)
        maxout = self.max(x, 1, True)
        x = self.concat([avgout, maxout], 1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(M.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return out


if __name__ == "__main__":
    data = mge.tensor(np.random.random((1, 16, 10, 10)).astype(np.float32))
    model = CBAM(16)
    opt = optim.SGD(model.parameters(), lr=0.1)

    for i in range(5):
        opt.zero_grad()
        loss = model(data).mean()
        opt.backward(loss)
        opt.step()
        print("loss = {:.3f}".format(loss.numpy()[0]))