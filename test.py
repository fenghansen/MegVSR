import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim


class SpatialAttention(M.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = M.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = M.Sigmoid()
        self.concat = M.Concat()
        self.mean = F.mean 
        self.max = F.max 

    def forward(self, x):
        avgout = self.mean(x, 1, True)
        maxout = self.max(x, 1, True)
        x = self.concat([avgout, maxout], 1)
        x = self.conv(x)
        return self.sigmoid(x)


if __name__ == "__main__":
    data = mge.tensor(np.random.random((1, 2, 10, 10)).astype(np.float32))
    model = SpatialAttention()
    opt = optim.SGD(model.parameters(), lr=0.1)

    for i in range(5):
        opt.zero_grad()
        loss = model(data).mean()
        opt.backward(loss)
        opt.step()
        print("loss = {:.3f}".format(loss.numpy()[0]))