import os
import time
import random
from functools import lru_cache
from torch.optim import Adam

from utils import *
from models import *

TRAIN_RAW_DATA = "F:/datasets/MegVSR/train.tar"
TEST_RAW_DATA = "F:/datasets/MegVSR/test.tar"
TRAIN_DATA_STORAGE = "./train_patches"
MODEL_PATH = "model.mge.state"
train_steps = 1000
batch_size = 8
input_h = 256
input_w = 256

net = DeepUnet()
optimizer = Adam(net.parameters(), lr=1e-4)

random.seed(100)

@lru_cache(maxsize=None)
def load_image(path):
    return np.load(path, mmap_mode="r")

train_patches = sorted([os.path.join(TRAIN_DATA_STORAGE, f) for f in os.listdir(TRAIN_DATA_STORAGE)])

def load_batch():
    batch_train = []
    batch_gt = []
    for i in range(batch_size):
        path = random.choice(train_patches)
        img = load_image(path)
        batch_train.append(img[0])
        batch_gt.append(img[1])
    return np.array(batch_train), np.array(batch_gt)

def train_iter(batch_train, batch_gt):
    pred = net(batch_train)
    loss = F.abs(batch_gt - pred).mean()
    optimizer.backward(loss)
    return loss, pred

loss_acc = 0
loss_acc0 = 0

for it in range(train_steps + 1):
    for g in optimizer.param_groups:
        g['lr'] = 2e-4 * (train_steps - it) / train_steps

    begin = time.time()
    (batch_train, batch_gt) = load_batch()
    data_load_end = time.time()

    optimizer.zero_grad()
    loss, pred = train_iter(batch_train, batch_gt)
    optimizer.step()
    loss_acc = loss_acc * 0.99 + loss
    loss_acc0 = loss_acc0 * 0.99 + 1
    end = time.time()
    
    total_time = end - begin
    data_load_time = data_load_end - begin
    if it % 100 == 0:
        print(
            "{}: loss: {}, speed: {:.2f}it/sec, tot: {:.4f}s, data: {:.4f}s, data/tot: {:.4f}"
            .format(it, loss_acc / loss_acc0, 1 / total_time, total_time,
                    data_load_time, data_load_time / total_time))

# 存储模型
state = {
    'net': net.state_dict(),
    'opt': optimizer.state_dict(),
}
with open(MODEL_PATH, 'wb') as fout:
    torch.save(state, fout)