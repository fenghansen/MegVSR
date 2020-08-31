import os
import time
import random
from functools import lru_cache
from torch import optim

from datasets import *
from utils import *
from models import *
from losses import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hostname, root_dir, multi_gpu = get_host_with_dir('/MegVSR')
    model_dir = "./saved_model"
    train_steps = 1000
    batch_size = 8
    crop_per_image = 8
    crop_size = 64
    num_workers = 2
    step_size = 50
    learning_rate = 1e-4
    lastepoch = 0
    save_freq = 1
    plog_freq = 1

    net = RRDBNet()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    random.seed(100)

    MegVSR_dst = MegVSR_Dataset(root_dir, crop_per_image=crop_per_image)
    dataloader_train = DataLoader(MegVSR_dst, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=step_size)
    net = net.to(device)

    for epoch in range(lastepoch+1, 501):
        for video_id in range(MegVSR_dst.num_of_videos):
            MegVSR_dst.video_id = video_id
            cnt = 0
            total_loss = 0
            with tqdm(total=len(dataloader_train)) as t:
                for k, data in enumerate(dataloader_train):
                    # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                    imgs_lr = tensor_dim5to4(data['lr'])
                    imgs_hr = tensor_dim5to4(data['hr'])
                    imgs_lr = imgs_lr.type(torch.FloatTensor).to(device)
                    imgs_hr = imgs_hr.type(torch.FloatTensor).to(device)
                    data_load_end = time.time()

                    optimizer.zero_grad()
                    pred = net(imgs_lr)
                    loss = F.l1_loss(pred, imgs_hr)

                    loss.backward()
                    optimizer.step()

                    # 更新tqdm的参数
                    with torch.no_grad():
                        pred = torch.clamp(pred, 0, 1)
                        psnr = PSNR_Loss(pred, imgs_hr)
                    total_loss += psnr.item()

                    cnt += 1
                    t.set_description(f'Epoch {epoch}, Video {video_id}')
                    t.set_postfix(PSNR=float(f"{total_loss/cnt:.6f}"))
                    t.update(1)
                    
        # 更新学习率
        scheduler.step()
        # 存储模型
        if epoch % save_freq == 0:
            model_dict = net.module.state_dict() if multi_gpu else net.state_dict()
            log(f"learning_rate: {scheduler.get_lr()[0]:.6f}")
            state = {
                'net': model_dict,
                'opt': optimizer.state_dict(),
            }
            save_path = os.path.join(model_dir, 'model.torch.state_e%04d'%((epoch//10) * 10))
            torch.save(state, save_path)