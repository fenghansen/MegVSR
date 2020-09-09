import os
import time
import random
from functools import lru_cache
from torch.optim import Adam, lr_scheduler

from datasets import *
from utils import *
from models import *
from losses import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hostname, root_dir, multi_gpu = get_host_with_dir('/MegVSR')
    model_name = "VRRDB_6"
    model_dir = "./saved_model"
    sample_dir = "./images/samples-video"
    test_dir = "./images/test"
    train_steps = 1000
    batch_size = 8
    crop_per_image = 4
    crop_size = 32
    nflames = 3
    num_workers = 0
    step_size = 2
    learning_rate = 1e-4
    last_epoch = 0
    stop_epoch = 20
    save_freq = 1
    plot_freq = 1
    mode = 'train'
    symbolic = True
    cv2_INTER = False

    net = VSR_RRDB(in_nc=9,nb=6, cv2_INTER=cv2_INTER)
    optimizer = Adam(net.parameters(), lr=learning_rate)
    # load weight
    # model = torch.load('last_model.pth')
    # net.load_state_dict(model['net'])
    # optimizer.load_state_dict(model['opt'])

    random.seed(100)
    # 训练
    if mode == 'train':
        train_dst = MegVSR_Dataset(root_dir, crop_per_image=crop_per_image, crop_size=crop_size,
                            mode='train', cv2_INTER=cv2_INTER, nflames=nflames)
        dataloader_train = DataLoader(train_dst, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        eval_dst = MegVSR_Dataset(root_dir, crop_per_image=crop_per_image, crop_size=crop_size,
                            mode='eval', cv2_INTER=cv2_INTER, nflames=nflames)
        dataloader_eval = DataLoader(eval_dst, batch_size=1, shuffle=False, num_workers=num_workers)

        scheduler = lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=step_size)
        net = net.to(device)
        net.train()
        for epoch in range(last_epoch+1, stop_epoch+1):
            for video_id in range(train_dst.num_of_videos):
                train_dst.next_video(video_id)
                cnt = 0
                total_loss = 0
                cf = nflames//2   # center_frame

                with tqdm(total=len(dataloader_train)) as t:
                    for k, data in enumerate(dataloader_train):
                        # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                        imgs_lr = tensor_dim5to4(data['lr'])
                        imgs_hr = tensor_dim5to4(data['hr'])[:,cf*3:cf*3+3,:,:]
                        
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
                        
                        total_loss = total_loss*0.9 + psnr.item()
                        cnt = cnt*0.9 + 1
                        t.set_description(f'Epoch {epoch}, Video {video_id}')
                        t.set_postfix(PSNR=float(f"{total_loss/cnt:.6f}"))
                        t.update(1)
                        break
                if video_id==0: break
                        
            # 更新学习率
            scheduler.step()
            log(f"learning_rate: {scheduler.get_lr()[0]:.6f}")
            # 输出采样
            if epoch % plot_freq == 0:
                net.eval()
                with tqdm(total=len(dataloader_eval)) as t:
                    for k, data in enumerate(dataloader_eval):
                        # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                        frame_id = data['frame_id']
                        imgs_lr = tensor_dim5to4(data['lr'])
                        imgs_hr = tensor_dim5to4(data['hr'])[:,cf*3:cf*3+3,:,:]
                        imgs_lr = imgs_lr.type(torch.FloatTensor).to(device)
                        imgs_hr = imgs_hr.type(torch.FloatTensor).to(device)
                        with torch.no_grad():
                            imgs_sr = net(imgs_lr)
                            imgs_sr = torch.clamp(imgs_sr, 0, 1)
                            img_lr = imgs_lr[0][cf*3:cf*3+3].detach().cpu().numpy()
                            img_sr = imgs_sr[0].detach().cpu().numpy()
                            img_hr = imgs_hr[0].detach().cpu().numpy()

                        t.set_description(f'Frame {k}')
                        t.update(1)

                        plot_sample(img_lr, img_sr, img_hr, frame_id=frame_id[0], epoch=epoch,
                                    save_path=sample_dir, plot_path=sample_dir, model_name='tiny_RRDB')
                                    
            # 存储模型
            if epoch % save_freq == 0:
                model_dict = net.module.state_dict() if multi_gpu else net.state_dict()
                state = {
                    'net': model_dict,
                    'opt': optimizer.state_dict(),
                }
                save_path = os.path.join(model_dir, 'model.torch.state_e%04d'%((epoch//10) * 10))
                torch.save(state, save_path)
                torch.save(state, 'last_model.pth')
    # 输出测试集
    elif mode == 'test':
        net.eval()
        bs_test = 4
        test_dst = MegVSR_Test_Dataset(root_dir)
        dataloader_test = DataLoader(test_dst, batch_size=bs_test, shuffle=False, num_workers=2)
        net = net.to(device)
        for video_id in range(90, 90+test_dst.num_of_videos):
            test_dst.next_video(video_id)
            with tqdm(total=len(test_dst)) as t:
                for k, data in enumerate(dataloader_test):
                    video_id = data['video_id']
                    frame_id = data['frame_id']
                    
                    with torch.no_grad():
                        imgs_lr = tensor_dim5to4(data['lr'])
                        imgs_lr = imgs_lr.type(torch.FloatTensor).to(device)
                        imgs_sr = net(imgs_lr)
                        imgs_sr = torch.clamp(imgs_sr, 0, 1)
                        imgs_sr = imgs_sr.detach().cpu().numpy()
                    
                    for i in range(imgs_sr.shape[0]):
                        save_dir = os.path.join(test_dir, video_id[i])
                        # 注意，frame_id是下标，文件名需要+1
                        save_picture(imgs_sr[i], save_path=save_dir, frame_id="%04d"%(int(frame_id[i])+1))

                        # tqdm update
                        t.set_description(f'Video {video_id[0]}')
                        t.update(1)