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
    model_name = "VRRDB_TSA"
    model_dir = "./saved_model"
    sample_dir = f"./images/{model_name}"
    os.makedirs(sample_dir, exist_ok=True)
    test_dir = "./images/test"
    train_steps = 1000
    batch_size = 8
    crop_per_image = 4
    crop_size = 32
    nflames = 5
    num_workers = 8
    step_size = 2
    learning_rate = 4e-5
    last_epoch = 0
    stop_epoch = 30
    save_freq = 1
    plot_freq = 1
    mode = 'train'
    symbolic = True
    cv2_INTER = False

    net = VSR_RRDB(in_nc=nflames*3,nb=6, cv2_INTER=cv2_INTER)
    optimizer = Adam(net.parameters(), lr=learning_rate)
    # load weight
    model = torch.load('last_model.pth')
    net = load_weights(net, model['net'], by_name=True)
    # optimizer.load_state_dict(model['opt'])

    random.seed(100)
    # 训练
    if mode == 'train':
        gbuffer_train = Global_Buffer(pool_size=15)
        train_dst = MegVSR_Dataset(root_dir, crop_per_image=crop_per_image, crop_size=crop_size,
                            mode='train', cv2_INTER=cv2_INTER, nflames=nflames, global_buffer=gbuffer_train)
        dataloader_train = DataLoader(train_dst, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        gbuffer_eval = Global_Buffer(pool_size=15)
        eval_dst = MegVSR_Dataset(root_dir, crop_per_image=crop_per_image, crop_size=crop_size,
                            mode='eval', cv2_INTER=cv2_INTER, nflames=nflames, global_buffer=gbuffer_eval)
        dataloader_eval = DataLoader(eval_dst, batch_size=1, shuffle=False, num_workers=num_workers)

        scheduler = lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=step_size)
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
                        imgs_lr = torch.split(imgs_lr, 3, dim=1)
                        imgs_lr = torch.stack(imgs_lr, dim=1)
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
                        
            # 更新学习率
            scheduler.step()
            log(f"learning_rate: {scheduler.get_lr()[0]:.6f}")

            # 存储模型
            if epoch % save_freq == 0:
                model_dict = net.module.state_dict() if multi_gpu else net.state_dict()
                state = {
                    'net': model_dict,
                    'opt': optimizer.state_dict(),
                }
                save_path = os.path.join(model_dir, '%s_e%04d.pth'% (model_name,(epoch//10)*10) )
                torch.save(state, 'last_model.pth')
                torch.save(state, save_path)

            # 输出采样
            if epoch % plot_freq == 0:
                net.eval()
                psnrs_bc = np.zeros(len(dataloader_eval), dtype=np.float32)
                ssims_bc = np.zeros(len(dataloader_eval), dtype=np.float32)
                psnrs_sr = np.zeros(len(dataloader_eval), dtype=np.float32)
                ssims_sr = np.zeros(len(dataloader_eval), dtype=np.float32)

                with tqdm(total=len(dataloader_eval)) as t:
                    for k, data in enumerate(dataloader_eval):
                        # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                        frame_id = data['frame_id']
                        imgs_lr = tensor_dim5to4(data['lr'])
                        imgs_lr = torch.split(imgs_lr, 3, dim=1)
                        imgs_lr = torch.stack(imgs_lr, dim=1)
                        imgs_hr = tensor_dim5to4(data['hr'])[:,cf*3:cf*3+3,:,:]
                        imgs_lr = imgs_lr.type(torch.FloatTensor).to(device)
                        imgs_hr = imgs_hr.type(torch.FloatTensor).to(device)
                        with torch.no_grad():
                            imgs_sr = net(imgs_lr)
                            imgs_sr = torch.clamp(imgs_sr, 0, 1)
                            img_lr = imgs_lr[0][cf].detach().cpu().numpy()
                            img_sr = imgs_sr[0].detach().cpu().numpy()
                            img_hr = imgs_hr[0].detach().cpu().numpy()

                        psnr, ssim = plot_sample(img_lr, img_sr, img_hr, frame_id=frame_id[0], epoch=epoch,
                                        save_path=sample_dir, plot_path=sample_dir, model_name=model_name)
                        psnrs_bc[k] = psnr[0]
                        psnrs_sr[k] = psnr[1]
                        ssims_bc[k] = ssim[0]
                        ssims_sr[k] = ssim[1]
                        t.set_description(f'Frame {k}')
                        t.set_postfix(PSNR=float(f"{np.mean(psnrs_sr[:k+1]):.6f}"))
                        t.update(1)
                
                log(f"Epoch {epoch}:\npsnrs_bc={np.mean(psnrs_bc):.2f}, psnrs_sr={np.mean(psnrs_sr):.2f}"
                    +f"\nssims_bc={np.mean(ssims_bc):.4f}, ssims_sr={np.mean(ssims_sr):.4f}", log='log.txt')
                                    
    # 输出测试集
    elif mode == 'test':
        from multiprocessing import Pool
        mp_pool = Pool(num_workers)
        net.eval()
        bs_test = 4
        gbuffer = Global_Buffer(pool_size=15)
        test_dst = MegVSR_Test_Dataset(root_dir, cv2_INTER=False, nflames=nflames, shuffle=True, global_buffer=gbuffer)
        dataloader_test = DataLoader(test_dst, batch_size=bs_test, shuffle=False, num_workers=4)
        net = net.to(device)
        for video_id in range(90, 90+test_dst.num_of_videos):
            test_dst.next_video(video_id-90)
            with tqdm(total=len(test_dst)) as t:
                for k, data in enumerate(dataloader_test):
                    video_ids = data['video_id']
                    frame_ids = data['frame_id']
                    
                    with torch.no_grad():
                        imgs_lr = tensor_dim5to4(data['lr'])
                        imgs_lr = torch.split(imgs_lr, 3, dim=1)
                        imgs_lr = torch.stack(imgs_lr, dim=1)
                        imgs_lr = imgs_lr.type(torch.FloatTensor).to(device)
                        imgs_sr = net(imgs_lr)
                        imgs_sr = torch.clamp(imgs_sr, 0, 1)
                        imgs_sr = imgs_sr.detach().cpu().numpy()
                    
                    for i in range(imgs_sr.shape[0]):
                        save_dir = os.path.join(test_dir, video_ids[i])
                        # 注意，frame_ids是下标，文件名需要+1
                        mp_pool.apply_async(save_picture, args=(imgs_sr[i],), 
                            kwds={'save_path':save_dir, 'frame_id':"%04d"%(int(frame_ids[i])+1)})
                        # save_picture(imgs_sr[i], save_path=save_dir, frame_id="%04d"%(int(frame_ids[i])+1))

                        # tqdm update
                        t.set_description(f'Video {video_ids[0]}')
                        t.update(1)