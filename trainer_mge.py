import os
import time
import random
from functools import lru_cache

from datasets import *
from utils import *
from models import *

if __name__ == '__main__':
    device = 'cuda' if torch.is_cuda_available() else 'cpu'
    hostname, root_dir, multi_gpu = get_host_with_dir('/MegVSR')
    model_name = "RRDB_6"
    model_dir = "./saved_model"
    sample_dir = f"./images/samples-{model_name}"
    test_dir = "./images/test"
    os.makedirs(sample_dir, exist_ok=True)
    train_steps = 1000
    batch_size = 4
    crop_per_image = 4
    crop_size = 64
    nflames = 1
    num_workers = 4
    step_size = 2
    learning_rate = 1e-5
    last_epoch = 20
    stop_epoch = 30
    save_freq = 1
    plot_freq = 1
    mode = 'train'
    symbolic = True
    cv2_INTER = True

    net = SR_RRDB(nf=64, nb=6, cv2_INTER=cv2_INTER)
    optimizer = Adam(net.parameters(), lr=learning_rate)

    model = torch.load('last_model.pkl')
    net.load_state_dict(model['net'])
    optimizer.load_state_dict(model['opt'])
    for g in optimizer.param_groups:
        g['lr'] = learning_rate
    log(f"learning_rate: {learning_rate:.6f}")

    random.seed(100)

    @trace(symbolic=symbolic)
    def train_iter(imgs_lr, imgs_hr, imgs_bc=None):
        net.train()
        imgs_sr = net(imgs_lr)
        if imgs_bc is not None:
            imgs_sr = imgs_sr + imgs_bc
        loss = F.l1_loss(imgs_hr, imgs_sr)
        optimizer.backward(loss)
        imgs_sr = F.clamp(imgs_sr, 0, 1)
        return loss, imgs_sr
    
    @trace(symbolic=symbolic)
    def test_iter(imgs_lr, imgs_bc=None):
        net.eval()
        imgs_sr = net(imgs_lr)
        if imgs_bc is not None:
            imgs_sr = imgs_sr + imgs_bc
        imgs_sr = F.clamp(imgs_sr, 0, 1)
        return imgs_sr
    
    @trace(symbolic=symbolic)
    def PSNR_Loss(low, high):
        return -10.0 * F.log(F.mean(F.power(high-low, 2))) / F.log(torch.tensor(10.0))

    if mode == 'train':
        train_dst = MegVSR_Dataset(root_dir, crop_per_image=crop_per_image, crop_size=crop_size,
                                    cv2_INTER=cv2_INTER, nflames=nflames)
        eval_dst = MegVSR_Dataset(root_dir, crop_per_image=crop_per_image, crop_size=crop_size,
                                    mode='eval', cv2_INTER=cv2_INTER, nflames=nflames)

        imgs_lr = torch.tensor(dtype=np.float32)
        imgs_hr = torch.tensor(dtype=np.float32)
        imgs_bc = torch.tensor(dtype=np.float32)

        for epoch in range(last_epoch+1, stop_epoch+1):
            for video_id in range(train_dst.num_of_videos):
                train_dst.next_video(video_id)

                sampler_train = RandomSampler(dataset=train_dst, batch_size=batch_size)
                sampler_eval = SequentialSampler(dataset=eval_dst, batch_size=1)

                dataloader_train = DataLoader(train_dst, sampler=sampler_train, num_workers=num_workers)
                dataloader_eval = DataLoader(eval_dst, sampler=sampler_eval, num_workers=num_workers)

                cnt = 0
                total_loss = 0
                center_frame = nflames//2

                with tqdm(total=len(dataloader_train)) as t:
                    for k, data in enumerate(dataloader_train):
                        # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                        imgs_lr_np = tensor_dim5to4(data['lr'])
                        imgs_hr_np = tensor_dim5to4(data['hr'])
                        imgs_lr.set_value(imgs_lr_np)
                        imgs_hr.set_value(imgs_hr_np[:,center_frame*3:center_frame*3+3,:,:])
                        if cv2_INTER:
                            imgs_bc_np = tensor_dim5to4(data['bc'])
                            imgs_bc.set_value(imgs_bc_np[:,center_frame*3:center_frame*3+3,:,:])

                        optimizer.zero_grad()
                        if cv2_INTER:
                            loss, imgs_sr = train_iter(imgs_lr, imgs_hr, imgs_bc)
                        else:
                            loss, imgs_sr = train_iter(imgs_lr, imgs_hr)
                        optimizer.step()

                        # 更新tqdm的参数
                        imgs_sr = F.clamp(imgs_sr, 0, 1)
                        psnr = PSNR_Loss(imgs_sr, imgs_hr)

                        total_loss += psnr.item()

                        cnt += 1
                        t.set_description(f'Epoch {epoch}, Video {video_id}')
                        t.set_postfix(PSNR=float(f"{total_loss/cnt:.6f}"))
                        t.update(1)
            
            # 更新学习率
            learning_rate *= 0.8
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
            log(f"learning_rate: {learning_rate:.6f}")

            # 存储模型
            if epoch % save_freq == 0:
                model_dict = net.module.state_dict() if multi_gpu else net.state_dict()
                state = {
                    'net': model_dict,
                    'opt': optimizer.state_dict(),
                }
                save_path = os.path.join(model_dir, 'RRDB_6.mge.state_e%04d'% ((epoch//10)*10) )
                torch.save(state, 'last_model.pkl')
                torch.save(state, save_path)

            # 输出采样
            if epoch % plot_freq == 0:
                net.eval()
                with tqdm(total=len(dataloader_eval)) as t:
                    for k, data in enumerate(dataloader_eval):
                        # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                        frame_id = data['frame_id']
                        imgs_lr_np = tensor_dim5to4(data['lr'])
                        imgs_hr_np = tensor_dim5to4(data['hr'])
                        imgs_lr.set_value(imgs_lr_np)
                        imgs_hr.set_value(imgs_hr_np[:,center_frame*3:center_frame*3+3,:,:])
                        
                        if cv2_INTER:
                            imgs_bc_np = tensor_dim5to4(data['bc'])
                            imgs_bc.set_value(imgs_bc_np[:,center_frame*3:center_frame*3+3,:,:])
                            imgs_sr = test_iter(imgs_lr, imgs_bc)
                        else:
                            imgs_sr = test_iter(imgs_lr)
                        
                        img_lr = imgs_lr[0].numpy()[center_frame*3:center_frame*3+3,:,:]
                        img_sr = imgs_sr[0].numpy()
                        img_hr = imgs_hr[0].numpy()

                        t.set_description(f'Frame {k}')
                        t.update(1)

                        plot_sample(img_lr, img_sr, img_hr, frame_id=frame_id[0], epoch=epoch,
                                    save_path=sample_dir, plot_path=sample_dir, model_name=model_name)
                                    

    elif mode == 'test':
        test_dst = MegVSR_Test_Dataset(root_dir, cv2_INTER=cv2_INTER, nflames=nflames)
        imgs_lr = torch.tensor(dtype=np.float32)
        imgs_bc = torch.tensor(dtype=np.float32)

        for video_id in range(90, 90+test_dst.num_of_videos):
            test_dst.next_video(video_id)
            sampler_test = SequentialSampler(dataset=test_dst, batch_size=6)
            dataloader_test = DataLoader(test_dst, sampler=sampler_test, num_workers=num_workers)
            with tqdm(total=len(test_dst)) as t:
                for k, data in enumerate(dataloader_test):
                    video_id = data['video_id']
                    frame_id = data['frame_id']

                    imgs_lr_np = tensor_dim5to4(data['lr'])
                    imgs_lr.set_value(imgs_lr_np)
                    if cv2_INTER:
                        imgs_bc_np = tensor_dim5to4(data['bc'])
                        imgs_bc.set_value(imgs_bc_np)
                        imgs_sr = test_iter(imgs_lr, imgs_bc)
                    else:
                        imgs_sr = test_iter(imgs_lr)

                    imgs_sr = imgs_sr.numpy()

                    for i in range(imgs_sr.shape[0]):
                        save_dir = os.path.join(test_dir, video_id[i])
                        # 注意，frame_id是下标，文件名需要+1
                        save_picture(imgs_sr[i], save_path=save_dir, frame_id="%04d"%(int(frame_id[i])+1))

                        # tqdm update
                        t.set_description(f'Video {video_id[0]}')
                        t.update(1)
        