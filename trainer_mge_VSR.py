import random
from functools import lru_cache

from datasets import *
from utils import *
from models import *
from base_parser import BaseParser


if __name__ == '__main__':
    parser = BaseParser()
    args = parser.parse()
    device = 'cuda' if torch.is_cuda_available() else 'cpu'
    hostname, root_dir, multi_gpu = get_host_with_dir('/MegVSR')
    model_name = args.model_name
    model_dir = args.checkpoint
    sample_dir = os.path.join(args.result_dir ,f"samples-{model_name}")
    test_dir = "./images/test"
    os.makedirs(sample_dir, exist_ok=True)
    batch_size = args.batch_size
    crop_per_image = args.crop_per_image
    crop_size = args.patch_size
    nflames = args.nframes
    cf = nflames//2   # center_frame
    num_workers = args.num_workers
    step_size = args.step_size
    learning_rate = args.learning_rate
    last_epoch = args.last_epoch
    stop_epoch = args.stop_epoch
    save_freq = 1
    plot_freq = 1
    mode = args.mode
    symbolic = True
    cv2_INTER = True

    net = VSR_RRDB(in_nc=3*nflames, nf=64, nb=6, cv2_INTER=cv2_INTER)
    optimizer = Adam(net.parameters(), lr=learning_rate)

    model = torch.load('VRRDB_5.pkl')
    net = load_weights(net, model['net'])
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
                                    cv2_INTER=cv2_INTER, nflames=nflames, shuffle=True)
        eval_dst = MegVSR_Dataset(root_dir, crop_per_image=crop_per_image, crop_size=crop_size,
                                    mode='eval', cv2_INTER=cv2_INTER, nflames=nflames)

        imgs_lr = torch.tensor(dtype=np.float32)
        imgs_hr = torch.tensor(dtype=np.float32)
        imgs_bc = torch.tensor(dtype=np.float32)

        video_train_list = list(range(train_dst.num_of_videos))
        random.shuffle(video_train_list)
        for epoch in range(last_epoch+1, stop_epoch+1):
            for video_id in video_train_list:
                train_dst.next_video(video_id)
                cnt = 0
                total_loss = 0

                sampler_train = RandomSampler(dataset=train_dst, batch_size=batch_size)
                sampler_eval = SequentialSampler(dataset=eval_dst, batch_size=1)

                dataloader_train = DataLoader(train_dst, sampler=sampler_train, num_workers=num_workers)
                dataloader_eval = DataLoader(eval_dst, sampler=sampler_eval, num_workers=num_workers)

                with tqdm(total=len(dataloader_train)) as t:
                    for k, data in enumerate(dataloader_train):
                        # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                        imgs_lr_np = tensor_dim5to4(data['lr'])
                        imgs_hr_np = tensor_dim5to4(data['hr'])
                        imgs_lr.set_value(imgs_lr_np)
                        imgs_hr.set_value(imgs_hr_np[:,cf*3:cf*3+3,:,:])
                        if cv2_INTER:
                            imgs_bc_np = tensor_dim5to4(data['bc'])
                            imgs_bc.set_value(imgs_bc_np[:,cf*3:cf*3+3,:,:])

                        optimizer.zero_grad()
                        if cv2_INTER:
                            loss, imgs_sr = train_iter(imgs_lr, imgs_hr, imgs_bc)
                        else:
                            loss, imgs_sr = train_iter(imgs_lr, imgs_hr)
                        optimizer.step()

                        # 更新tqdm的参数
                        imgs_sr = F.clamp(imgs_sr, 0, 1)
                        psnr = PSNR_Loss(imgs_sr, imgs_hr)

                        total_loss = total_loss*0.9 + psnr.item()
                        cnt = cnt*0.9 + 1
                        
                        t.set_description(f'Epoch {epoch}, Video {video_id}')
                        t.set_postfix(PSNR=float(f"{total_loss/cnt:.6f}"))
                        t.update(1)

                # epoch过程中存储模型
                if epoch % (save_freq*10) == 9:
                    model_dict = net.module.state_dict() if multi_gpu else net.state_dict()
                    state = {
                        'net': model_dict,
                        'opt': optimizer.state_dict(),
                    }
                    torch.save(state, 'last_model.pkl')
            
            # 更新学习率
            learning_rate *= 0.9
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
                save_path = os.path.join(model_dir, '%s_e%04d.pkl'% (model_name,(epoch//10)*10) )
                torch.save(state, 'last_model.pkl')
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
                        imgs_lr_np = tensor_dim5to4(data['lr'])
                        imgs_hr_np = tensor_dim5to4(data['hr'])
                        imgs_lr.set_value(imgs_lr_np)
                        imgs_hr.set_value(imgs_hr_np[:,cf*3:cf*3+3,:,:])
                        
                        if cv2_INTER:
                            imgs_bc_np = tensor_dim5to4(data['bc'])
                            imgs_bc.set_value(imgs_bc_np[:,cf*3:cf*3+3,:,:])
                            imgs_sr = test_iter(imgs_lr, imgs_bc)
                        else:
                            imgs_sr = test_iter(imgs_lr)
                        
                        img_lr = imgs_lr[0].numpy()[cf*3:cf*3+3,:,:]
                        img_sr = imgs_sr[0].numpy()
                        img_hr = imgs_hr[0].numpy()

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
                                    

    elif mode == 'test':
        test_dst = MegVSR_Test_Dataset(root_dir, cv2_INTER=cv2_INTER, nflames=nflames)
        imgs_lr = torch.tensor(dtype=np.float32)
        imgs_bc = torch.tensor(dtype=np.float32)

        for video_id in range(90, 90+test_dst.num_of_videos):
            test_dst.next_video(video_id)
            sampler_test = SequentialSampler(dataset=test_dst, batch_size=6)
            dataloader_test = DataLoader(test_dst, sampler=sampler_test, num_workers=0)
            with tqdm(total=len(test_dst)) as t:
                for k, data in enumerate(dataloader_test):
                    if video_id < 92: break
                    video_ids = data['video_id']
                    frame_ids = data['frame_id']

                    imgs_lr_np = tensor_dim5to4(data['lr'])
                    imgs_lr.set_value(imgs_lr_np)
                    if cv2_INTER:
                        imgs_bc_np = tensor_dim5to4(data['bc'])
                        imgs_bc.set_value(imgs_bc_np[:,cf*3:cf*3+3,:,:])
                        imgs_sr = test_iter(imgs_lr, imgs_bc)
                    else:
                        imgs_sr = test_iter(imgs_lr)

                    imgs_sr = imgs_sr.numpy()
                    mp_pool = Pool(num_workers)
                    for i in range(imgs_sr.shape[0]):
                        save_dir = os.path.join(test_dir, video_ids[i])
                        # 注意，frame_ids是下标，文件名需要+1
                        mp_pool.apply_async(save_picture, args=(imgs_sr[i],), 
                            kwds={'save_path':save_dir, 'frame_id':"%04d"%(int(frame_ids[i])+1)})
                        # save_picture(imgs_sr[i], save_path=save_dir, frame_id="%04d"%(int(frame_ids[i])+1))

                        # tqdm update
                        t.set_description(f'Video {video_ids[0]}')
                        t.update(1)
                    mp_pool.close()
                    mp_pool.join()
        